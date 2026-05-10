import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from src.config import Settings
from src.dedup_service import DedupService
from src.image_io import image_ref_for_openai
from src.milvus_store import MilvusDedupStore
from src.triton_clients import TritonClassificationClient, TritonEmbeddingClient


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pipeline-fastapi")


def normalize_images_payload(
    request: dict[str, Any],
    max_images_per_request: int,
) -> tuple[str, list[dict[str, str]]]:
    request_id = str(request.get("request_id") or uuid.uuid4())

    if "images" in request:
        images = request["images"]
    elif "image_base64" in request or "image_url" in request:
        images = [
            {
                "image_id": request.get("image_id"),
                "image_base64": request.get("image_base64"),
                "image_url": request.get("image_url"),
            }
        ]
    else:
        raise ValueError("Request must contain 'images', 'image_base64', or 'image_url'.")

    if not isinstance(images, list):
        raise ValueError("'images' must be a list.")
    if len(images) > max_images_per_request:
        raise ValueError(f"Too many images. Max is {max_images_per_request}.")

    items = []
    for idx, item in enumerate(images):
        if isinstance(item, str):
            image_id = f"{request_id}:{idx}"
            image_ref = item
        elif isinstance(item, dict):
            image_id = str(item.get("image_id") or f"{request_id}:{idx}")
            image_ref = (
                item.get("image_base64")
                or item.get("b64")
                or item.get("image_url")
                or item.get("url")
            )
        else:
            raise ValueError("Each image must be a string or object.")

        if not image_ref or not isinstance(image_ref, str):
            raise ValueError("Each image must contain image_base64 or image_url.")

        items.append(
            {
                "request_id": request_id,
                "image_id": image_id,
                "image_ref": image_ref,
            }
        )

    return request_id, items


class OpenAIVisionClient:
    def __init__(self, model: str, timeout: float, proxy: str | None = None):
        from openai import DefaultHttpxClient, OpenAI

        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if proxy:
            client_kwargs["http_client"] = DefaultHttpxClient(proxy=proxy)

        self.client = OpenAI(**client_kwargs)
        self.model = model

    def analyze(
        self,
        image_ref: str,
        prompt: str,
        classification: dict[str, Any],
    ) -> dict[str, Any]:
        import json

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"{prompt}\n\n"
                                "Triton classification result:\n"
                                f"{json.dumps(classification, ensure_ascii=False)}"
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_ref,
                            "detail": "auto",
                        },
                    ],
                }
            ],
        )

        return {
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "output_text": getattr(response, "output_text", None),
        }


@dataclass
class PipelineRuntime:
    cfg: Settings
    dedup_service: DedupService
    classifier: TritonClassificationClient
    openai_client: OpenAIVisionClient


def build_runtime() -> PipelineRuntime:
    cfg = Settings.from_env()

    embedder = TritonEmbeddingClient(
        url=cfg.triton_client_url,
        model_name=cfg.dedup_model_name,
        input_name=cfg.dedup_input_name,
        output_name=cfg.dedup_output_name,
        image_size=cfg.dedup_image_size,
        embedding_dim=cfg.dedup_embedding_dim,
        timeout=cfg.request_timeout,
    )
    cfg.dedup_embedding_dim = embedder.embedding_dim

    classifier = TritonClassificationClient(
        url=cfg.triton_client_url,
        model_name=cfg.classification_model_name,
        input_name=cfg.classification_input_name,
        output_name=cfg.classification_output_name,
        image_size=cfg.classification_image_size,
        labels=cfg.classification_labels,
        timeout=cfg.request_timeout,
    )

    store = MilvusDedupStore(cfg)
    dedup_service = DedupService(
        embedder=embedder,
        store=store,
        dup_threshold=cfg.dup_threshold,
    )
    dedup_service.prepare()

    openai_client = OpenAIVisionClient(
        model=cfg.openai_model,
        timeout=cfg.openai_timeout,
        proxy=cfg.openai_proxy,
    )

    return PipelineRuntime(
        cfg=cfg,
        dedup_service=dedup_service,
        classifier=classifier,
        openai_client=openai_client,
    )


def health(runtime: PipelineRuntime) -> bool:
    try:
        return (
            runtime.dedup_service.health()
            and runtime.classifier.is_ready()
            and runtime.openai_client is not None
        )
    except Exception:
        logger.exception("Health check failed.")
        return False


def is_fake_classification(classification: dict[str, Any]) -> bool:
    """Return True only when Triton classification label is fake."""
    label = classification.get("label")
    return isinstance(label, str) and label.strip().lower() == "fake"


# Patch đề xuất cho pipeline_server.py
# Mục tiêu: log thời gian chạy riêng cho dedup, classification, và OpenAI VLM.
# Ghi chú: OpenAI VLM chỉ được gọi khi classification label == fake.


def run_pipeline_on_items(
    runtime: PipelineRuntime,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dedup_started = time.perf_counter()
    dedup_results, unique_records = runtime.dedup_service.deduplicate(items)
    dedup_elapsed = time.perf_counter() - dedup_started

    logger.info(
        "Dedup completed: total_images=%s unique_records=%s elapsed_seconds=%.4f",
        len(items),
        len(unique_records),
        dedup_elapsed,
    )

    unique_by_flat_idx = {record["flat_idx"]: record for record in unique_records}

    classification_by_flat_idx: dict[int, dict[str, Any]] = {}
    if unique_records:
        classification_started = time.perf_counter()
        classify_results = runtime.classifier.classify([x["image"] for x in unique_records])
        classification_elapsed = time.perf_counter() - classification_started

        logger.info(
            "Classification completed: unique_records=%s elapsed_seconds=%.4f avg_seconds_per_image=%.4f",
            len(unique_records),
            classification_elapsed,
            classification_elapsed / len(unique_records),
        )

        for record, classify_result in zip(unique_records, classify_results):
            classification_by_flat_idx[record["flat_idx"]] = classify_result
    else:
        logger.info("Classification skipped: unique_records=0")

    openai_call_count = 0
    openai_total_elapsed = 0.0

    pipeline_results: list[dict[str, Any]] = []
    for idx, (item, dedup_result) in enumerate(zip(items, dedup_results)):
        if dedup_result.get("status") == "duplicate":
            pipeline_results.append(
                {
                    "image_id": item["image_id"],
                    "status": "skipped",
                    "stage": "dedup",
                    "dedup": dedup_result,
                }
            )
            continue

        if dedup_result.get("status") != "unique":
            pipeline_results.append(
                {
                    "image_id": item["image_id"],
                    "status": "error",
                    "stage": "dedup",
                    "dedup": dedup_result,
                    "error": dedup_result.get("error"),
                }
            )
            continue

        classification = classification_by_flat_idx.get(idx)
        if not classification or classification.get("status") != "classified":
            pipeline_results.append(
                {
                    "image_id": item["image_id"],
                    "status": "error",
                    "stage": "classification",
                    "dedup": dedup_result,
                    "classification": classification,
                }
            )
            continue

        # Only fake images should be sent to OpenAI VLM.
        # Real images stop here after Triton classification.
        if not is_fake_classification(classification):
            logger.info(
                "OpenAI VLM skipped: image_id=%s classification_label=%s reason=classification_label_is_not_fake",
                item["image_id"],
                classification.get("label"),
            )
            pipeline_results.append(
                {
                    "image_id": item["image_id"],
                    "status": "completed",
                    "stage": "classification",
                    "dedup": dedup_result,
                    "classification": classification,
                    "vlm": None,
                    "openai_skipped": True,
                    "skip_reason": "classification_label_is_not_fake",
                }
            )
            continue

        record = unique_by_flat_idx[idx]
        try:
            openai_started = time.perf_counter()
            vlm_result = runtime.openai_client.analyze(
                image_ref=image_ref_for_openai(record["image_ref"]),
                prompt=runtime.cfg.vlm_prompt,
                classification=classification,
            )
            openai_elapsed = time.perf_counter() - openai_started
            openai_call_count += 1
            openai_total_elapsed += openai_elapsed

            logger.info(
                "OpenAI VLM completed: image_id=%s model=%s elapsed_seconds=%.4f",
                item["image_id"],
                runtime.cfg.openai_model,
                openai_elapsed,
            )
        except Exception as exc:
            openai_elapsed = time.perf_counter() - openai_started
            logger.exception(
                "OpenAI VLM failed: image_id=%s model=%s elapsed_seconds=%.4f",
                item["image_id"],
                runtime.cfg.openai_model,
                openai_elapsed,
            )
            pipeline_results.append(
                {
                    "image_id": item["image_id"],
                    "status": "error",
                    "stage": "openai",
                    "dedup": dedup_result,
                    "classification": classification,
                    "error": str(exc),
                }
            )
            continue

        pipeline_results.append(
            {
                "image_id": item["image_id"],
                "status": "completed",
                "stage": "openai",
                "dedup": dedup_result,
                "classification": classification,
                "vlm": vlm_result,
            }
        )

    if openai_call_count:
        logger.info(
            "OpenAI VLM summary: calls=%s total_elapsed_seconds=%.4f avg_seconds_per_call=%.4f",
            openai_call_count,
            openai_total_elapsed,
            openai_total_elapsed / openai_call_count,
        )
    else:
        logger.info("OpenAI VLM summary: calls=0 total_elapsed_seconds=0.0000")

    return pipeline_results




def build_response(
    runtime: PipelineRuntime,
    request_id: str,
    item_results: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    dedup_statuses = [
        result.get("dedup", {}).get("status")
        for result in item_results
        if isinstance(result.get("dedup"), dict)
    ]
    return {
        "request_id": request_id,
        "total_images": len(item_results),
        "unique_count": sum(1 for status in dedup_statuses if status == "unique"),
        "duplicate_count": sum(1 for status in dedup_statuses if status == "duplicate"),
        "completed_count": sum(1 for result in item_results if result.get("status") == "completed"),
        "error_count": sum(1 for result in item_results if result.get("status") == "error"),
        "results": item_results,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "milvus_recovered_this_predict": runtime.dedup_service.store.recovered_this_predict,
        "last_milvus_error": runtime.dedup_service.store.last_error,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Triton pipeline FastAPI server starting.")
    app.state.runtime = await run_in_threadpool(build_runtime)

    import asyncio

    # MilvusDedupStore has per-request mutable state:
    # recovered_this_predict and last_error.
    # This lock keeps /analyze safe without changing your existing store code.
    app.state.analyze_lock = asyncio.Lock()
    logger.info("Triton pipeline worker setup completed.")

    try:
        yield
    finally:
        try:
            from pymilvus import connections

            connections.disconnect(alias="default")
        except Exception:
            logger.exception("Failed to disconnect Milvus cleanly.")


app = FastAPI(
    title="Triton Dedup Classification OpenAI API",
    version="1.0.0",
    lifespan=lifespan,
)


def get_runtime(request: Request) -> PipelineRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not initialized.")
    return runtime


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    runtime = get_runtime(request)
    is_healthy = await run_in_threadpool(health, runtime)
    if not is_healthy:
        raise HTTPException(
            status_code=503,
            detail="Triton, Milvus, or OpenAI runtime is not ready.",
        )
    return {"status": "ok"}


@app.get("/health")
async def health_alias(request: Request) -> dict[str, Any]:
    return await healthz(request)


@app.post("/analyze")
async def analyze(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = get_runtime(request)

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be JSON object.")

    try:
        request_id, items = normalize_images_payload(
            payload,
            runtime.cfg.max_images_per_request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    started = time.perf_counter()

    async with request.app.state.analyze_lock:
        runtime.dedup_service.store.reset_predict_state()
        item_results = await run_in_threadpool(run_pipeline_on_items, runtime, items)

    return build_response(
        runtime=runtime,
        request_id=request_id,
        item_results=item_results,
        elapsed_seconds=time.perf_counter() - started,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "pipeline_server:app",
        host=os.getenv("HOST", os.getenv("API_HOST", "127.0.0.1")),
        port=int(
            os.getenv(
                "PIPELINE_PORT",
                os.getenv("API_PORT", os.getenv("PORT", "8002")),
            )
        ),
        workers=int(os.getenv("FASTAPI_WORKERS", "1")),
        timeout_keep_alive=int(os.getenv("FASTAPI_TIMEOUT_KEEP_ALIVE", "75")),
    )


if __name__ == "__main__":
    main()

