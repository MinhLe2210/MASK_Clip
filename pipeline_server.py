import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from src.config import Settings
from src.dedup_service import DedupService
from src.image_io import (
    bytes_from_ref,
    bytes_to_data_url,
    bytes_to_pil,
    guess_image_mime,
    image_ref_for_openai,
)
from src.milvus_store import MilvusDedupStore
from src.nfa_vit import TritonNfaVitClient
from src.triton_clients import TritonClassificationClient, TritonEmbeddingClient


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pipeline-fastapi")


def upload_to_image_ref(content: bytes, filename: str | None, content_type: str | None) -> str:
    mime = content_type if content_type and content_type.startswith("image/") else None
    if not mime:
        mime = guess_image_mime(content, filename)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def normalize_images_payload(
    request: dict[str, Any],
    max_images_per_request: int,
) -> tuple[str, list[dict[str, str]]]:
    request_id = str(request.get("request_id") or uuid.uuid4())

    if "images" in request:
        images = request["images"]
    elif (
        "image_base64" in request
        or "image_url" in request
        or "image_path" in request
        or "path" in request
        or "file_path" in request
    ):
        images = [
            {
                "image_id": request.get("image_id"),
                "image_base64": request.get("image_base64"),
                "image_url": request.get("image_url"),
                "image_path": request.get("image_path"),
                "path": request.get("path"),
                "file_path": request.get("file_path"),
            }
        ]
    else:
        raise ValueError(
            "Request must contain 'images', 'image_base64', 'image_url', or 'image_path'."
        )

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
                or item.get("image_path")
                or item.get("path")
                or item.get("file_path")
            )
        else:
            raise ValueError("Each image must be a string or object.")

        if not image_ref or not isinstance(image_ref, str):
            raise ValueError(
                "Each image must contain image_base64, image_url, image_path, path, or file_path."
            )

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
        detector_context: dict[str, Any],
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
                                "Internal detector context:\n"
                                f"{json.dumps(detector_context, ensure_ascii=False)}"
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
    nfa_vit_client: TritonNfaVitClient
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
    nfa_vit_client = TritonNfaVitClient(
        url=cfg.triton_client_url,
        model_name=cfg.nfa_model_name,
        image_input_name=cfg.nfa_image_input_name,
        mask_input_name=cfg.nfa_mask_input_name,
        label_input_name=cfg.nfa_label_input_name,
        mask_output_name=cfg.nfa_mask_output_name,
        label_output_name=cfg.nfa_label_output_name,
        image_size=cfg.nfa_image_size,
        threshold=cfg.nfa_threshold,
        white_ratio_threshold=cfg.nfa_white_ratio_threshold,
        label_value=cfg.nfa_label_value,
        preprocess_mode=cfg.nfa_preprocess_mode,
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
        nfa_vit_client=nfa_vit_client,
        openai_client=openai_client,
    )


def health(runtime: PipelineRuntime) -> bool:
    try:
        return (
            runtime.dedup_service.health()
            and runtime.classifier.is_ready()
            and runtime.nfa_vit_client.is_ready()
            and runtime.openai_client is not None
        )
    except Exception:
        logger.exception("Health check failed.")
        return False


def build_cached_payload(
    *,
    image_id: str,
    image_sha256: str | None,
    pipeline_result: dict[str, Any],
) -> dict[str, Any] | None:
    classification = pipeline_result.get("classification")
    if not isinstance(classification, dict):
        return None

    return {
        "image_id": image_id,
        "image_sha256": image_sha256,
        "pipeline_stage": pipeline_result.get("stage"),
        "classification": deepcopy(classification),
        "nfa_vit": deepcopy(pipeline_result.get("nfa_vit")),
        "final_label": pipeline_result.get("final_label"),
        "vlm": deepcopy(pipeline_result.get("vlm")),
        "openai_skipped": bool(pipeline_result.get("openai_skipped")),
        "skip_reason": pipeline_result.get("skip_reason"),
    }


def build_duplicate_cache_response(
    *,
    image_id: str,
    dedup_result: dict[str, Any],
    cached_payload: dict[str, Any],
    cache_source: str,
) -> dict[str, Any]:
    result = {
        "image_id": image_id,
        "status": "completed",
        "stage": cached_payload.get("pipeline_stage") or "dedup",
        "dedup": dedup_result,
        "classification": deepcopy(cached_payload.get("classification")),
        "nfa_vit": deepcopy(cached_payload.get("nfa_vit")),
        "final_label": cached_payload.get("final_label"),
        "vlm": deepcopy(cached_payload.get("vlm")),
        "served_from_cache": True,
        "cache_source": cache_source,
        "cached_from_image_id": cached_payload.get("image_id"),
        "cached_from_sha256": cached_payload.get("image_sha256"),
    }
    if cached_payload.get("openai_skipped"):
        result["openai_skipped"] = True
    if cached_payload.get("skip_reason"):
        result["skip_reason"] = cached_payload.get("skip_reason")
    return result


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

    duplicate_hashes = sorted(
        {
            result.get("matched_sha256")
            for result in dedup_results
            if isinstance(result, dict)
            and result.get("status") == "duplicate"
            and result.get("matched_sha256")
        }
    )
    persistent_cache_by_hash = (
        runtime.dedup_service.store.with_retry(
            lambda: runtime.dedup_service.store.query_cached_results(duplicate_hashes),
            op_name="query_cached_results",
        )
        if duplicate_hashes
        else {}
    )

    nfa_call_count = 0
    nfa_total_elapsed = 0.0
    openai_call_count = 0
    openai_total_elapsed = 0.0
    cached_payload_by_image_id: dict[str, dict[str, Any]] = {}
    cached_payload_by_sha256: dict[str, dict[str, Any]] = {}
    processed_rows_to_insert: dict[str, dict[str, Any]] = {}

    pipeline_results: list[dict[str, Any]] = []
    for idx, (item, dedup_result) in enumerate(zip(items, dedup_results)):
        if dedup_result.get("status") == "duplicate":
            matched_image_id = dedup_result.get("matched_image_id")
            matched_sha256 = dedup_result.get("matched_sha256")
            cached_payload = None
            cache_source = None

            if matched_image_id and matched_image_id in cached_payload_by_image_id:
                cached_payload = cached_payload_by_image_id[matched_image_id]
                cache_source = "request_memory"
            elif matched_sha256 and matched_sha256 in cached_payload_by_sha256:
                cached_payload = cached_payload_by_sha256[matched_sha256]
                cache_source = "request_memory"
            elif matched_sha256 and matched_sha256 in persistent_cache_by_hash:
                cached_payload = persistent_cache_by_hash[matched_sha256]
                cache_source = "milvus_result_cache"

            if cached_payload:
                pipeline_results.append(
                    build_duplicate_cache_response(
                        image_id=item["image_id"],
                        dedup_result=dedup_result,
                        cached_payload=cached_payload,
                        cache_source=cache_source or "unknown",
                    )
                )
                cached_payload_by_image_id[item["image_id"]] = cached_payload
                if matched_sha256:
                    cached_payload_by_sha256[matched_sha256] = cached_payload
            elif matched_sha256:
                logger.info(
                    "Duplicate cache miss. Rebuilding classification/nfa_vit/OpenAI result: image_id=%s matched_image_id=%s matched_sha256=%s",
                    item["image_id"],
                    matched_image_id,
                    matched_sha256,
                )
                try:
                    duplicate_image = bytes_to_pil(bytes_from_ref(item["image_ref"]))
                    duplicate_classification = runtime.classifier.classify([duplicate_image])[0]
                    if duplicate_classification.get("status") != "classified":
                        raise RuntimeError("duplicate_cache_rebuild_classification_failed")

                    nfa_started = time.perf_counter()
                    duplicate_nfa_vit = runtime.nfa_vit_client.infer(duplicate_image)
                    nfa_elapsed = time.perf_counter() - nfa_started
                    nfa_call_count += 1
                    nfa_total_elapsed += nfa_elapsed
                    logger.info(
                        "nfa_vit completed during duplicate cache rebuild: image_id=%s model=%s elapsed_seconds=%.4f final_label=%s",
                        item["image_id"],
                        runtime.cfg.nfa_model_name,
                        nfa_elapsed,
                        duplicate_nfa_vit.get("final_label"),
                    )

                    detector_context = {
                        "classification": duplicate_classification,
                        "nfa_vit": duplicate_nfa_vit,
                        "final_label": duplicate_nfa_vit.get("final_label"),
                    }

                    openai_started = time.perf_counter()
                    duplicate_vlm_result = runtime.openai_client.analyze(
                        image_ref=image_ref_for_openai(item["image_ref"]),
                        prompt=runtime.cfg.vlm_prompt,
                        detector_context=detector_context,
                    )
                    openai_elapsed = time.perf_counter() - openai_started
                    openai_call_count += 1
                    openai_total_elapsed += openai_elapsed
                    logger.info(
                        "OpenAI VLM completed during duplicate cache rebuild: image_id=%s model=%s elapsed_seconds=%.4f",
                        item["image_id"],
                        runtime.cfg.openai_model,
                        openai_elapsed,
                    )
                    pipeline_results.append(
                        {
                            "image_id": item["image_id"],
                            "status": "completed",
                            "stage": "openai",
                            "dedup": dedup_result,
                            "classification": duplicate_classification,
                            "nfa_vit": duplicate_nfa_vit,
                            "final_label": duplicate_nfa_vit.get("final_label"),
                            "vlm": duplicate_vlm_result,
                            "cache_backfilled": True,
                        }
                    )

                    cached_payload = build_cached_payload(
                        image_id=str(matched_image_id or item["image_id"]),
                        image_sha256=matched_sha256,
                        pipeline_result=pipeline_results[-1],
                    )
                    if cached_payload:
                        if matched_image_id:
                            cached_payload_by_image_id[str(matched_image_id)] = cached_payload
                        cached_payload_by_image_id[item["image_id"]] = cached_payload
                        cached_payload_by_sha256[matched_sha256] = cached_payload
                except Exception as exc:
                    logger.exception(
                        "Duplicate cache rebuild failed: image_id=%s matched_image_id=%s matched_sha256=%s",
                        item["image_id"],
                        matched_image_id,
                        matched_sha256,
                    )
                    pipeline_results.append(
                        {
                            "image_id": item["image_id"],
                            "status": "error",
                            "stage": "dedup_cache_rebuild",
                            "dedup": dedup_result,
                            "error": str(exc),
                        }
                    )
            else:
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

        record = unique_by_flat_idx[idx]
        try:
            nfa_started = time.perf_counter()
            nfa_vit_result = runtime.nfa_vit_client.infer(record["image"])
            nfa_elapsed = time.perf_counter() - nfa_started
            nfa_call_count += 1
            nfa_total_elapsed += nfa_elapsed
            logger.info(
                "nfa_vit completed: image_id=%s model=%s elapsed_seconds=%.4f final_label=%s",
                item["image_id"],
                runtime.cfg.nfa_model_name,
                nfa_elapsed,
                nfa_vit_result.get("final_label"),
            )
        except Exception as exc:
            pipeline_results.append(
                {
                    "image_id": item["image_id"],
                    "status": "error",
                    "stage": "nfa_vit",
                    "dedup": dedup_result,
                    "classification": classification,
                    "error": str(exc),
                }
            )
            continue

        detector_context = {
            "classification": classification,
            "nfa_vit": nfa_vit_result,
            "final_label": nfa_vit_result.get("final_label"),
        }
        try:
            openai_started = time.perf_counter()
            vlm_result = runtime.openai_client.analyze(
                image_ref=image_ref_for_openai(record["image_ref"]),
                prompt=runtime.cfg.vlm_prompt,
                detector_context=detector_context,
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
                    "nfa_vit": nfa_vit_result,
                    "final_label": nfa_vit_result.get("final_label"),
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
                "nfa_vit": nfa_vit_result,
                "final_label": nfa_vit_result.get("final_label"),
                "vlm": vlm_result,
            }
        )
        cached_payload = build_cached_payload(
            image_id=item["image_id"],
            image_sha256=dedup_result.get("sha256"),
            pipeline_result=pipeline_results[-1],
        )
        if cached_payload and dedup_result.get("sha256"):
            cached_payload_by_image_id[item["image_id"]] = cached_payload
            cached_payload_by_sha256[dedup_result["sha256"]] = cached_payload
            processed_rows_to_insert[dedup_result["sha256"]] = {
                "flat_idx": idx,
                "image_id": item["image_id"],
                "sha256": dedup_result["sha256"],
                "vector": record["vector"],
                "pipeline_stage": "openai",
                "classification": classification,
                "nfa_vit": nfa_vit_result,
                "final_label": nfa_vit_result.get("final_label"),
                "vlm": vlm_result,
                "openai_skipped": False,
                "skip_reason": None,
            }

    if nfa_call_count:
        logger.info(
            "nfa_vit summary: calls=%s total_elapsed_seconds=%.4f avg_seconds_per_call=%.4f",
            nfa_call_count,
            nfa_total_elapsed,
            nfa_total_elapsed / nfa_call_count,
        )
    else:
        logger.info("nfa_vit summary: calls=0 total_elapsed_seconds=0.0000")

    if openai_call_count:
        logger.info(
            "OpenAI VLM summary: calls=%s total_elapsed_seconds=%.4f avg_seconds_per_call=%.4f",
            openai_call_count,
            openai_total_elapsed,
            openai_total_elapsed / openai_call_count,
        )
    else:
        logger.info("OpenAI VLM summary: calls=0 total_elapsed_seconds=0.0000")

    if processed_rows_to_insert:
        runtime.dedup_service.store.with_retry(
            lambda: runtime.dedup_service.store.insert_processed_images(
                list(processed_rows_to_insert.values()),
                dedup_results,
            ),
            op_name="insert_processed_images",
        )

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
    title="Triton Dedup Classification NFA OpenAI API",
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
            detail="Triton, Milvus, NFA, or OpenAI runtime is not ready.",
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


@app.post("/analyze_multipart")
async def analyze_multipart(
    request: Request,
    files: list[UploadFile] = File(...),
    request_id: str | None = Form(None),
    image_ids: str | None = Form(None),
) -> dict[str, Any]:
    runtime = get_runtime(request)

    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")

    if len(files) > runtime.cfg.max_images_per_request:
        raise HTTPException(
            status_code=422,
            detail=f"Too many images. Max is {runtime.cfg.max_images_per_request}.",
        )

    rid = str(request_id or uuid.uuid4())

    parsed_image_ids: list[str] = []
    if image_ids:
        parsed_image_ids = [x.strip() for x in image_ids.split(",") if x.strip()]

    items: list[dict[str, Any]] = []
    for idx, upload in enumerate(files):
        content = await upload.read()
        if not content:
            raise HTTPException(
                status_code=422,
                detail=f"Uploaded file is empty: {upload.filename}",
            )

        image_id = (
            parsed_image_ids[idx]
            if idx < len(parsed_image_ids)
            else upload.filename
            or f"{rid}:{idx}"
        )

        items.append(
            {
                "request_id": rid,
                "image_id": image_id,
                "image_ref": upload_to_image_ref(
                    content=content,
                    filename=upload.filename,
                    content_type=upload.content_type,
                ),
            }
        )

    started = time.perf_counter()

    async with request.app.state.analyze_lock:
        runtime.dedup_service.store.reset_predict_state()
        item_results = await run_in_threadpool(run_pipeline_on_items, runtime, items)

    return build_response(
        runtime=runtime,
        request_id=rid,
        item_results=item_results,
        elapsed_seconds=time.perf_counter() - started,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "pipeline_server:app",
        host=os.getenv("HOST", os.getenv("API_HOST", "0.0.0.0")),
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
