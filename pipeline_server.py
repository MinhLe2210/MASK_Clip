import json
import logging
import os
import time
import uuid
from typing import Any

import litserve as ls
from dotenv import load_dotenv

try:
    from litserve.callbacks import Callback
except Exception:
    from litserve.callbacks.base import Callback

from src.config import Settings
from src.dedup_service import DedupService
from src.image_io import image_ref_for_openai
from src.milvus_store import MilvusDedupStore
from src.triton_clients import TritonClassificationClient, TritonEmbeddingClient


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pipeline-litserve")


class PipelineCallback(Callback):
    def on_server_start(self, *args, **kwargs):
        logger.info("Triton pipeline LitServe server starting.")

    def on_after_setup(self, *args, **kwargs):
        logger.info("Triton pipeline worker setup completed.")


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


class TritonDedupClassificationOpenAIAPI(ls.LitAPI):
    def __init__(self):
        max_batch_size = int(
            os.getenv(
                "PIPELINE_BATCH_SIZE",
                os.getenv("MAX_BATCH_SIZE", "8"),
            )
        )
        batch_timeout = float(
            os.getenv(
                "PIPELINE_BATCH_TIMEOUT",
                os.getenv("BATCH_TIMEOUT", "0.05"),
            )
        )

        super().__init__(
            api_path="/analyze",
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )
        self.cfg: Settings | None = None
        self.dedup_service: DedupService | None = None
        self.classifier: TritonClassificationClient | None = None
        self.openai_client: OpenAIVisionClient | None = None

    def setup(self, device):
        self.cfg = Settings.from_env()

        embedder = TritonEmbeddingClient(
            url=self.cfg.triton_client_url,
            model_name=self.cfg.dedup_model_name,
            input_name=self.cfg.dedup_input_name,
            output_name=self.cfg.dedup_output_name,
            image_size=self.cfg.dedup_image_size,
            embedding_dim=self.cfg.dedup_embedding_dim,
            timeout=self.cfg.request_timeout,
        )
        self.cfg.dedup_embedding_dim = embedder.embedding_dim
        classifier = TritonClassificationClient(
            url=self.cfg.triton_client_url,
            model_name=self.cfg.classification_model_name,
            input_name=self.cfg.classification_input_name,
            output_name=self.cfg.classification_output_name,
            image_size=self.cfg.classification_image_size,
            labels=self.cfg.classification_labels,
            timeout=self.cfg.request_timeout,
        )
        store = MilvusDedupStore(self.cfg)
        dedup_service = DedupService(
            embedder=embedder,
            store=store,
            dup_threshold=self.cfg.dup_threshold,
        )
        dedup_service.prepare()

        self.dedup_service = dedup_service
        self.classifier = classifier
        self.openai_client = OpenAIVisionClient(
            model=self.cfg.openai_model,
            timeout=self.cfg.openai_timeout,
            proxy=self.cfg.openai_proxy,
        )

    def decode_request(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Request body must be JSON object.")
        request_id, items = normalize_images_payload(request, self._cfg.max_images_per_request)
        return {"request_id": request_id, "items": items}

    def batch(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        flat_items = []
        request_ids = []
        request_sizes = []
        for req in inputs:
            request_ids.append(req["request_id"])
            request_sizes.append(len(req["items"]))
            flat_items.extend(req["items"])
        return {
            "request_ids": request_ids,
            "request_sizes": request_sizes,
            "flat_items": flat_items,
        }

    def predict(
        self,
        batch: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        self._dedup_service.store.reset_predict_state()

        if "request_id" in batch and "items" in batch:
            return self._predict_one(batch)
        if "flat_items" in batch:
            return self._predict_many(batch)
        raise TypeError("Unsupported batch payload for pipeline predict().")

    def unbatch(self, output: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
        return output

    def encode_response(self, output: dict[str, Any], **kwargs) -> dict[str, Any]:
        return output

    def health(self) -> bool:
        try:
            return (
                self.dedup_service is not None
                and self.classifier is not None
                and self.openai_client is not None
                and self._dedup_service.health()
                and self._classifier.is_ready()
            )
        except Exception:
            logger.exception("Health check failed.")
            return False

    def _predict_one(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        item_results = self._run_pipeline_on_items(request["items"])
        return self._build_response(
            request_id=request["request_id"],
            item_results=item_results,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _predict_many(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        started = time.perf_counter()
        item_results = self._run_pipeline_on_items(batch["flat_items"])
        responses = []
        cursor = 0
        for request_id, size in zip(batch["request_ids"], batch["request_sizes"]):
            request_results = item_results[cursor : cursor + size]
            cursor += size
            responses.append(
                self._build_response(
                    request_id=request_id,
                    item_results=request_results,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        return responses

    def _run_pipeline_on_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dedup_results, unique_records = self._dedup_service.deduplicate(items)
        unique_by_flat_idx = {record["flat_idx"]: record for record in unique_records}

        classification_by_flat_idx: dict[int, dict[str, Any]] = {}
        if unique_records:
            classify_results = self._classifier.classify([x["image"] for x in unique_records])
            for record, classify_result in zip(unique_records, classify_results):
                classification_by_flat_idx[record["flat_idx"]] = classify_result

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

            record = unique_by_flat_idx[idx]
            try:
                vlm_result = self._openai_client.analyze(
                    image_ref=image_ref_for_openai(record["image_ref"]),
                    prompt=self._cfg.vlm_prompt,
                    classification=classification,
                )
            except Exception as exc:
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
                    "dedup": dedup_result,
                    "classification": classification,
                    "vlm": vlm_result,
                }
            )

        return pipeline_results

    def _build_response(
        self,
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
            "milvus_recovered_this_predict": self._dedup_service.store.recovered_this_predict,
            "last_milvus_error": self._dedup_service.store.last_error,
        }

    @property
    def _cfg(self) -> Settings:
        if self.cfg is None:
            raise RuntimeError("Pipeline settings are not initialized.")
        return self.cfg

    @property
    def _dedup_service(self) -> DedupService:
        if self.dedup_service is None:
            raise RuntimeError("Dedup service is not initialized.")
        return self.dedup_service

    @property
    def _classifier(self) -> TritonClassificationClient:
        if self.classifier is None:
            raise RuntimeError("Classification client is not initialized.")
        return self.classifier

    @property
    def _openai_client(self) -> OpenAIVisionClient:
        if self.openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")
        return self.openai_client


def main() -> None:
    api = TritonDedupClassificationOpenAIAPI()
    server = ls.LitServer(
        api,
        accelerator=os.getenv("LITSERVE_ACCELERATOR", "cpu"),
        devices=os.getenv("LITSERVE_DEVICES", "1"),
        workers_per_device=int(os.getenv("LITSERVE_WORKERS_PER_DEVICE", "1")),
        timeout=int(os.getenv("LITSERVE_REQUEST_TIMEOUT", "180")),
        track_requests=True,
        max_payload_size=os.getenv("LITSERVE_MAX_PAYLOAD_SIZE", "100MB"),
        callbacks=[PipelineCallback()],
    )
    server.run(
        host=os.getenv("HOST", os.getenv("API_HOST", "0.0.0.0")),
        port=int(
            os.getenv(
                "PIPELINE_PORT",
                os.getenv("API_PORT", os.getenv("PORT", "8002")),
            )
        ),
    )


if __name__ == "__main__":
    main()
