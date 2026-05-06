import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import litserve as ls
import requests
from dotenv import load_dotenv

try:
    from litserve.callbacks import Callback
except Exception:
    from litserve.callbacks.base import Callback


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pipeline-litserve")


@dataclass
class PipelineSettings:
    dedup_api_url: str
    classifier_api_url: str
    openai_model: str
    openai_proxy: str | None
    openai_timeout: float
    request_timeout: float
    vlm_prompt: str

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        return cls(
            dedup_api_url=os.getenv("DEDUP_API_URL", "http://127.0.0.1:8000/dedup"),
            classifier_api_url=os.getenv(
                "CLASSIFIER_API_URL",
                "http://127.0.0.1:8001/classify",
            ),
            openai_model=os.getenv("OPENAI_VLM_MODEL", "gpt-5-mini"),
            openai_proxy=os.getenv("OPENAI_PROXY") or None,
            openai_timeout=float(os.getenv("OPENAI_TIMEOUT", "60")),
            request_timeout=float(os.getenv("PIPELINE_REQUEST_TIMEOUT", "60")),
            vlm_prompt=os.getenv(
                "OPENAI_VLM_PROMPT",
                (
                    "Analyze the image after local deduplication and TensorRT "
                    "classification. Use the classification result as a signal, "
                    "but rely on the visual content when explaining the final answer. "
                    "Return concise JSON with keys: summary, decision, reasons."
                ),
            ),
        )


class PipelineCallback(Callback):
    def on_server_start(self, *args, **kwargs):
        logger.info("Pipeline LitServe server starting.")

    def on_after_setup(self, *args, **kwargs):
        logger.info("Pipeline worker setup completed.")


def normalize_images_payload(request: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
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

    items = []
    for idx, item in enumerate(images):
        if isinstance(item, str):
            image_id = f"{request_id}:{idx}"
            image_ref = item
            image_base64 = item if not item.startswith(("http://", "https://")) else None
            image_url = item if item.startswith(("http://", "https://")) else None
        elif isinstance(item, dict):
            image_id = str(item.get("image_id") or f"{request_id}:{idx}")
            image_base64 = item.get("image_base64") or item.get("b64")
            image_url = item.get("image_url") or item.get("url")
            image_ref = image_base64 or image_url
        else:
            raise ValueError("Each image must be a string or object.")

        if not image_ref:
            raise ValueError("Each image must contain image_base64 or image_url.")

        normalized = {"image_id": image_id}
        if image_base64:
            normalized["image_base64"] = image_base64
        if image_url:
            normalized["image_url"] = image_url
        items.append(normalized)

    return request_id, items


def response_to_dict(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError("Upstream returned an empty response list.")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise RuntimeError("Upstream response must be a JSON object.")
    return payload


def image_item_for_upstream(item: dict[str, str]) -> dict[str, str]:
    payload = {"image_id": item["image_id"]}
    if item.get("image_base64"):
        payload["image_base64"] = item["image_base64"]
    if item.get("image_url"):
        payload["image_url"] = item["image_url"]
    return payload


def image_ref_for_openai(item: dict[str, str]) -> str:
    if item.get("image_url"):
        return item["image_url"]

    image_base64 = item["image_base64"]
    if image_base64.startswith("data:image/"):
        return image_base64
    return f"data:image/jpeg;base64,{image_base64}"


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
                                "TensorRT classification result:\n"
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


class DedupClassifyVLMAPI(ls.LitAPI):
    def __init__(self):
        max_batch_size = int(os.getenv("PIPELINE_BATCH_SIZE", "1"))
        batch_timeout = float(os.getenv("PIPELINE_BATCH_TIMEOUT", "0.0"))
        super().__init__(
            api_path="/analyze",
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )
        self.cfg: PipelineSettings | None = None
        self.openai_client: OpenAIVisionClient | None = None

    def setup(self, device):
        self.cfg = PipelineSettings.from_env()
        self.openai_client = OpenAIVisionClient(
            model=self.cfg.openai_model,
            timeout=self.cfg.openai_timeout,
            proxy=self.cfg.openai_proxy,
        )

    def decode_request(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Request body must be JSON object.")
        request_id, items = normalize_images_payload(request)
        return {
            "request_id": request_id,
            "items": items,
        }

    def batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return inputs

    def predict(
        self,
        batch: dict[str, Any] | list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(batch, dict):
            return self._predict_one(batch)
        if isinstance(batch, list):
            return [self._predict_one(request) for request in batch]
        raise TypeError(
            "predict() expected a decoded request dict or a batched list of request dicts."
        )

    def unbatch(self, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return output

    def _predict_one(self, request: dict[str, Any]) -> dict[str, Any]:
        cfg = self._cfg
        started = time.perf_counter()
        items = request["items"]

        dedup_response = self._call_dedup(request["request_id"], items)
        dedup_results = dedup_response.get("results", [])
        items_by_id = {item["image_id"]: item for item in items}

        unique_items = []
        pipeline_results = []
        for dedup_result in dedup_results:
            image_id = dedup_result.get("image_id")
            item = items_by_id.get(image_id)
            if item is None:
                continue

            if dedup_result.get("status") != "unique":
                pipeline_results.append(
                    {
                        "image_id": image_id,
                        "status": "skipped",
                        "stage": "dedup",
                        "dedup": dedup_result,
                    }
                )
                continue

            unique_items.append(item)

        classification_by_id = {}
        if unique_items:
            classify_response = self._call_classification(request["request_id"], unique_items)
            for classify_result in classify_response.get("results", []):
                classification_by_id[classify_result.get("image_id")] = classify_result

        for item in unique_items:
            image_id = item["image_id"]
            classification = classification_by_id.get(image_id)
            if not classification or classification.get("status") != "classified":
                pipeline_results.append(
                    {
                        "image_id": image_id,
                        "status": "error",
                        "stage": "classification",
                        "classification": classification,
                    }
                )
                continue

            vlm_result = self._openai_client.analyze(
                image_ref=image_ref_for_openai(item),
                prompt=cfg.vlm_prompt,
                classification=classification,
            )
            pipeline_results.append(
                {
                    "image_id": image_id,
                    "status": "completed",
                    "dedup": next(
                        x for x in dedup_results if x.get("image_id") == image_id
                    ),
                    "classification": classification,
                    "vlm": vlm_result,
                }
            )

        return {
            "request_id": request["request_id"],
            "total_images": len(items),
            "results": pipeline_results,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }

    def encode_response(self, output: dict[str, Any], **kwargs) -> dict[str, Any]:
        return output

    def health(self) -> bool:
        return self.cfg is not None and self.openai_client is not None

    def _call_dedup(
        self,
        request_id: str,
        items: list[dict[str, str]],
    ) -> dict[str, Any]:
        response = requests.post(
            self._cfg.dedup_api_url,
            json={
                "request_id": request_id,
                "images": [image_item_for_upstream(item) for item in items],
            },
            timeout=self._cfg.request_timeout,
        )
        return response_to_dict(response)

    def _call_classification(
        self,
        request_id: str,
        items: list[dict[str, str]],
    ) -> dict[str, Any]:
        response = requests.post(
            self._cfg.classifier_api_url,
            json={
                "request_id": request_id,
                "images": [image_item_for_upstream(item) for item in items],
            },
            timeout=self._cfg.request_timeout,
        )
        return response_to_dict(response)

    @property
    def _cfg(self) -> PipelineSettings:
        if self.cfg is None:
            raise RuntimeError("Pipeline settings are not initialized.")
        return self.cfg

    @property
    def _openai_client(self) -> OpenAIVisionClient:
        if self.openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")
        return self.openai_client

if __name__ == "__main__":
    api = DedupClassifyVLMAPI()
    server = ls.LitServer(
        api,
        accelerator=os.getenv("LITSERVE_ACCELERATOR", "cpu"),
        devices=os.getenv("LITSERVE_DEVICES", "1"),
        workers_per_device=int(os.getenv("LITSERVE_WORKERS_PER_DEVICE", "1")),
        timeout=int(os.getenv("LITSERVE_REQUEST_TIMEOUT", "180")),
        track_requests=True,
        callbacks=[PipelineCallback()],
    )
    server.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PIPELINE_PORT", os.getenv("PORT", "8002"))),
    )
