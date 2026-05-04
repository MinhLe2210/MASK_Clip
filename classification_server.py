import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

import litserve as ls
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageFile

from src.image_io import bytes_to_pil, decode_base64_to_bytes
from src.trt_runner import TensorRTRunner

try:
    from litserve.callbacks import Callback
except Exception:
    from litserve.callbacks.base import Callback


ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("classification-litserve")


@dataclass
class ClassifierSettings:
    engine_path: str
    input_name: str
    output_name: str | None
    labels: list[str]
    input_size: int
    max_images_per_request: int

    @classmethod
    def from_env(cls) -> "ClassifierSettings":
        engine_path = os.getenv("CLASSIFIER_ENGINE_PATH")
        if not engine_path:
            raise ValueError("CLASSIFIER_ENGINE_PATH is missing.")

        labels = [
            x.strip()
            for x in os.getenv("CLASSIFIER_LABELS", "negative,positive").split(",")
            if x.strip()
        ]
        if len(labels) != 2:
            raise ValueError("CLASSIFIER_LABELS must contain exactly 2 labels.")

        return cls(
            engine_path=engine_path,
            input_name=os.getenv("CLASSIFIER_INPUT_NAME", "input"),
            output_name=os.getenv("CLASSIFIER_OUTPUT_NAME", "output") or None,
            labels=labels,
            input_size=int(os.getenv("CLASSIFIER_INPUT_SIZE", "384")),
            max_images_per_request=int(os.getenv("MAX_IMAGES_PER_REQUEST", "16")),
        )


class ClassificationCallback(Callback):
    def on_server_start(self, *args, **kwargs):
        logger.info("Classification TensorRT LitServe server starting.")

    def on_after_setup(self, *args, **kwargs):
        logger.info("Classification worker setup completed.")


class TensorRTImageClassifier:
    def __init__(self, cfg: ClassifierSettings):
        self.cfg = cfg
        logger.info("Reading TensorRT engine from file %s", cfg.engine_path)
        self.runner = TensorRTRunner(
            engine_path=cfg.engine_path,
            input_name=cfg.input_name,
            output_name=cfg.output_name,
        )
        logger.info("Input tensor: %s", cfg.input_name)
        logger.info("Output tensor: %s", self.runner.output_name)

    def classify(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        started = time.perf_counter()
        inputs = self._preprocess_for_trt(images)
        outputs = self.runner.infer(inputs)

        output_name = self.runner.output_name or next(iter(outputs.keys()))
        logits = outputs[output_name]

        if logits.ndim == 3:
            logits = logits[:, 0, :]
        elif logits.ndim > 2:
            logits = logits.reshape((logits.shape[0], -1))

        logits = logits.astype(np.float32)
        probs = softmax(logits)
        elapsed = time.perf_counter() - started

        results = []
        for row_logits, row_probs in zip(logits, probs):
            class_index = int(np.argmax(row_probs))
            results.append(
                {
                    "status": "classified",
                    "backend": "tensorrt",
                    "label": self.cfg.labels[class_index],
                    "class_index": class_index,
                    "confidence": float(row_probs[class_index]),
                    "logits": row_logits.astype(float).tolist(),
                    "scores": {
                        label: float(row_probs[i])
                        for i, label in enumerate(self.cfg.labels)
                    },
                    "elapsed_seconds": elapsed,
                }
            )

        return results

    def _preprocess_for_trt(self, images: list[Image.Image]) -> np.ndarray:
        batch = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for img in images:
            img = img.convert("RGB").resize(
                (self.cfg.input_size, self.cfg.input_size),
                Image.Resampling.BICUBIC,
            )

            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2, 0, 1))
            batch.append(arr)

        return np.ascontiguousarray(np.stack(batch).astype(np.float32))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def load_url_image(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "OpenSDI-classifier/1.0"})
    with urlopen(req, timeout=15) as response:
        return bytes_to_pil(response.read())


def image_from_ref(value: str) -> Image.Image:
    if value.startswith("http://") or value.startswith("https://"):
        return load_url_image(value)
    return bytes_to_pil(decode_base64_to_bytes(value))


def decode_classification_request(
    request: dict[str, Any],
    cfg: ClassifierSettings,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("Request body must be JSON object.")

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
    if len(images) > cfg.max_images_per_request:
        raise ValueError(f"Too many images. Max is {cfg.max_images_per_request}.")

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

    return {
        "request_id": request_id,
        "items": items,
    }


class TensorRTClassificationAPI(ls.LitAPI):
    def __init__(self):
        max_batch_size = int(
            os.getenv(
                "CLASSIFIER_BATCH_SIZE",
                os.getenv("LITSERVE_MAX_BATCH_SIZE", "16"),
            )
        )
        batch_timeout = float(
            os.getenv(
                "CLASSIFIER_BATCH_TIMEOUT",
                os.getenv("LITSERVE_BATCH_TIMEOUT", "0.05"),
            )
        )
        super().__init__(
            api_path="/classify",
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )
        self.cfg: ClassifierSettings | None = None
        self.classifier: TensorRTImageClassifier | None = None

    def setup(self, device):
        self.cfg = ClassifierSettings.from_env()
        self.classifier = TensorRTImageClassifier(self.cfg)

    def decode_request(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        return decode_classification_request(request, self._cfg)

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

    def predict(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        flat_items = batch["flat_items"]
        flat_results: list[dict[str, Any] | None] = [None] * len(flat_items)

        valid_images = []
        valid_indices = []
        for idx, item in enumerate(flat_items):
            try:
                valid_images.append(image_from_ref(item["image_ref"]))
                valid_indices.append(idx)
            except Exception as exc:
                flat_results[idx] = {
                    "image_id": item["image_id"],
                    "status": "error",
                    "error": f"invalid_image: {exc}",
                }

        if valid_images:
            classify_results = self._classifier.classify(valid_images)
            for idx, result in zip(valid_indices, classify_results):
                result["image_id"] = flat_items[idx]["image_id"]
                flat_results[idx] = result

        return {
            "request_ids": batch["request_ids"],
            "request_sizes": batch["request_sizes"],
            "flat_results": flat_results,
        }

    def unbatch(self, output: dict[str, Any]) -> list[dict[str, Any]]:
        responses = []
        cursor = 0

        for request_id, size in zip(output["request_ids"], output["request_sizes"]):
            request_results = output["flat_results"][cursor : cursor + size]
            cursor += size
            responses.append(
                {
                    "request_id": request_id,
                    "total_images": size,
                    "results": request_results,
                }
            )

        return responses

    def encode_response(self, output: dict[str, Any], **kwargs) -> dict[str, Any]:
        return output

    def health(self) -> bool:
        return self.classifier is not None

    @property
    def _cfg(self) -> ClassifierSettings:
        if self.cfg is None:
            raise RuntimeError("Classifier settings are not initialized.")
        return self.cfg

    @property
    def _classifier(self) -> TensorRTImageClassifier:
        if self.classifier is None:
            raise RuntimeError("TensorRT classifier is not initialized.")
        return self.classifier


if __name__ == "__main__":
    api = TensorRTClassificationAPI()
    server = ls.LitServer(
        api,
        accelerator=os.getenv("LITSERVE_ACCELERATOR", "auto"),
        devices=os.getenv("LITSERVE_DEVICES", "auto"),
        workers_per_device=int(os.getenv("LITSERVE_WORKERS_PER_DEVICE", "1")),
        timeout=int(os.getenv("LITSERVE_REQUEST_TIMEOUT", "120")),
        track_requests=True,
        callbacks=[ClassificationCallback()],
    )
    server.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("CLASSIFIER_PORT", os.getenv("PORT", "8001"))),
    )
