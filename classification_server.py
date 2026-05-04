import base64
import binascii
import io
import json
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
    engine_path: str | None
    input_name: str
    output_name: str | None
    labels: list[str]
    image_size: tuple[int, int]
    api_model_name: str
    max_images_per_request: int

    @classmethod
    def from_env(cls) -> "ClassifierSettings":
        labels = [
            x.strip()
            for x in os.getenv("CLASSIFIER_LABELS", "negative,positive").split(",")
            if x.strip()
        ]
        if len(labels) != 2:
            raise ValueError("CLASSIFIER_LABELS must contain exactly 2 labels.")

        image_height = int(os.getenv("CLASSIFIER_IMAGE_HEIGHT", "224"))
        image_width = int(os.getenv("CLASSIFIER_IMAGE_WIDTH", "224"))

        return cls(
            engine_path=os.getenv("CLASSIFIER_ENGINE_PATH") or None,
            input_name=os.getenv("CLASSIFIER_INPUT_NAME", "pixel_values"),
            output_name=os.getenv("CLASSIFIER_OUTPUT_NAME") or None,
            labels=labels,
            image_size=(image_height, image_width),
            api_model_name=os.getenv("CLASSIFIER_API_MODEL", "image-classifier-trt"),
            max_images_per_request=int(os.getenv("MAX_IMAGES_PER_REQUEST", "1")),
        )


class ClassificationCallback(Callback):
    def on_server_start(self, *args, **kwargs):
        logger.info("Classification LitServe server starting.")

    def on_after_setup(self, *args, **kwargs):
        logger.info("Classification worker setup completed.")

    def on_before_predict(self, *args, **kwargs):
        logger.debug("Classification predict started.")

    def on_after_predict(self, *args, **kwargs):
        logger.debug("Classification predict finished.")


class DummyClassificationRunner:
    """Deterministic runner for API integration before TensorRT engine is ready."""

    backend = "dummy"

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        batch = pixel_values.shape[0]
        brightness = pixel_values.mean(axis=(1, 2, 3))
        logits = np.zeros((batch, 2), dtype=np.float32)
        logits[:, 0] = -brightness
        logits[:, 1] = brightness
        return logits


class TensorRTClassificationRunner:
    backend = "tensorrt"

    def __init__(
        self,
        engine_path: str,
        input_name: str,
        output_name: str | None,
    ) -> None:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runner requires packages: tensorrt, pycuda."
            ) from exc

        self.cuda = cuda
        self.trt = trt
        self.input_name = input_name
        self.output_name = output_name
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        if hasattr(self.engine, "num_io_tensors"):
            self._api = "v10"
            self._init_v10_names()
        else:
            self._api = "v8"
            self._init_v8_bindings()

    def _init_v10_names(self) -> None:
        trt = self.trt
        input_names = []
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if self.input_name not in input_names:
            raise ValueError(
                f"Input tensor {self.input_name!r} not found. Available: {input_names}"
            )

        if self.output_name is None:
            if len(output_names) != 1:
                raise ValueError(
                    "CLASSIFIER_OUTPUT_NAME is required when engine has multiple "
                    f"outputs: {output_names}"
                )
            self.output_name = output_names[0]
        elif self.output_name not in output_names:
            raise ValueError(
                f"Output tensor {self.output_name!r} not found. Available: {output_names}"
            )

    def _init_v8_bindings(self) -> None:
        input_names = []
        output_names = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                input_names.append(name)
            else:
                output_names.append(name)

        if self.input_name not in input_names:
            raise ValueError(
                f"Input binding {self.input_name!r} not found. Available: {input_names}"
            )

        if self.output_name is None:
            if len(output_names) != 1:
                raise ValueError(
                    "CLASSIFIER_OUTPUT_NAME is required when engine has multiple "
                    f"outputs: {output_names}"
                )
            self.output_name = output_names[0]
        elif self.output_name not in output_names:
            raise ValueError(
                f"Output binding {self.output_name!r} not found. Available: {output_names}"
            )

    def infer(self, pixel_values: np.ndarray) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values.astype(np.float32))
        if self._api == "v10":
            return self._infer_v10(pixel_values)
        return self._infer_v8(pixel_values)

    def _infer_v10(self, pixel_values: np.ndarray) -> np.ndarray:
        cuda = self.cuda
        trt = self.trt

        self.context.set_input_shape(self.input_name, tuple(pixel_values.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        output = np.empty(output_shape, dtype=output_dtype)

        input_mem = cuda.mem_alloc(pixel_values.nbytes)
        output_mem = cuda.mem_alloc(output.nbytes)

        cuda.memcpy_htod_async(input_mem, pixel_values, self.stream)
        self.context.set_tensor_address(self.input_name, int(input_mem))
        self.context.set_tensor_address(self.output_name, int(output_mem))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output, output_mem, self.stream)
        self.stream.synchronize()

        return output.astype(np.float32)

    def _infer_v8(self, pixel_values: np.ndarray) -> np.ndarray:
        cuda = self.cuda
        trt = self.trt

        input_idx = self.engine.get_binding_index(self.input_name)
        output_idx = self.engine.get_binding_index(self.output_name)
        if self.engine.is_shape_binding(input_idx) is False:
            self.context.set_binding_shape(input_idx, tuple(pixel_values.shape))

        output_shape = tuple(self.context.get_binding_shape(output_idx))
        output_dtype = trt.nptype(self.engine.get_binding_dtype(output_idx))
        output = np.empty(output_shape, dtype=output_dtype)

        input_mem = cuda.mem_alloc(pixel_values.nbytes)
        output_mem = cuda.mem_alloc(output.nbytes)
        bindings = [0] * self.engine.num_bindings
        bindings[input_idx] = int(input_mem)
        bindings[output_idx] = int(output_mem)

        cuda.memcpy_htod_async(input_mem, pixel_values, self.stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output, output_mem, self.stream)
        self.stream.synchronize()

        return output.astype(np.float32)


def decode_base64_to_bytes(image_base64: str) -> bytes:
    if "," in image_base64 and image_base64.lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        return base64.b64decode(image_base64, validate=True)
    except binascii.Error:
        compact = "".join(image_base64.split())
        return base64.b64decode(compact, validate=True)


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.convert("RGB").copy()


def load_url_image(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "OpenSDI-classifier/1.0"})
    with urlopen(req, timeout=15) as response:
        return bytes_to_pil(response.read())


def image_from_any(value: str) -> Image.Image:
    if value.startswith("http://") or value.startswith("https://"):
        return load_url_image(value)
    return bytes_to_pil(decode_base64_to_bytes(value))


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def extract_image_refs_from_openai_request(request: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    if "image_base64" in request:
        refs.append(request["image_base64"])
    if "image_url" in request:
        refs.append(request["image_url"])
    if "images" in request and isinstance(request["images"], list):
        for item in request["images"]:
            if isinstance(item, str):
                refs.append(item)
            elif isinstance(item, dict):
                ref = item.get("image_base64") or item.get("b64") or item.get("image_url")
                if ref:
                    refs.append(ref)

    for message in request.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            if content.startswith("data:image/"):
                refs.append(content)
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"image_url", "input_image"}:
                image_url = block.get("image_url")
                if isinstance(image_url, dict):
                    ref = image_url.get("url")
                else:
                    ref = image_url
                if ref:
                    refs.append(ref)

    return refs


def openai_chat_response(
    *,
    model: str,
    content: dict[str, Any],
    request_id: str | None = None,
) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": request_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


class OpenAIImageClassificationAPI(ls.LitAPI):
    def __init__(self):
        max_batch_size = int(os.getenv("LITSERVE_MAX_BATCH_SIZE", "8"))
        batch_timeout = float(os.getenv("LITSERVE_BATCH_TIMEOUT", "0.05"))
        super().__init__(
            api_path="/v1/chat/completions",
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )

    def setup(self, device):
        self.cfg = ClassifierSettings.from_env()
        if self.cfg.engine_path:
            logger.info("Loading TensorRT engine: %s", self.cfg.engine_path)
            self.runner = TensorRTClassificationRunner(
                engine_path=self.cfg.engine_path,
                input_name=self.cfg.input_name,
                output_name=self.cfg.output_name,
            )
        else:
            logger.warning("CLASSIFIER_ENGINE_PATH is empty. Using dummy runner.")
            self.runner = DummyClassificationRunner()

    def decode_request(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Request body must be a JSON object.")

        dedup = request.get("dedup") or {}
        if isinstance(dedup, dict) and dedup.get("status") == "duplicate":
            return {
                "skip": True,
                "model": request.get("model") or self.cfg.api_model_name,
                "request_id": request.get("id"),
                "reason": "duplicate_image",
                "dedup": dedup,
            }

        refs = extract_image_refs_from_openai_request(request)
        if not refs:
            raise ValueError(
                "No image found. Send image_url/image_base64 or OpenAI content "
                "block {type: 'image_url', image_url: {url: ...}}."
            )
        if len(refs) > self.cfg.max_images_per_request:
            raise ValueError(
                f"Too many images. Max is {self.cfg.max_images_per_request}."
            )

        image = image_from_any(refs[0])
        pixel_values = preprocess_image(image, self.cfg.image_size)

        return {
            "skip": False,
            "model": request.get("model") or self.cfg.api_model_name,
            "request_id": request.get("id"),
            "pixel_values": pixel_values,
        }

    def batch(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        runnable = [x for x in inputs if not x["skip"]]
        skipped = {idx: item for idx, item in enumerate(inputs) if item["skip"]}

        if runnable:
            pixel_values = np.stack([x["pixel_values"] for x in runnable]).astype(
                np.float32
            )
        else:
            height, width = self.cfg.image_size
            pixel_values = np.empty((0, 3, height, width), dtype=np.float32)

        return {
            "inputs": inputs,
            "runnable": runnable,
            "skipped": skipped,
            "pixel_values": pixel_values,
        }

    def predict(self, batch: dict[str, Any], **kwargs) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any] | None] = [None] * len(batch["inputs"])

        if batch["pixel_values"].shape[0] > 0:
            logits = self.runner.infer(batch["pixel_values"])
            probs = softmax(logits)

            run_idx = 0
            for original_idx, item in enumerate(batch["inputs"]):
                if item["skip"]:
                    continue

                row_probs = probs[run_idx]
                pred_idx = int(np.argmax(row_probs))
                outputs[original_idx] = {
                    "status": "classified",
                    "backend": self.runner.backend,
                    "label": self.cfg.labels[pred_idx],
                    "class_index": pred_idx,
                    "confidence": float(row_probs[pred_idx]),
                    "scores": {
                        label: float(row_probs[i])
                        for i, label in enumerate(self.cfg.labels)
                    },
                }
                run_idx += 1

        for original_idx, item in batch["skipped"].items():
            outputs[original_idx] = {
                "status": "skipped",
                "reason": item["reason"],
                "dedup": item["dedup"],
            }

        return [x for x in outputs if x is not None]

    def unbatch(self, output: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return output

    def encode_response(self, output: dict[str, Any], **kwargs) -> dict[str, Any]:
        return openai_chat_response(
            model=self.cfg.api_model_name,
            content=output,
        )

    def health(self) -> bool:
        return hasattr(self, "runner")


if __name__ == "__main__":
    api = OpenAIImageClassificationAPI()
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
