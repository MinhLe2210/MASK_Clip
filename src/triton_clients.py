import logging
from typing import Any

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype


logger = logging.getLogger("pipeline-litserve")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(np.sum(exp_x, axis=-1, keepdims=True), 1e-12, None)


def resize_and_normalize(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


class TritonModelClient:
    def __init__(self, url: str, model_name: str, timeout: float):
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.client = httpclient.InferenceServerClient(
            url=url,
            concurrency=8,
            connection_timeout=timeout,
            network_timeout=timeout,
        )

    def is_ready(self) -> bool:
        return bool(self.client.is_model_ready(self.model_name))

    def _get_model_metadata(self) -> dict[str, Any]:
        return self.client.get_model_metadata(self.model_name)

    def _infer_fixed_image_size(self, input_name: str) -> int:
        metadata = self._get_model_metadata()
        for item in metadata.get("inputs", []):
            if item.get("name") != input_name:
                continue

            dims = [int(x) for x in item.get("shape", [])]
            if len(dims) >= 2 and dims[-1] > 0 and dims[-2] > 0:
                if dims[-1] != dims[-2]:
                    raise ValueError(
                        f"Model '{self.model_name}' input '{input_name}' is not square: {dims}"
                    )
                return dims[-1]

        raise ValueError(
            f"Could not infer fixed image size for model '{self.model_name}' input '{input_name}'. "
            "Set the image size explicitly in the environment."
        )

    def _infer_output_dim(self, output_name: str) -> int:
        metadata = self._get_model_metadata()
        for item in metadata.get("outputs", []):
            if item.get("name") != output_name:
                continue

            dims = [int(x) for x in item.get("shape", [])]
            for dim in reversed(dims):
                if dim > 0:
                    return dim

        raise ValueError(
            f"Could not infer output dim for model '{self.model_name}' output '{output_name}'. "
            "Set EMBED_DIM explicitly in the environment."
        )


class TritonEmbeddingClient(TritonModelClient):
    def __init__(
        self,
        url: str,
        model_name: str,
        input_name: str,
        output_name: str,
        image_size: int | None,
        embedding_dim: int | None,
        timeout: float,
    ):
        super().__init__(url=url, model_name=model_name, timeout=timeout)
        self.input_name = input_name
        self.output_name = output_name
        self.image_size = image_size or self._infer_fixed_image_size(input_name)
        self.embedding_dim = embedding_dim or self._infer_output_dim(output_name)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim or 0), dtype=np.float32)

        batch = np.stack(
            [resize_and_normalize(image, self.image_size) for image in images],
            axis=0,
        ).astype(np.float32)

        inputs = [
            httpclient.InferInput(
                self.input_name,
                list(batch.shape),
                np_to_triton_dtype(batch.dtype),
            )
        ]
        inputs[0].set_data_from_numpy(batch)

        outputs = [httpclient.InferRequestedOutput(self.output_name)]
        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )

        vectors = result.as_numpy(self.output_name)
        if vectors is None:
            raise RuntimeError(
                f"No output named '{self.output_name}' returned from Triton model '{self.model_name}'."
            )

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim == 3:
            vectors = vectors.mean(axis=1)
        elif vectors.ndim > 3:
            vectors = vectors.reshape((vectors.shape[0], -1))

        if self.embedding_dim is not None and vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch for model '{self.model_name}': "
                f"expected {self.embedding_dim}, got {vectors.shape[1]}"
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class TritonClassificationClient(TritonModelClient):
    def __init__(
        self,
        url: str,
        model_name: str,
        input_name: str,
        output_name: str,
        image_size: int | None,
        labels: list[str],
        timeout: float,
    ):
        super().__init__(url=url, model_name=model_name, timeout=timeout)
        self.input_name = input_name
        self.output_name = output_name
        self.image_size = image_size or self._infer_fixed_image_size(input_name)
        self.labels = labels

    def classify(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        if not images:
            return []

        batch = np.stack(
            [resize_and_normalize(image, self.image_size) for image in images],
            axis=0,
        ).astype(np.float32)

        inputs = [
            httpclient.InferInput(
                self.input_name,
                list(batch.shape),
                np_to_triton_dtype(batch.dtype),
            )
        ]
        inputs[0].set_data_from_numpy(batch)

        outputs = [httpclient.InferRequestedOutput(self.output_name)]
        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )

        logits = result.as_numpy(self.output_name)
        if logits is None:
            raise RuntimeError(
                f"No output named '{self.output_name}' returned from Triton model '{self.model_name}'."
            )

        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        elif logits.ndim == 3:
            logits = logits[:, 0, :]
        elif logits.ndim > 3:
            logits = logits.reshape((logits.shape[0], -1))

        probs = softmax(logits)
        results = []
        for row_logits, row_probs in zip(logits, probs):
            class_index = int(np.argmax(row_probs))
            label = self.labels[class_index] if class_index < len(self.labels) else str(class_index)
            results.append(
                {
                    "status": "classified",
                    "backend": "triton",
                    "label": label,
                    "class_index": class_index,
                    "confidence": float(row_probs[class_index]),
                    "logits": row_logits.astype(float).tolist(),
                    "scores": {
                        label_name: float(row_probs[idx])
                        for idx, label_name in enumerate(self.labels)
                        if idx < len(row_probs)
                    },
                }
            )

        return results
