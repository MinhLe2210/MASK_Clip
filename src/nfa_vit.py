from typing import Any

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

from src.triton_clients import MEAN, STD, TritonModelClient


def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def resize_and_normalize(image: Image.Image, image_size: int) -> np.ndarray:
    image = pil_to_rgb(image)
    image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def pad_top_left_and_normalize(image: Image.Image, image_size: int) -> np.ndarray:
    image = pil_to_rgb(image)
    src = np.asarray(image, dtype=np.uint8)
    height, width = src.shape[:2]

    new_height = max(height, image_size)
    new_width = max(width, image_size)

    canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    canvas[:height, :width] = src

    if new_height != image_size or new_width != image_size:
        resized = Image.fromarray(canvas).resize((image_size, image_size), Image.BILINEAR)
        canvas = np.asarray(resized, dtype=np.uint8)

    arr = canvas.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def get_white_ratio_from_numpy(
    pred_mask_np: np.ndarray,
    threshold: float,
) -> tuple[int, int, float, np.ndarray]:
    mask_np = np.squeeze(pred_mask_np)
    mask_np = np.clip(mask_np, 0.0, 1.0)
    binary_u8 = ((mask_np >= threshold) * 255).astype(np.uint8)

    white_pixels = int((binary_u8 == 255).sum())
    total_pixels = int(binary_u8.shape[0] * binary_u8.shape[1])
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
    return white_pixels, total_pixels, white_ratio, binary_u8


class TritonNfaVitClient(TritonModelClient):
    def __init__(
        self,
        url: str,
        model_name: str,
        image_input_name: str,
        mask_input_name: str,
        label_input_name: str,
        mask_output_name: str,
        label_output_name: str,
        image_size: int,
        threshold: float,
        white_ratio_threshold: float,
        label_value: float,
        preprocess_mode: str,
        timeout: float,
    ):
        super().__init__(url=url, model_name=model_name, timeout=timeout)
        if preprocess_mode not in {"resizing", "padding"}:
            raise ValueError(
                "NFA_PREPROCESS_MODE must be either 'resizing' or 'padding'."
            )

        self.image_input_name = image_input_name
        self.mask_input_name = mask_input_name
        self.label_input_name = label_input_name
        self.mask_output_name = mask_output_name
        self.label_output_name = label_output_name
        self.image_size = image_size
        self.threshold = threshold
        self.white_ratio_threshold = white_ratio_threshold
        self.label_value = label_value
        self.preprocess_mode = preprocess_mode

    def _build_input_tensor(self, image: Image.Image) -> np.ndarray:
        if self.preprocess_mode == "padding":
            tensor = pad_top_left_and_normalize(image, self.image_size)
        else:
            tensor = resize_and_normalize(image, self.image_size)
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def infer(self, image: Image.Image) -> dict[str, Any]:
        image_np = self._build_input_tensor(image)
        dummy_mask = np.zeros(
            (1, 1, image_np.shape[2], image_np.shape[3]),
            dtype=np.float32,
        )
        dummy_label = np.array([self.label_value], dtype=np.float32)

        inputs = [
            httpclient.InferInput(
                self.image_input_name,
                list(image_np.shape),
                np_to_triton_dtype(image_np.dtype),
            ),
            httpclient.InferInput(
                self.mask_input_name,
                list(dummy_mask.shape),
                np_to_triton_dtype(dummy_mask.dtype),
            ),
            httpclient.InferInput(
                self.label_input_name,
                list(dummy_label.shape),
                np_to_triton_dtype(dummy_label.dtype),
            ),
        ]
        inputs[0].set_data_from_numpy(image_np)
        inputs[1].set_data_from_numpy(dummy_mask)
        inputs[2].set_data_from_numpy(dummy_label)

        outputs = [
            httpclient.InferRequestedOutput(self.mask_output_name),
            httpclient.InferRequestedOutput(self.label_output_name),
        ]
        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )

        pred_mask = result.as_numpy(self.mask_output_name)
        pred_label = result.as_numpy(self.label_output_name)
        if pred_mask is None:
            raise RuntimeError(
                f"No output named '{self.mask_output_name}' returned from Triton model '{self.model_name}'."
            )
        if pred_label is None:
            raise RuntimeError(
                f"No output named '{self.label_output_name}' returned from Triton model '{self.model_name}'."
            )

        white_pixels, total_pixels, white_ratio, _ = get_white_ratio_from_numpy(
            pred_mask,
            threshold=self.threshold,
        )
        final_label = "real" if white_ratio < self.white_ratio_threshold else "fake"

        return {
            "status": "classified",
            "backend": "triton",
            "mode": self.preprocess_mode,
            "threshold": float(self.threshold),
            "white_ratio_threshold": float(self.white_ratio_threshold),
            "white_pixels": int(white_pixels),
            "total_pixels": int(total_pixels),
            "white_ratio": float(white_ratio),
            "pred_label_raw": pred_label.reshape(-1).astype(np.float32).tolist(),
            "pred_mask_shape": list(pred_mask.shape),
            "final_label": final_label,
        }
