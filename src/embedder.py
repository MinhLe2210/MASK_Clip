import ctypes
import logging
import os
import threading

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from PIL import Image

from src.config import Settings


logger = logging.getLogger("image-dedup-litserve")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    if not os.path.exists(engine_file_path):
        raise FileNotFoundError(f"TensorRT engine not found: {engine_file_path}")

    logger.info("Reading TensorRT engine from %s", engine_file_path)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_file_path}")

    return engine


class ImageEmbedder:
    """
    TensorRT Image Embedder.

    Expected:
    - Input tensor: shape (B, 3, H, W)
    - Output tensor:
        - (B, D): embedding directly
        - (B, N, D): hidden states, will use mean pooling over N
    """

    def __init__(self, cfg: Settings, device=None):
        self.cfg = cfg
        self.lock = threading.Lock()

        self.engine_path = getattr(cfg, "dedup_engine_path", None) or cfg.model_path
        self.device_id = self._parse_device_id(device)

        err, = cudart.cudaSetDevice(self.device_id)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaSetDevice({self.device_id}) failed: {err}")

        logger.info(
            "Loading TensorRT engine from %s on cuda:%s",
            self.engine_path,
            self.device_id,
        )

        self.engine = load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()

        self.input_name, self.output_names = self._get_io_names()

        if len(self.output_names) < 1:
            raise RuntimeError("TensorRT engine has no output tensor")

        cfg_output_name = getattr(cfg, "output_name", None)
        if cfg_output_name:
            if cfg_output_name not in self.output_names:
                raise ValueError(
                    f"cfg.output_name={cfg_output_name} not found in engine outputs: "
                    f"{self.output_names}"
                )
            self.output_name = cfg_output_name
        else:
            self.output_name = self.output_names[0]

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))

        logger.info(
            "TensorRT IO: input=%s dtype=%s, outputs=%s, selected_output=%s",
            self.input_name,
            self.input_dtype,
            self.output_names,
            self.output_name,
        )

    @staticmethod
    def _parse_device_id(device) -> int:
        if device is None:
            return 0

        if isinstance(device, int):
            return device

        device_str = str(device)
        if ":" in device_str:
            return int(device_str.split(":")[-1])

        return 0

    def _get_io_names(self):
        input_name = None
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_names.append(name)

        if input_name is None:
            raise RuntimeError("No input tensor found in TensorRT engine")

        return input_name, output_names

    def _get_input_hw(self) -> tuple[int, int]:
        shape = tuple(self.engine.get_tensor_shape(self.input_name))

        if len(shape) != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {shape}")

        _, c, h, w = shape

        if c not in (3, -1):
            raise ValueError(f"Expected input channel=3, got shape={shape}")

        if h > 0 and w > 0:
            return int(h), int(w)

        image_size = getattr(self.cfg, "image_size", None)
        input_height = getattr(self.cfg, "input_height", None)
        input_width = getattr(self.cfg, "input_width", None)

        if image_size is not None:
            return int(image_size), int(image_size)

        if input_height is not None and input_width is not None:
            return int(input_height), int(input_width)

        raise ValueError(
            "Engine has dynamic H/W. Please set cfg.image_size or "
            "cfg.input_height/cfg.input_width"
        )

    def _preprocess_one(self, image: Image.Image, height: int, width: int) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.BICUBIC)

        arr = np.asarray(image).astype(np.float32) / 255.0

        mean = np.array(
            getattr(self.cfg, "image_mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        )
        std = np.array(
            getattr(self.cfg, "image_std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        )

        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))

        return arr.astype(np.float32)

    def _preprocess(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("images must not be empty")

        height, width = self._get_input_hw()

        batch = np.stack(
            [self._preprocess_one(img, height, width) for img in images],
            axis=0,
        )

        batch = np.ascontiguousarray(batch.astype(self.input_dtype))

        return batch

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4:
            raise ValueError(f"Expected batch shape (B, C, H, W), got {batch.shape}")

        if batch.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {batch.shape[1]}")

        batch = np.ascontiguousarray(batch)

        host_output_ptr = None
        device_input = None
        device_output = None
        stream = None

        try:
            ok = self.context.set_input_shape(self.input_name, batch.shape)
            if not ok:
                raise RuntimeError(
                    f"Failed to set input shape {batch.shape} for tensor {self.input_name}"
                )

            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(f"Unresolved tensors after infer_shapes(): {unresolved}")

            input_shape = tuple(self.context.get_tensor_shape(self.input_name))
            output_shape = tuple(self.context.get_tensor_shape(self.output_name))

            logger.debug("TRT input shape: %s", input_shape)
            logger.debug("TRT output shape: %s", output_shape)

            output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            output_size = trt.volume(output_shape)

            if output_size <= 0:
                raise RuntimeError(f"Invalid output shape: {output_shape}")

            output_nbytes = output_size * np.dtype(output_dtype).itemsize

            err, device_input = cudart.cudaMalloc(batch.nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc input failed: {err}")

            self.context.set_tensor_address(self.input_name, device_input)

            err, host_output_ptr = cudart.cudaMallocHost(output_nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMallocHost output failed: {err}")

            ptr_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(output_dtype))
            host_output_arr = np.ctypeslib.as_array(
                ctypes.cast(host_output_ptr, ptr_type),
                shape=(output_size,),
            )

            err, device_output = cudart.cudaMalloc(output_nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc output failed: {err}")

            self.context.set_tensor_address(self.output_name, device_output)

            err, stream = cudart.cudaStreamCreate()
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamCreate failed: {err}")

            err, = cudart.cudaMemcpyAsync(
                device_input,
                batch.ctypes.data,
                batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err}")

            ok = self.context.execute_async_v3(stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")

            err, = cudart.cudaMemcpyAsync(
                host_output_arr.ctypes.data,
                device_output,
                output_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync D2H failed: {err}")

            err, = cudart.cudaStreamSynchronize(stream)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

            output = np.array(host_output_arr).reshape(output_shape)

            return output.astype(np.float32)

        finally:
            if device_input is not None:
                cudart.cudaFree(device_input)

            if device_output is not None:
                cudart.cudaFree(device_output)

            if host_output_ptr is not None:
                cudart.cudaFreeHost(host_output_ptr)

            if stream is not None:
                cudart.cudaStreamDestroy(stream)

    def extract(self, images: list[Image.Image]) -> np.ndarray:
        batch = self._preprocess(images)

        with self.lock:
            output = self._infer(batch)

        if output.ndim == 3:
            vectors = output.mean(axis=1)
        elif output.ndim == 2:
            vectors = output
        else:
            vectors = output.reshape(output.shape[0], -1)

        vectors = vectors.astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors.astype(np.float32)
