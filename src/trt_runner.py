from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


class TensorRTRunner:
    """
    TensorRT runner using cuda-python / cuda.bindings, not PyCUDA.

    Supports TensorRT 10.x tensor API and keeps a small TensorRT 8.x fallback.
    Input is expected to be a numpy array. By default it is converted to
    contiguous float32 before inference, matching image embedder input tensors.
    """

    def __init__(
        self,
        engine_path: str,
        input_name: str,
        output_name: str | None = None,
    ) -> None:
        self.trt = trt
        self.cudart = cudart
        self.input_name = input_name
        self.output_name = output_name
        self.logger = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        if hasattr(self.engine, "num_io_tensors"):
            self.api = "v10"
            self.input_names, self.output_names = self._io_names_v10()
        else:
            self.api = "v8"
            self.input_names, self.output_names = self._io_names_v8()

        if self.input_name not in self.input_names:
            raise ValueError(
                f"Input tensor {self.input_name!r} not found. "
                f"Available inputs: {self.input_names}"
            )

        if self.output_name is None:
            if len(self.output_names) != 1:
                raise ValueError(
                    "Output name is required when the engine has multiple outputs: "
                    f"{self.output_names}"
                )
            self.output_name = self.output_names[0]
        elif self.output_name not in self.output_names:
            raise ValueError(
                f"Output tensor {self.output_name!r} not found. "
                f"Available outputs: {self.output_names}"
            )

    def infer(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        input_array = np.ascontiguousarray(input_array.astype(np.float32, copy=False))

        if self.api == "v10":
            return self._infer_v10(input_array)
        return self._infer_v8(input_array)

    def _check_cuda(self, err: Any, message: str) -> None:
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{message}: {err}")

    def _io_names_v10(self) -> tuple[list[str], list[str]]:
        input_names: list[str] = []
        output_names: list[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        return input_names, output_names

    def _io_names_v8(self) -> tuple[list[str], list[str]]:
        input_names: list[str] = []
        output_names: list[str] = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                input_names.append(name)
            else:
                output_names.append(name)

        return input_names, output_names

    def _malloc_device(self, nbytes: int):
        err, ptr = cudart.cudaMalloc(nbytes)
        self._check_cuda(err, "cudaMalloc failed")
        return ptr

    def _malloc_host_array(self, shape: tuple[int, ...], dtype: np.dtype):
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize

        err, host_ptr = cudart.cudaMallocHost(nbytes)
        self._check_cuda(err, "cudaMallocHost failed")

        ptr_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
        flat = np.ctypeslib.as_array(ctypes.cast(host_ptr, ptr_type), (size,))
        array = flat.reshape(shape)
        return host_ptr, array

    def _infer_v10(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        ok = self.context.set_input_shape(self.input_name, tuple(input_array.shape))
        if not ok:
            engine_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            raise ValueError(
                f"Failed to set input shape for {self.input_name!r}. "
                f"Engine shape={engine_shape}, input shape={tuple(input_array.shape)}"
            )

        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"Unresolved tensors after infer_shapes(): {unresolved}")

        device_ptrs: list[Any] = []
        host_ptrs: list[Any] = []
        output_device_ptrs: dict[str, Any] = {}
        outputs: dict[str, np.ndarray] = {}
        stream = None

        try:
            input_device = self._malloc_device(input_array.nbytes)
            device_ptrs.append(input_device)
            self.context.set_tensor_address(self.input_name, input_device)

            for output_name in self.output_names:
                output_shape = tuple(self.context.get_tensor_shape(output_name))
                if any(dim < 0 for dim in output_shape):
                    raise ValueError(f"Invalid output shape for {output_name}: {output_shape}")

                output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(output_name)))
                host_ptr, host_output = self._malloc_host_array(output_shape, output_dtype)
                host_ptrs.append(host_ptr)

                output_device = self._malloc_device(host_output.nbytes)
                device_ptrs.append(output_device)

                self.context.set_tensor_address(output_name, output_device)
                output_device_ptrs[output_name] = output_device
                outputs[output_name] = host_output

            err, stream = cudart.cudaStreamCreate()
            self._check_cuda(err, "cudaStreamCreate failed")

            err, = cudart.cudaMemcpyAsync(
                input_device,
                input_array.ctypes.data,
                input_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
            self._check_cuda(err, "cudaMemcpyAsync H2D failed")

            ok = self.context.execute_async_v3(stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")

            for output_name, host_output in outputs.items():
                err, = cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    output_device_ptrs[output_name],
                    host_output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
                self._check_cuda(err, f"cudaMemcpyAsync D2H failed for {output_name}")

            err, = cudart.cudaStreamSynchronize(stream)
            self._check_cuda(err, "cudaStreamSynchronize failed")

            return {
                name: np.array(output, copy=True).astype(np.float32, copy=False)
                for name, output in outputs.items()
            }

        finally:
            if stream is not None:
                cudart.cudaStreamDestroy(stream)
            for ptr in device_ptrs:
                cudart.cudaFree(ptr)
            for ptr in host_ptrs:
                cudart.cudaFreeHost(ptr)

    def _infer_v8(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        input_idx = self.engine.get_binding_index(self.input_name)
        if not self.engine.is_shape_binding(input_idx):
            self.context.set_binding_shape(input_idx, tuple(input_array.shape))

        bindings = [0] * self.engine.num_bindings
        device_ptrs: list[Any] = []
        host_ptrs: list[Any] = []
        output_device_ptrs: dict[str, Any] = {}
        outputs: dict[str, np.ndarray] = {}
        stream = None

        try:
            input_device = self._malloc_device(input_array.nbytes)
            device_ptrs.append(input_device)
            bindings[input_idx] = input_device

            for output_name in self.output_names:
                output_idx = self.engine.get_binding_index(output_name)
                output_shape = tuple(self.context.get_binding_shape(output_idx))
                if any(dim < 0 for dim in output_shape):
                    raise ValueError(f"Invalid output shape for {output_name}: {output_shape}")

                output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(output_idx)))
                host_ptr, host_output = self._malloc_host_array(output_shape, output_dtype)
                host_ptrs.append(host_ptr)

                output_device = self._malloc_device(host_output.nbytes)
                device_ptrs.append(output_device)
                bindings[output_idx] = output_device

                output_device_ptrs[output_name] = output_device
                outputs[output_name] = host_output

            err, stream = cudart.cudaStreamCreate()
            self._check_cuda(err, "cudaStreamCreate failed")

            err, = cudart.cudaMemcpyAsync(
                input_device,
                input_array.ctypes.data,
                input_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
            self._check_cuda(err, "cudaMemcpyAsync H2D failed")

            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 failed")

            for output_name, host_output in outputs.items():
                err, = cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    output_device_ptrs[output_name],
                    host_output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
                self._check_cuda(err, f"cudaMemcpyAsync D2H failed for {output_name}")

            err, = cudart.cudaStreamSynchronize(stream)
            self._check_cuda(err, "cudaStreamSynchronize failed")

            return {
                name: np.array(output, copy=True).astype(np.float32, copy=False)
                for name, output in outputs.items()
            }

        finally:
            if stream is not None:
                cudart.cudaStreamDestroy(stream)
            for ptr in device_ptrs:
                cudart.cudaFree(ptr)
            for ptr in host_ptrs:
                cudart.cudaFreeHost(ptr)
