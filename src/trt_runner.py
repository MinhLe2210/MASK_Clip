from __future__ import annotations

from typing import Any

import numpy as np


class TensorRTRunner:
    def __init__(
        self,
        engine_path: str,
        input_name: str,
        output_name: str | None = None,
    ) -> None:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runner requires packages: tensorrt and pycuda."
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
        input_array = np.ascontiguousarray(input_array.astype(np.float32))
        if self.api == "v10":
            return self._infer_v10(input_array)
        return self._infer_v8(input_array)

    def _io_names_v10(self) -> tuple[list[str], list[str]]:
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

        return input_names, output_names

    def _io_names_v8(self) -> tuple[list[str], list[str]]:
        input_names = []
        output_names = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                input_names.append(name)
            else:
                output_names.append(name)

        return input_names, output_names

    def _infer_v10(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        cuda = self.cuda
        trt = self.trt

        self.context.set_input_shape(self.input_name, tuple(input_array.shape))

        device_allocations: list[Any] = []
        input_mem = cuda.mem_alloc(input_array.nbytes)
        device_allocations.append(input_mem)
        self.context.set_tensor_address(self.input_name, int(input_mem))

        outputs: dict[str, np.ndarray] = {}
        output_mems = {}
        for output_name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(output_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))
            output = np.empty(shape, dtype=dtype)
            output_mem = cuda.mem_alloc(output.nbytes)
            device_allocations.append(output_mem)
            self.context.set_tensor_address(output_name, int(output_mem))
            outputs[output_name] = output
            output_mems[output_name] = output_mem

        cuda.memcpy_htod_async(input_mem, input_array, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for output_name, output in outputs.items():
            cuda.memcpy_dtoh_async(output, output_mems[output_name], self.stream)

        self.stream.synchronize()
        return {name: output.astype(np.float32) for name, output in outputs.items()}

    def _infer_v8(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        cuda = self.cuda
        trt = self.trt

        input_idx = self.engine.get_binding_index(self.input_name)
        if not self.engine.is_shape_binding(input_idx):
            self.context.set_binding_shape(input_idx, tuple(input_array.shape))

        bindings = [0] * self.engine.num_bindings
        device_allocations: list[Any] = []

        input_mem = cuda.mem_alloc(input_array.nbytes)
        device_allocations.append(input_mem)
        bindings[input_idx] = int(input_mem)

        outputs: dict[str, np.ndarray] = {}
        output_mems = {}
        for output_name in self.output_names:
            output_idx = self.engine.get_binding_index(output_name)
            shape = tuple(self.context.get_binding_shape(output_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(output_idx))
            output = np.empty(shape, dtype=dtype)
            output_mem = cuda.mem_alloc(output.nbytes)
            device_allocations.append(output_mem)
            bindings[output_idx] = int(output_mem)
            outputs[output_name] = output
            output_mems[output_name] = output_mem

        cuda.memcpy_htod_async(input_mem, input_array, self.stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        for output_name, output in outputs.items():
            cuda.memcpy_dtoh_async(output, output_mems[output_name], self.stream)

        self.stream.synchronize()
        return {name: output.astype(np.float32) for name, output in outputs.items()}
