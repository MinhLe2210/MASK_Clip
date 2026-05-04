import argparse
import base64
import binascii
import inspect
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_MODEL_PATH = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


class ImageEmbeddingWrapper(torch.nn.Module):
    """Export DINOv3 pooler_output, optionally normalized like main.py vectors."""

    def __init__(self, model: torch.nn.Module, normalize: bool):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            raise RuntimeError("Model output does not contain pooler_output.")
        if not self.normalize:
            return features

        norm = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
        return features / torch.clamp(norm, min=1e-12)


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


def load_image_file(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB").copy()


def load_image_url(url: str) -> Image.Image:
    image = load_image(url)
    return image.convert("RGB")


def request_json_to_images(path: str, max_images: int | None) -> list[Image.Image]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Request JSON must be an object.")

    if "images" in payload:
        raw_images = payload["images"]
    elif "image_base64" in payload:
        raw_images = [payload]
    else:
        raise ValueError("Request JSON must contain 'images' or 'image_base64'.")

    if not isinstance(raw_images, list):
        raise ValueError("'images' in request JSON must be a list.")

    if max_images is not None:
        raw_images = raw_images[:max_images]

    images: list[Image.Image] = []
    for idx, item in enumerate(raw_images):
        if isinstance(item, str):
            image_base64 = item
        elif isinstance(item, dict):
            image_base64 = item.get("image_base64") or item.get("b64")
        else:
            raise ValueError(f"Image item at index {idx} must be a string or object.")

        if not image_base64 or not isinstance(image_base64, str):
            raise ValueError(f"Image item at index {idx} must contain image_base64.")

        images.append(bytes_to_pil(decode_base64_to_bytes(image_base64)))

    return images


def processor_default_image_size(processor: Any) -> tuple[int, int]:
    size = getattr(processor, "size", None)

    if isinstance(size, dict):
        height = size.get("height") or size.get("shortest_edge") or size.get("longest_edge")
        width = size.get("width") or size.get("shortest_edge") or size.get("longest_edge")
        if height and width:
            return int(height), int(width)

    if isinstance(size, int):
        return int(size), int(size)

    return 224, 224


def build_sample_images(args: argparse.Namespace, processor: Any) -> list[Image.Image]:
    images: list[Image.Image] = []

    for image_path in args.image:
        images.append(load_image_file(image_path))

    for image_url in args.image_url:
        images.append(load_image_url(image_url))

    for image_base64 in args.image_base64:
        images.append(bytes_to_pil(decode_base64_to_bytes(image_base64)))

    for image_base64_path in args.image_base64_file:
        image_base64 = Path(image_base64_path).read_text(encoding="utf-8").strip()
        images.append(bytes_to_pil(decode_base64_to_bytes(image_base64)))

    if args.request_json:
        images.extend(request_json_to_images(args.request_json, args.max_sample_images))

    if images:
        return images[: args.max_sample_images] if args.max_sample_images else images

    height, width = processor_default_image_size(processor)
    return [
        Image.new("RGB", (width, height), (127, 127, 127))
        for _ in range(args.batch_size)
    ]


def export_onnx(
    wrapper: torch.nn.Module,
    pixel_values: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_spatial: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes: dict[str, dict[int, str]] = {
        "pixel_values": {0: "batch"},
        "image_embedding": {0: "batch"},
    }
    if dynamic_spatial:
        dynamic_axes["pixel_values"].update({2: "height", 3: "width"})

    export_kwargs: dict[str, Any] = {
        "model": wrapper,
        "args": (pixel_values,),
        "f": str(output_path),
        "input_names": ["pixel_values"],
        "output_names": ["image_embedding"],
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
        "do_constant_folding": True,
    }

    # Keep the classic exporter for broad transformer compatibility.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(**export_kwargs)


def check_onnx(output_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("Skip ONNX checker: package 'onnx' is not installed.")
        return

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: OK")


def verify_onnx_runtime(
    wrapper: torch.nn.Module,
    pixel_values: torch.Tensor,
    output_path: Path,
    rtol: float,
    atol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Skip ONNX Runtime verification: package 'onnxruntime' is not installed.")
        return

    with torch.no_grad():
        torch_output = wrapper(pixel_values).detach().cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["image_embedding"],
        {"pixel_values": pixel_values.detach().cpu().numpy()},
    )[0]

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    if not np.allclose(torch_output, onnx_output, rtol=rtol, atol=atol):
        raise RuntimeError(
            "ONNX Runtime output differs from PyTorch output. "
            f"max_abs_diff={max_abs_diff}, rtol={rtol}, atol={atol}"
        )

    print(f"ONNX Runtime verification: OK, max_abs_diff={max_abs_diff:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the DINOv3 image embedding model used by main.py to ONNX. "
            "The ONNX graph takes preprocessed pixel_values and returns the "
            "pooler_output embedding, normalized by default like main.py vectors."
        )
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        help=(
            "Hugging Face model path/name. Defaults to MODEL_PATH from environment "
            f"or {DEFAULT_MODEL_PATH!r}."
        ),
    )
    parser.add_argument(
        "--output",
        default="model.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Sample image file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--image-url",
        action="append",
        default=[],
        help="Sample image URL accepted by transformers.image_utils.load_image.",
    )
    parser.add_argument(
        "--image-base64",
        action="append",
        default=[],
        help="Sample image base64 string, with or without data URI prefix.",
    )
    parser.add_argument(
        "--image-base64-file",
        action="append",
        default=[],
        help="Text file containing one sample image base64 string.",
    )
    parser.add_argument(
        "--request-json",
        help=(
            "JSON payload using the same shape as main.py: "
            "{'image_base64': ...} or {'images': [...]}."
        ),
    )
    parser.add_argument(
        "--max-sample-images",
        type=int,
        default=1,
        help="Maximum images from --request-json or sample args used for export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size when no sample image is provided.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used during export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dynamic-spatial",
        action="store_true",
        help=(
            "Also mark height/width dynamic. Leave disabled for ViT-style models "
            "whose processor always resizes to a fixed size."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Return raw outputs.pooler_output exactly like the Hugging Face snippet.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    parser.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip onnx.checker validation.",
    )
    parser.add_argument(
        "--skip-runtime-verify",
        action="store_true",
        help="Skip ONNX Runtime vs PyTorch output comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for ONNX Runtime verification.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for ONNX Runtime verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path:
        raise SystemExit("Missing --model-path or MODEL_PATH environment variable.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    output_path = Path(args.output)

    print(f"Loading processor from: {args.model_path}")
    processor = AutoImageProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading model from: {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    sample_images = build_sample_images(args, processor)
    inputs = processor(images=sample_images, return_tensors="pt")
    if "pixel_values" not in inputs:
        raise RuntimeError("Processor did not return 'pixel_values'.")

    pixel_values = inputs["pixel_values"].to(device)
    wrapper = ImageEmbeddingWrapper(model, normalize=not args.no_normalize).to(device)
    wrapper.eval()

    print(f"Sample input: pixel_values shape={tuple(pixel_values.shape)}")
    export_onnx(
        wrapper=wrapper,
        pixel_values=pixel_values,
        output_path=output_path,
        opset=args.opset,
        dynamic_spatial=args.dynamic_spatial,
    )

    if not args.skip_onnx_check:
        check_onnx(output_path)

    if not args.skip_runtime_verify:
        verify_onnx_runtime(
            wrapper=wrapper,
            pixel_values=pixel_values,
            output_path=output_path,
            rtol=args.rtol,
            atol=args.atol,
        )

    print(f"Exported ONNX: {output_path.resolve()}")
    print("ONNX input : pixel_values float32 [batch, channels, height, width]")
    output_kind = (
        "normalized pooler_output" if not args.no_normalize else "raw pooler_output"
    )
    print(f"ONNX output: image_embedding float32 [batch, embedding_dim] ({output_kind})")


if __name__ == "__main__":
    main()
