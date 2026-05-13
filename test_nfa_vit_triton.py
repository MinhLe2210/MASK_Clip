import argparse
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the nfa_vit model directly through Triton HTTP API."
    )
    parser.add_argument(
        "--image",
        default="image3.png",
        help="Local image path.",
    )
    parser.add_argument(
        "--triton-url",
        default="http://172.20.152.100:32455",
        help="Triton HTTP base URL.",
    )
    parser.add_argument(
        "--model-name",
        default="nfa_vit",
        help="Triton model name.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input size used for preprocessing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to pred_mask before white_ratio calculation.",
    )
    parser.add_argument(
        "--white-ratio-threshold",
        type=float,
        default=0.001,
        help="white_ratio cutoff used to derive final_label.",
    )
    parser.add_argument(
        "--mode",
        choices=["resizing", "padding"],
        default="resizing",
        help="Preprocess mode before inference.",
    )
    parser.add_argument(
        "--label-value",
        type=float,
        default=0.0,
        help="Dummy label input value sent to Triton.",
    )
    parser.add_argument(
        "--image-input-name",
        default="image",
        help="Input tensor name for the image.",
    )
    parser.add_argument(
        "--mask-input-name",
        default="mask",
        help="Input tensor name for the dummy mask.",
    )
    parser.add_argument(
        "--label-input-name",
        default="label",
        help="Input tensor name for the dummy label.",
    )
    parser.add_argument(
        "--mask-output-name",
        default="pred_mask",
        help="Output tensor name for the predicted mask.",
    )
    parser.add_argument(
        "--label-output-name",
        default="pred_label",
        help="Output tensor name for the predicted label.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--save-mask",
        default=None,
        help="Optional output path for the thresholded mask image.",
    )
    parser.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Skip Triton and model readiness checks.",
    )
    return parser.parse_args()


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
        canvas = np.asarray(
            Image.fromarray(canvas).resize((image_size, image_size), Image.BILINEAR),
            dtype=np.uint8,
        )

    arr = canvas.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def build_input_tensor(image: Image.Image, image_size: int, mode: str) -> np.ndarray:
    if mode == "padding":
        tensor = pad_top_left_and_normalize(image, image_size=image_size)
    else:
        tensor = resize_and_normalize(image, image_size=image_size)
    return np.expand_dims(tensor, axis=0).astype(np.float32)


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


def check_ready(triton_url: str, model_name: str, timeout: int) -> None:
    base_url = triton_url.rstrip("/")
    health_url = f"{base_url}/v2/health/ready"
    model_ready_url = f"{base_url}/v2/models/{model_name}/ready"

    health_response = requests.get(health_url, timeout=timeout)
    model_response = requests.get(model_ready_url, timeout=timeout)

    print(f"Triton ready: {health_response.status_code == 200}")
    print(f"Model ready : {model_response.status_code == 200}")

    health_response.raise_for_status()
    model_response.raise_for_status()


def infer(args: argparse.Namespace) -> dict:
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as f:
        image = Image.open(f).convert("RGB")

    image_np = build_input_tensor(
        image=image,
        image_size=args.image_size,
        mode=args.mode,
    )
    dummy_mask = np.zeros((1, 1, image_np.shape[2], image_np.shape[3]), dtype=np.float32)
    dummy_label = np.array([args.label_value], dtype=np.float32)

    payload = {
        "inputs": [
            {
                "name": args.image_input_name,
                "shape": list(image_np.shape),
                "datatype": "FP32",
                "data": image_np.reshape(-1).tolist(),
            },
            {
                "name": args.mask_input_name,
                "shape": list(dummy_mask.shape),
                "datatype": "FP32",
                "data": dummy_mask.reshape(-1).tolist(),
            },
            {
                "name": args.label_input_name,
                "shape": list(dummy_label.shape),
                "datatype": "FP32",
                "data": dummy_label.reshape(-1).tolist(),
            },
        ],
        "outputs": [
            {"name": args.mask_output_name},
            {"name": args.label_output_name},
        ],
    }

    infer_url = f"{args.triton_url.rstrip('/')}/v2/models/{args.model_name}/infer"
    response = requests.post(infer_url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    result = response.json()

    outputs = {item["name"]: item for item in result.get("outputs", [])}
    if args.mask_output_name not in outputs or args.label_output_name not in outputs:
        raise RuntimeError(f"Unexpected Triton response: {json.dumps(result, ensure_ascii=False)}")

    pred_mask = np.array(
        outputs[args.mask_output_name]["data"],
        dtype=np.float32,
    ).reshape(outputs[args.mask_output_name]["shape"])
    pred_label = np.array(
        outputs[args.label_output_name]["data"],
        dtype=np.float32,
    ).reshape(outputs[args.label_output_name]["shape"])

    white_pixels, total_pixels, white_ratio, binary_u8 = get_white_ratio_from_numpy(
        pred_mask,
        threshold=args.threshold,
    )
    final_label = "real" if white_ratio < args.white_ratio_threshold else "fake"

    if args.save_mask:
        output_path = Path(args.save_mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary_u8, mode="L").save(output_path)

    return {
        "image": str(image_path),
        "triton_url": args.triton_url,
        "model_name": args.model_name,
        "mode": args.mode,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "white_ratio_threshold": args.white_ratio_threshold,
        "white_pixels": white_pixels,
        "total_pixels": total_pixels,
        "white_ratio": white_ratio,
        "final_label": final_label,
        "pred_label_raw": pred_label.reshape(-1).astype(float).tolist(),
        "pred_mask_shape": list(pred_mask.shape),
        "saved_mask": args.save_mask,
    }


def main() -> None:
    args = parse_args()

    if not args.skip_ready_check:
        check_ready(args.triton_url, args.model_name, args.timeout)

    result = infer(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
