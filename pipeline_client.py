import argparse
import base64
import json
from pathlib import Path
from typing import Any

import requests


def image_file_to_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Client for dedup -> TensorRT classification -> OpenAI VLM pipeline."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8002/analyze",
        help="Pipeline API URL.",
    )
    parser.add_argument("--image-id", default="img-001")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Local image file.")
    source.add_argument("--image-url", help="HTTP(S) image URL or data URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_ref = image_file_to_data_url(args.image) if args.image else args.image_url

    image_item: dict[str, Any] = {"image_id": args.image_id}
    if image_ref.startswith(("http://", "https://")):
        image_item["image_url"] = image_ref
    else:
        image_item["image_base64"] = image_ref

    response = requests.post(
        args.url,
        json={
            "request_id": "demo-pipeline",
            "images": [image_item],
        },
        timeout=180,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
