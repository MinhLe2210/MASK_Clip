import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

import requests


def encode_image_base64(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_payload(
    image_path: str | None = None,
    image_url: str | None = None,
    image_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    if request_id:
        payload["request_id"] = request_id

    if image_path:
        payload["image_id"] = image_id or Path(image_path).name
        payload["image_base64"] = encode_image_base64(image_path)
        return payload

    if image_url:
        payload["image_id"] = image_id or "image_url_0"
        payload["image_url"] = image_url
        return payload

    raise ValueError("You must provide either --image or --image-url")


def check_health(base_url: str, timeout: int) -> None:
    url = f"{base_url.rstrip('/')}/health"
    response = requests.get(url, timeout=timeout)

    print("Health status:", response.status_code)
    print(response.text)

    response.raise_for_status()


def analyze(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/analyze"

    response = requests.post(url, json=payload, timeout=timeout)

    print("Status:", response.status_code)
    print("Raw response:")
    print(response.text)

    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for AI detector pipeline API")
    parser.add_argument(
        "--url",
        default="http://172.20.152.112:31129",
        help="Pipeline API base URL",
    )
    parser.add_argument(
        "--image",
        default="image.png",
        help="Local image path",
    )
    parser.add_argument(
        "--image-url",
        default=None,
        help="Remote image URL instead of local image",
    )
    parser.add_argument(
        "--image-id",
        default=None,
        help="Optional image_id",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Optional request_id",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip /health check before /analyze",
    )

    args = parser.parse_args()

    try:
        if not args.skip_health:
            check_health(args.url, args.timeout)

        payload = make_payload(
            image_path=None if args.image_url else args.image,
            image_url=args.image_url,
            image_id=args.image_id,
            request_id=args.request_id,
        )

        result = analyze(args.url, payload, args.timeout)

        print("\nParsed JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.HTTPError as exc:
        print("\nHTTP error:", exc, file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print("\nError:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()