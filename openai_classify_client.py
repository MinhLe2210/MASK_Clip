import argparse
import base64
import json
from pathlib import Path
from typing import Any


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


def build_messages(image_ref: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Classify this image into one of the 2 configured classes.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_ref,
                    },
                },
            ],
        }
    ]


def classify_with_openai_client(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_ref: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(image_ref),
        temperature=0,
    )
    return response.choices[0].message.content


def classify_with_requests(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_ref: str,
) -> dict[str, Any]:
    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": build_messages(image_ref),
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dummy OpenAI-compatible client for the LitServe image classifier."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api-key", default="dummy", help="API key placeholder.")
    parser.add_argument(
        "--model",
        default="image-classifier-trt",
        help="Model name sent to the server.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Local image file.")
    source.add_argument("--image-url", help="HTTP(S) image URL or data URL.")
    parser.add_argument(
        "--raw-requests",
        action="store_true",
        help="Use requests instead of the OpenAI Python client.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_ref = image_file_to_data_url(args.image) if args.image else args.image_url

    if args.raw_requests:
        payload = classify_with_requests(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            image_ref=image_ref,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    content = classify_with_openai_client(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        image_ref=image_ref,
    )
    print(content)


if __name__ == "__main__":
    main()
