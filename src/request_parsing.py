import uuid
from typing import Any

from src.config import Settings


def decode_dedup_request(
    request: dict[str, Any],
    cfg: Settings,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("Request body must be JSON object.")

    request_id = str(request.get("request_id") or uuid.uuid4())

    if "images" in request:
        images = request["images"]
    elif "image_base64" in request:
        images = [
            {
                "image_id": request.get("image_id"),
                "image_base64": request["image_base64"],
            }
        ]
    else:
        raise ValueError("Request must contain 'images' or 'image_base64'.")

    if not isinstance(images, list):
        raise ValueError("'images' must be a list.")

    if len(images) == 0:
        return {"request_id": request_id, "items": []}

    if len(images) > cfg.max_images_per_request:
        raise ValueError(f"Too many images. Max is {cfg.max_images_per_request}.")

    items = []
    for idx, item in enumerate(images):
        if isinstance(item, str):
            image_id = f"{request_id}:{idx}"
            image_b64 = item
        elif isinstance(item, dict):
            image_id = str(item.get("image_id") or f"{request_id}:{idx}")
            image_b64 = item.get("image_base64") or item.get("b64")
        else:
            raise ValueError("Each image must be a string or object.")

        if not image_b64 or not isinstance(image_b64, str):
            raise ValueError("Each image must contain image_base64.")

        items.append(
            {
                "request_id": request_id,
                "image_id": image_id,
                "image_base64": image_b64,
            }
        )

    return {
        "request_id": request_id,
        "items": items,
    }


def batch_dedup_requests(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    flat_items = []
    request_ids = []
    request_sizes = []

    for req in inputs:
        request_ids.append(req["request_id"])
        request_sizes.append(len(req["items"]))
        flat_items.extend(req["items"])

    return {
        "request_ids": request_ids,
        "request_sizes": request_sizes,
        "flat_items": flat_items,
    }


def unbatch_dedup_output(
    output: dict[str, Any],
    milvus_recovered_this_predict: int,
    last_milvus_error: str | None,
) -> list[dict[str, Any]]:
    results = output["flat_results"]
    request_ids = output["request_ids"]
    request_sizes = output["request_sizes"]

    responses = []
    cursor = 0

    for request_id, size in zip(request_ids, request_sizes):
        request_results = results[cursor : cursor + size]
        cursor += size

        responses.append(
            {
                "request_id": request_id,
                "total_images": size,
                "unique_count": sum(
                    1 for r in request_results if r and r.get("status") == "unique"
                ),
                "duplicate_count": sum(
                    1
                    for r in request_results
                    if r and r.get("status") == "duplicate"
                ),
                "error_count": sum(
                    1 for r in request_results if r and r.get("status") == "error"
                ),
                "results": request_results,
                "milvus_recovered_this_predict": milvus_recovered_this_predict,
                "last_milvus_error": last_milvus_error,
            }
        )

    return responses
