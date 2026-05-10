import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymilvus import Collection, connections

from src.config import Settings
from src.milvus_store import MilvusDedupStore
from src.triton_clients import TritonEmbeddingClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one local image against Milvus, optionally insert one dummy "
            "row from its embedding, then test dedup behavior."
        )
    )
    parser.add_argument(
        "--image",
        default="image.png",
        help="Local image path. Default: image.png",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of nearest Milvus hits to print. Default: 3",
    )
    parser.add_argument(
        "--insert-dummy",
        action="store_true",
        help="Insert one dummy row into Milvus using the image embedding.",
    )
    parser.add_argument(
        "--dummy-match",
        choices=["vector", "exact"],
        default="vector",
        help=(
            "Dummy insert mode. 'vector' keeps the same embedding but uses a different "
            "sha256 so vector dedup can be tested. 'exact' uses the real sha256."
        ),
    )
    parser.add_argument(
        "--dummy-image-id",
        default=None,
        help="Optional custom image_id for the dummy row.",
    )
    parser.add_argument(
        "--cleanup-dummy",
        action="store_true",
        help="Delete the inserted dummy row at the end of the script.",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pretty_type(field: Any) -> str:
    dtype = getattr(field, "dtype", None)
    if dtype is None:
        return "unknown"
    return str(dtype)


def safe_json_loads(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def simplify_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "classification_json" in out:
        out["classification_json"] = safe_json_loads(out["classification_json"])
    if "vlm_json" in out:
        out["vlm_json"] = safe_json_loads(out["vlm_json"])
    return out


def build_query_output_fields(field_names: set[str]) -> list[str]:
    preferred = [
        "image_id",
        "image_sha256",
        "pipeline_stage",
        "classification_json",
        "vlm_json",
        "openai_skipped",
        "skip_reason",
    ]
    return [name for name in preferred if name in field_names]


def print_schema(collection: Collection) -> set[str]:
    fields = getattr(collection.schema, "fields", None) or []
    field_names = {getattr(field, "name", "") for field in fields}
    print("=== Collection ===")
    print(f"name: {collection.name}")
    print(f"num_entities: {collection.num_entities}")
    print("schema:")
    for field in fields:
        name = getattr(field, "name", "unknown")
        params = getattr(field, "params", None) or {}
        extras = []
        if getattr(field, "is_primary", False):
            extras.append("primary")
        if getattr(field, "auto_id", False):
            extras.append("auto_id")
        if params:
            extras.append(str(params))
        extras_str = f" ({', '.join(extras)})" if extras else ""
        print(f"  - {name}: {pretty_type(field)}{extras_str}")
    print()
    return field_names


def query_hash_rows(
    collection: Collection,
    sha256: str,
    field_names: set[str],
) -> list[dict[str, Any]]:
    output_fields = build_query_output_fields(field_names)
    rows = collection.query(
        expr=f'image_sha256 in ["{sha256}"]',
        output_fields=output_fields,
    )
    return [simplify_row(dict(row)) for row in rows]


def print_hash_query(
    collection: Collection,
    sha256: str,
    field_names: set[str],
    title: str,
) -> list[dict[str, Any]]:
    print(f"=== {title} ===")
    rows = query_hash_rows(collection, sha256, field_names)
    if rows:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print("no rows")
    print()
    return rows


def load_embedder(cfg: Settings) -> TritonEmbeddingClient:
    return TritonEmbeddingClient(
        url=cfg.triton_client_url,
        model_name=cfg.dedup_model_name,
        input_name=cfg.dedup_input_name,
        output_name=cfg.dedup_output_name,
        image_size=cfg.dedup_image_size,
        embedding_dim=cfg.dedup_embedding_dim,
        timeout=cfg.request_timeout,
    )


def embed_local_image(cfg: Settings, image_path: Path) -> Any:
    from PIL import Image

    embedder = load_embedder(cfg)
    with Image.open(image_path) as img:
        image = img.convert("RGB").copy()
    return embedder.embed([image])[0]


def search_nearest(
    collection: Collection,
    cfg: Settings,
    vector: Any,
    topk: int,
    field_names: set[str],
) -> list[dict[str, Any]]:
    output_fields = build_query_output_fields(field_names)
    results = collection.search(
        data=[vector.astype("float32").tolist()],
        anns_field="vector",
        param={
            "metric_type": "IP",
            "params": {"ef": cfg.search_ef},
        },
        limit=topk,
        output_fields=output_fields,
    )

    hits: list[dict[str, Any]] = []
    for batch in results:
        for hit in batch:
            item = {
                "milvus_id": int(hit.id),
                "score": float(hit.score),
            }
            entity = hit.entity
            for field in output_fields:
                item[field] = entity.get(field)
            hits.append(simplify_row(item))
    return hits


def build_dummy_sha256(actual_sha256: str, mode: str) -> str:
    if mode == "exact":
        return actual_sha256
    return hashlib.sha256(f"debug-vector::{actual_sha256}".encode("utf-8")).hexdigest()


def build_dummy_image_id(actual_sha256: str, mode: str, custom_image_id: str | None) -> str:
    if custom_image_id:
        return custom_image_id
    return f"debug:{mode}:{actual_sha256[:16]}"


def build_dummy_row(
    cfg: Settings,
    actual_sha256: str,
    vector: Any,
    mode: str,
    custom_image_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = cfg.classification_labels or ["fake", "real"]
    primary_label = "fake" if "fake" in labels else labels[0]
    secondary_labels = [label for label in labels if label != primary_label]
    scores = {label: 0.0 for label in labels}
    scores[primary_label] = 1.0
    for label in secondary_labels:
        scores[label] = 0.0

    dummy_sha256 = build_dummy_sha256(actual_sha256, mode)
    dummy_image_id = build_dummy_image_id(actual_sha256, mode, custom_image_id)
    classification = {
        "status": "classified",
        "backend": "debug_script",
        "label": primary_label,
        "class_index": labels.index(primary_label),
        "confidence": 1.0,
        "logits": [1.0 if label == primary_label else 0.0 for label in labels],
        "scores": scores,
    }

    row = {
        "flat_idx": 0,
        "image_id": dummy_image_id,
        "sha256": dummy_sha256,
        "vector": vector,
        "pipeline_stage": "classification",
        "classification": classification,
        "vlm": None,
        "openai_skipped": True,
        "skip_reason": "debug_dummy_insert",
    }
    result = {
        "image_id": dummy_image_id,
        "status": "unique",
        "sha256": dummy_sha256,
        "inserted": False,
    }
    return row, result


def insert_dummy_row(
    store: MilvusDedupStore,
    collection: Collection,
    field_names: set[str],
    cfg: Settings,
    actual_sha256: str,
    vector: Any,
    mode: str,
    custom_image_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    row, result = build_dummy_row(cfg, actual_sha256, vector, mode, custom_image_id)
    results = [result]

    print("=== Dummy Insert Plan ===")
    print(
        json.dumps(
            {
                "image_id": row["image_id"],
                "dummy_match": mode,
                "actual_sha256": actual_sha256,
                "insert_sha256": row["sha256"],
                "pipeline_stage": row["pipeline_stage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print()

    existing_dummy = query_hash_rows(collection, row["sha256"], field_names)
    if existing_dummy:
        print("=== Dummy Row Already Exists ===")
        print(json.dumps(existing_dummy, ensure_ascii=False, indent=2))
        print()
        return row, results[0]

    print("=== Dummy Insert Attempt ===")
    try:
        store.insert_processed_images([row], results)
        print("insert ok")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
        print()
        return row, results[0]
    except Exception as exc:
        print("insert failed")
        print(repr(exc))
        print()
        traceback.print_exc()
        print()
        return None


def cleanup_dummy_row(collection: Collection, sha256: str) -> None:
    print("=== Cleanup Dummy Row ===")
    result = collection.delete(expr=f'image_sha256 in ["{sha256}"]')
    print(f"delete expr sha256={sha256}")
    print(f"delete result: {result}")
    print()


def main() -> None:
    load_dotenv()
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = Settings.from_env()
    store = MilvusDedupStore(cfg)
    collection = store.connect_and_prepare_collection()
    field_names = print_schema(collection)

    actual_sha256 = compute_sha256(image_path)
    vector = embed_local_image(cfg, image_path)

    print("=== Image ===")
    print(f"path: {image_path}")
    print(f"sha256: {actual_sha256}")
    print(f"embedding_dim: {len(vector)}")
    print()

    print_hash_query(
        collection=collection,
        sha256=actual_sha256,
        field_names=field_names,
        title="Exact SHA256 Query Before Insert",
    )

    inserted_row = None
    if args.insert_dummy:
        inserted_row = insert_dummy_row(
            store=store,
            collection=collection,
            field_names=field_names,
            cfg=cfg,
            actual_sha256=actual_sha256,
            vector=vector,
            mode=args.dummy_match,
            custom_image_id=args.dummy_image_id,
        )
        if inserted_row is not None:
            dummy_row, _ = inserted_row
            print_hash_query(
                collection=collection,
                sha256=dummy_row["sha256"],
                field_names=field_names,
                title="Dummy SHA256 Query After Insert",
            )

    print("=== Nearest Vector Search ===")
    nearest_hits = search_nearest(collection, cfg, vector, args.topk, field_names)
    if nearest_hits:
        print(json.dumps(nearest_hits, ensure_ascii=False, indent=2))
    else:
        print("no nearest hits")
    print()

    if args.insert_dummy and inserted_row is not None and args.cleanup_dummy:
        cleanup_dummy_row(collection, inserted_row[0]["sha256"])

    try:
        connections.disconnect(alias="default")
    except Exception:
        pass


if __name__ == "__main__":
    main()
