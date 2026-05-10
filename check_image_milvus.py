import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymilvus import Collection, connections, utility

from src.config import Settings
from src.triton_clients import TritonEmbeddingClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one local image against Milvus dedup collection."
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
    return parser.parse_args()


def connect_milvus(cfg: Settings) -> None:
    try:
        connections.disconnect(alias="default")
    except Exception:
        pass

    connect_kwargs: dict[str, Any] = {
        "alias": "default",
        "host": cfg.milvus_host,
        "port": cfg.milvus_port,
    }
    if cfg.milvus_database:
        connect_kwargs["db_name"] = cfg.milvus_database
    connections.connect(**connect_kwargs)


def compute_sha256(path: Path) -> tuple[str, bytes]:
    image_bytes = path.read_bytes()
    return hashlib.sha256(image_bytes).hexdigest(), image_bytes


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


def query_exact_hash(
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


def search_nearest(
    collection: Collection,
    cfg: Settings,
    image_path: Path,
    topk: int,
    field_names: set[str],
) -> list[dict[str, Any]]:
    from PIL import Image

    embedder = TritonEmbeddingClient(
        url=cfg.triton_client_url,
        model_name=cfg.dedup_model_name,
        input_name=cfg.dedup_input_name,
        output_name=cfg.dedup_output_name,
        image_size=cfg.dedup_image_size,
        embedding_dim=cfg.dedup_embedding_dim,
        timeout=cfg.request_timeout,
    )

    with Image.open(image_path) as img:
        image = img.convert("RGB").copy()

    vector = embedder.embed([image])
    output_fields = build_query_output_fields(field_names)
    results = collection.search(
        data=vector.astype("float32").tolist(),
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


def main() -> None:
    load_dotenv()
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = Settings.from_env()
    connect_milvus(cfg)

    if not utility.has_collection(cfg.collection_name, using="default"):
        raise RuntimeError(f"Milvus collection not found: {cfg.collection_name}")

    collection = Collection(cfg.collection_name, using="default")
    collection.load()

    sha256, _ = compute_sha256(image_path)

    print("=== Image ===")
    print(f"path: {image_path}")
    print(f"sha256: {sha256}")
    print()

    field_names = print_schema(collection)

    print("=== Exact SHA256 Query ===")
    exact_rows = query_exact_hash(collection, sha256, field_names)
    if exact_rows:
        print(json.dumps(exact_rows, ensure_ascii=False, indent=2))
    else:
        print("no exact hash match")
    print()

    print("=== Nearest Vector Search ===")
    nearest_hits = search_nearest(collection, cfg, image_path, args.topk, field_names)
    if nearest_hits:
        print(json.dumps(nearest_hits, ensure_ascii=False, indent=2))
    else:
        print("no nearest hits")

    try:
        connections.disconnect(alias="default")
    except Exception:
        pass


if __name__ == "__main__":
    main()
