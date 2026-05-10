import json
import logging
import time
from threading import Lock
from typing import Any, Callable

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.config import Settings

try:
    from pymilvus.exceptions import ConnectionNotExistException, MilvusException
except Exception:
    from pymilvus.exceptions import MilvusException

    ConnectionNotExistException = MilvusException


logger = logging.getLogger("pipeline-fastapi")

CACHE_JSON_MAX_LENGTH = 65535
CACHE_SKIP_REASON_MAX_LENGTH = 512


class MilvusDedupStore:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.collection: Collection | None = None
        self._lock = Lock()
        self.recovered_this_predict = 0
        self.last_error: str | None = None

    def reset_predict_state(self) -> None:
        self.recovered_this_predict = 0
        self.last_error = None

    def connect_and_prepare_collection(self) -> Collection:
        with self._lock:
            try:
                connections.disconnect(alias="default")
            except Exception:
                pass

            connect_kwargs: dict[str, Any] = {
                "alias": "default",
                "host": self.cfg.milvus_host,
                "port": self.cfg.milvus_port,
            }
            if self.cfg.milvus_database:
                connect_kwargs["db_name"] = self.cfg.milvus_database

            connections.connect(**connect_kwargs)

            if utility.has_collection(self.cfg.collection_name, using="default"):
                collection = Collection(self.cfg.collection_name, using="default")
                self._validate_collection_schema(collection)
                collection.load()
                self.collection = collection
                self._wait_for_collection_loaded(self.cfg.collection_name)
                logger.info("Loaded Milvus collection: %s", self.cfg.collection_name)
            else:
                collection = self._create_collection()
                self.collection = collection
                logger.info("Created Milvus collection: %s", self.cfg.collection_name)

            return collection

    def health(self) -> bool:
        return bool(
            self.with_retry(
                lambda: utility.has_collection(self.cfg.collection_name, using="default"),
                op_name="health_check",
                retries=1,
            )
        )

    def num_entities(self) -> int:
        return int(self._require_collection().num_entities)

    def query_existing_hashes(self, hashes: list[str]) -> dict[str, dict[str, str]]:
        collection = self._require_collection()
        unique_hashes = sorted(set(h for h in hashes if h))
        if not unique_hashes:
            return {}

        found: dict[str, dict[str, str]] = {}
        chunk_size = 100
        for i in range(0, len(unique_hashes), chunk_size):
            chunk = unique_hashes[i : i + chunk_size]
            quoted = ",".join(f'"{h}"' for h in chunk)
            expr = f'image_sha256 in [{quoted}]'
            rows = collection.query(
                expr=expr,
                output_fields=["image_id", "image_sha256"],
            )
            for row in rows:
                found[row["image_sha256"]] = {
                    "image_id": row["image_id"],
                    "image_sha256": row["image_sha256"],
                }
        return found

    def query_cached_results(self, hashes: list[str]) -> dict[str, dict[str, Any]]:
        collection = self._require_collection()
        unique_hashes = sorted(set(h for h in hashes if h))
        if not unique_hashes:
            return {}

        found: dict[str, dict[str, Any]] = {}
        chunk_size = 100
        for i in range(0, len(unique_hashes), chunk_size):
            chunk = unique_hashes[i : i + chunk_size]
            quoted = ",".join(f'"{h}"' for h in chunk)
            expr = f'image_sha256 in [{quoted}]'
            rows = collection.query(
                expr=expr,
                output_fields=[
                    "image_sha256",
                    "image_id",
                    "pipeline_stage",
                    "classification_json",
                    "vlm_json",
                    "openai_skipped",
                    "skip_reason",
                ],
            )
            for row in rows:
                image_sha256 = row.get("image_sha256")
                if not image_sha256:
                    continue

                classification = self._deserialize_json_field(
                    row.get("classification_json")
                )
                if classification is None:
                    continue

                found[image_sha256] = {
                    "image_id": row.get("image_id"),
                    "image_sha256": image_sha256,
                    "pipeline_stage": row.get("pipeline_stage"),
                    "classification": classification,
                    "vlm": self._deserialize_json_field(row.get("vlm_json")),
                    "openai_skipped": bool(row.get("openai_skipped")),
                    "skip_reason": row.get("skip_reason") or None,
                }
        return found

    def search_nearest_batch(self, vectors: np.ndarray) -> list[dict[str, Any] | None]:
        if len(vectors) == 0:
            return []

        collection = self._require_collection()
        results = collection.search(
            data=vectors.astype(np.float32).tolist(),
            anns_field="vector",
            param={
                "metric_type": "IP",
                "params": {"ef": self.cfg.search_ef},
            },
            limit=1,
            output_fields=["image_id", "image_sha256"],
        )

        nearest_items = []
        for result in results:
            if not result:
                nearest_items.append(None)
                continue

            hit = result[0]
            nearest_items.append(
                {
                    "milvus_id": int(hit.id),
                    "image_id": hit.entity.get("image_id"),
                    "image_sha256": hit.entity.get("image_sha256"),
                    "cosine_similarity": float(hit.score),
                }
            )
        return nearest_items

    def insert_processed_images(
        self,
        rows: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> None:
        collection = self._require_collection()
        existing_by_hash = self.query_existing_hashes([row["sha256"] for row in rows])

        rows_to_insert = []
        for row in rows:
            flat_idx = row["flat_idx"]
            image_id = row["image_id"]
            sha256 = row["sha256"]
            existing = existing_by_hash.get(sha256)

            if existing:
                result = results[flat_idx]
                if existing["image_id"] == image_id:
                    result["inserted"] = True
                    result["already_existed_after_retry"] = True
                else:
                    result.clear()
                    result.update(
                        {
                            "image_id": image_id,
                            "status": "duplicate",
                            "duplicate_type": "exact_sha256_race_or_retry",
                            "matched_image_id": existing["image_id"],
                            "matched_sha256": sha256,
                            "cosine_similarity": 1.0,
                            "distance": 0.0,
                        }
                    )
                continue

            classification_json = self._serialize_json_field(row.get("classification"))
            if classification_json is None:
                logger.warning(
                    "Skipping insert because classification is missing or too large: image_id=%s sha256=%s",
                    image_id,
                    sha256,
                )
                continue

            vlm_json = self._serialize_json_field(row.get("vlm"))
            if row.get("vlm") is not None and vlm_json is None:
                logger.warning(
                    "Skipping insert because VLM payload is too large: image_id=%s sha256=%s",
                    image_id,
                    sha256,
                )
                continue

            skip_reason = row.get("skip_reason")
            if skip_reason is not None:
                skip_reason = str(skip_reason)[:CACHE_SKIP_REASON_MAX_LENGTH]

            rows_to_insert.append(
                {
                    "flat_idx": flat_idx,
                    "image_id": image_id,
                    "sha256": sha256,
                    "vector": row["vector"],
                    "pipeline_stage": str(row.get("pipeline_stage") or ""),
                    "classification_json": classification_json,
                    "vlm_json": vlm_json or "",
                    "openai_skipped": bool(row.get("openai_skipped")),
                    "skip_reason": skip_reason or "",
                }
            )

        if not rows_to_insert:
            return

        mutation_result = collection.insert(
            [
                [row["image_id"] for row in rows_to_insert],
                [row["sha256"] for row in rows_to_insert],
                np.stack([row["vector"] for row in rows_to_insert]).astype(np.float32).tolist(),
                [row["pipeline_stage"] for row in rows_to_insert],
                [row["classification_json"] for row in rows_to_insert],
                [row["vlm_json"] for row in rows_to_insert],
                [row["openai_skipped"] for row in rows_to_insert],
                [row["skip_reason"] for row in rows_to_insert],
            ]
        )
        if self.cfg.flush_on_insert:
            collection.flush()

        primary_keys = getattr(mutation_result, "primary_keys", None) or []
        for idx, row in enumerate(rows_to_insert):
            result = results[row["flat_idx"]]
            result["inserted"] = True
            if idx < len(primary_keys):
                try:
                    result["milvus_id"] = int(primary_keys[idx])
                except Exception:
                    result["milvus_primary_key"] = str(primary_keys[idx])

    def with_retry(
        self,
        fn: Callable[[], Any],
        op_name: str,
        retries: int | None = None,
    ) -> Any:
        retries = self.cfg.milvus_retries if retries is None else retries
        delay = self.cfg.milvus_retry_sleep

        for attempt in range(retries + 1):
            try:
                return fn()
            except Exception as exc:
                if not self._looks_like_milvus_failure(exc) or attempt >= retries:
                    raise

                self.last_error = repr(exc)
                self.recovered_this_predict += 1
                logger.exception(
                    "Milvus op failed: %s. Reconnecting. attempt=%s/%s",
                    op_name,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                self.connect_and_prepare_collection()

        raise RuntimeError(f"Milvus operation failed after retries: {op_name}")

    def _create_collection(self) -> Collection:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="image_sha256", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.cfg.dedup_embedding_dim or 384,
            ),
            FieldSchema(name="pipeline_stage", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(
                name="classification_json",
                dtype=DataType.VARCHAR,
                max_length=CACHE_JSON_MAX_LENGTH,
            ),
            FieldSchema(
                name="vlm_json",
                dtype=DataType.VARCHAR,
                max_length=CACHE_JSON_MAX_LENGTH,
            ),
            FieldSchema(name="openai_skipped", dtype=DataType.BOOL),
            FieldSchema(
                name="skip_reason",
                dtype=DataType.VARCHAR,
                max_length=CACHE_SKIP_REASON_MAX_LENGTH,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Triton embedding image dedup collection with cached pipeline results",
        )

        collection = Collection(
            name=self.cfg.collection_name,
            schema=schema,
            using="default",
        )
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        collection.load()
        self._wait_for_collection_loaded(self.cfg.collection_name)
        return collection

    def _validate_collection_schema(self, collection: Collection) -> None:
        schema = getattr(collection, "schema", None)
        fields = getattr(schema, "fields", None) or []
        field_names = {getattr(field, "name", None) for field in fields}
        required_fields = {
            "id",
            "image_id",
            "image_sha256",
            "vector",
            "pipeline_stage",
            "classification_json",
            "vlm_json",
            "openai_skipped",
            "skip_reason",
        }
        missing_fields = sorted(name for name in required_fields if name not in field_names)
        if missing_fields:
            raise RuntimeError(
                "Milvus collection schema is outdated. "
                f"Collection '{self.cfg.collection_name}' is missing fields: {', '.join(missing_fields)}. "
                "Delete the collection and restart the API so it can be recreated with the new schema."
            )

    def _wait_for_collection_loaded(self, collection_name: str) -> None:
        try:
            utility.wait_for_loading_complete(
                collection_name=collection_name,
                timeout=self.cfg.milvus_load_timeout,
                using="default",
            )
        except AttributeError:
            pass

    def _require_collection(self) -> Collection:
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized.")
        return self.collection

    @staticmethod
    def _serialize_json_field(value: Any) -> str | None:
        if value is None:
            return None

        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > CACHE_JSON_MAX_LENGTH:
            return None
        return serialized

    @staticmethod
    def _deserialize_json_field(value: Any) -> Any:
        if value in (None, ""):
            return None

        if isinstance(value, (dict, list)):
            return value

        try:
            return json.loads(value)
        except Exception:
            logger.warning("Failed to decode cached JSON payload from Milvus.")
            return None

    @staticmethod
    def _looks_like_milvus_failure(exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                MilvusException,
                ConnectionNotExistException,
                ConnectionError,
                TimeoutError,
                OSError,
            ),
        ):
            return True

        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in [
                "milvus",
                "connection",
                "connect",
                "grpc",
                "unavailable",
                "deadline",
                "timeout",
                "reset by peer",
                "refused",
            ]
        )
