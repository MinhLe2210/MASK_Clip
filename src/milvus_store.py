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


logger = logging.getLogger("pipeline-litserve")


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
                collection.load()
                self.collection = collection
                self._wait_for_collection_loaded()
                logger.info("Loaded Milvus collection: %s", self.cfg.collection_name)
                return collection

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

    def insert_accepted(
        self,
        accepted: list[dict[str, Any]],
        results: list[dict[str, Any] | None],
    ) -> None:
        collection = self._require_collection()
        existing_by_hash = self.query_existing_hashes([x["sha256"] for x in accepted])

        rows_to_insert = []
        for row in accepted:
            existing = existing_by_hash.get(row["sha256"])
            flat_idx = row["flat_idx"]

            if existing:
                if existing["image_id"] == row["image_id"]:
                    results[flat_idx] = {
                        "image_id": row["image_id"],
                        "status": "unique",
                        "sha256": row["sha256"],
                        "inserted": True,
                        "already_existed_after_retry": True,
                    }
                else:
                    results[flat_idx] = {
                        "image_id": row["image_id"],
                        "status": "duplicate",
                        "duplicate_type": "exact_sha256_race_or_retry",
                        "matched_image_id": existing["image_id"],
                        "matched_sha256": row["sha256"],
                        "cosine_similarity": 1.0,
                        "distance": 0.0,
                    }
                continue

            rows_to_insert.append(row)

        if not rows_to_insert:
            return

        image_ids = [x["image_id"] for x in rows_to_insert]
        image_hashes = [x["sha256"] for x in rows_to_insert]
        vectors = np.stack([x["vector"] for x in rows_to_insert]).astype(np.float32)

        mutation_result = collection.insert([image_ids, image_hashes, vectors.tolist()])
        if self.cfg.flush_on_insert:
            collection.flush()

        primary_keys = getattr(mutation_result, "primary_keys", None) or []
        for idx, row in enumerate(rows_to_insert):
            flat_idx = row["flat_idx"]
            result = results[flat_idx] or {}
            result["inserted"] = True
            if idx < len(primary_keys):
                result["milvus_id"] = int(primary_keys[idx])
            results[flat_idx] = result

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
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Triton embedding image dedup collection",
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
        self._wait_for_collection_loaded()
        return collection

    def _wait_for_collection_loaded(self) -> None:
        try:
            utility.wait_for_loading_complete(
                collection_name=self.cfg.collection_name,
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
