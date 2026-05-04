# server.py
import base64
import binascii
import hashlib
import io
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import litserve as ls
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image, ImageFile
from starlette.middleware.base import BaseHTTPMiddleware
from transformers import AutoImageProcessor, AutoModel

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

try:
    from pymilvus.exceptions import MilvusException, ConnectionNotExistException
except Exception:
    from pymilvus.exceptions import MilvusException

    ConnectionNotExistException = MilvusException

try:
    from litserve.callbacks import Callback
except Exception:
    from litserve.callbacks.base import Callback


ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("image-dedup-litserve")


@dataclass
class Settings:
    model_path: str
    dim: int
    dup_threshold: float

    collection_name: str
    milvus_host: str
    milvus_port: str

    search_ef: int
    milvus_retries: int
    milvus_retry_sleep: float
    milvus_load_timeout: float

    max_images_per_request: int
    flush_on_insert: bool

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = os.getenv("MODEL_PATH")
        milvus_host = os.getenv("MILVUS_HOST")
        milvus_port = os.getenv("MILVUS_PORT", "19530")

        if not model_path:
            raise ValueError("MODEL_PATH is missing.")
        if not milvus_host:
            raise ValueError("MILVUS_HOST is missing.")

        return cls(
            model_path=model_path,
            dim=int(os.getenv("EMBED_DIM", "768")),
            dup_threshold=float(os.getenv("DUP_THRESHOLD", "0.999995")),
            collection_name=os.getenv(
                "COLLECTION_NAME",
                "AI_detector_image_dedup_b64",
            ),
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            search_ef=int(os.getenv("SEARCH_EF", "64")),
            milvus_retries=int(os.getenv("MILVUS_RETRIES", "5")),
            milvus_retry_sleep=float(os.getenv("MILVUS_RETRY_SLEEP", "1.0")),
            milvus_load_timeout=float(os.getenv("MILVUS_LOAD_TIMEOUT", "120")),
            max_images_per_request=int(os.getenv("MAX_IMAGES_PER_REQUEST", "64")),
            flush_on_insert=os.getenv("MILVUS_FLUSH_ON_INSERT", "true").lower()
            in {"1", "true", "yes", "y"},
        )


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class DedupCallback(Callback):
    def on_server_start(self, *args, **kwargs):
        logger.info("LitServe server starting.")

    def on_after_setup(self, *args, **kwargs):
        logger.info("LitServe worker setup completed.")

    def on_before_predict(self, lit_api=None, *args, **kwargs):
        if lit_api is not None:
            lit_api.milvus_recovered_this_predict = 0
            lit_api.last_milvus_error = None

    def on_after_predict(self, lit_api=None, *args, **kwargs):
        if lit_api is None:
            return

        recovered = getattr(lit_api, "milvus_recovered_this_predict", 0)
        if recovered:
            logger.warning("Milvus recovered %s time(s) in this predict().", recovered)

    def on_response(self, *args, **kwargs):
        logger.info("Response generated.")


class Base64ImageDedupAPI(ls.LitAPI):
    def __init__(self):
        max_batch_size = int(os.getenv("LITSERVE_MAX_BATCH_SIZE", "8"))
        batch_timeout = float(os.getenv("LITSERVE_BATCH_TIMEOUT", "0.05"))

        super().__init__(
            api_path="/dedup",
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
        )

    def setup(self, device):
        self.cfg = Settings.from_env()

        self.device = torch.device(
            device if isinstance(device, str) else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        logger.info("Loading model from %s on %s", self.cfg.model_path, self.device)

        self.processor = AutoImageProcessor.from_pretrained(self.cfg.model_path)
        self.model = AutoModel.from_pretrained(self.cfg.model_path).to(self.device)
        self.model.eval()

        self.collection: Collection | None = None
        self._milvus_lock = Lock()
        self.milvus_recovered_this_predict = 0
        self.last_milvus_error = None

        self._connect_and_prepare_collection()

    def decode_request(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Accepted payload:

        {
          "request_id": "optional-client-request-id",
          "images": [
            {
              "image_id": "img-001",
              "image_base64": "/9j/4AAQSkZJRgABAQ..."
            },
            {
              "image_id": "img-002",
              "image_base64": "data:image/png;base64,iVBORw0..."
            }
          ]
        }

        For convenience, this also accepts a single image:

        {
          "image_id": "img-001",
          "image_base64": "..."
        }
        """
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

        if len(images) > self.cfg.max_images_per_request:
            raise ValueError(
                f"Too many images. Max is {self.cfg.max_images_per_request}."
            )

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

    def batch(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        """
        LitServe batches multiple HTTP requests.

        Since each request can itself contain many images, we flatten:
        [
          request_1: [img_a, img_b],
          request_2: [img_c]
        ]

        into:
        [img_a, img_b, img_c]

        Then unbatch() groups results back per request.
        """
        flat_items = []
        request_ids = []
        request_sizes = []

        for req in inputs:
            request_ids.append(req["request_id"])
            request_sizes.append(len(req["items"]))

            for item in req["items"]:
                flat_items.append(item)

        return {
            "request_ids": request_ids,
            "request_sizes": request_sizes,
            "flat_items": flat_items,
        }

    def predict(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        started_at = time.time()

        flat_items = batch["flat_items"]
        flat_results: list[dict[str, Any] | None] = [None] * len(flat_items)

        if not flat_items:
            return {
                "request_ids": batch["request_ids"],
                "request_sizes": batch["request_sizes"],
                "flat_results": [],
            }

        decoded = []
        for idx, item in enumerate(flat_items):
            try:
                image_bytes = self._decode_base64_to_bytes(item["image_base64"])
                sha256 = hashlib.sha256(image_bytes).hexdigest()
                image = self._bytes_to_pil(image_bytes)

                decoded.append(
                    {
                        "flat_idx": idx,
                        "image_id": item["image_id"],
                        "sha256": sha256,
                        "image": image,
                    }
                )
            except Exception as exc:
                flat_results[idx] = {
                    "image_id": item.get("image_id"),
                    "status": "error",
                    "error": f"invalid_image_base64: {exc}",
                }

        if decoded:
            self._process_decoded_images(decoded, flat_results)

        elapsed = round(time.time() - started_at, 4)

        for result in flat_results:
            if result is not None:
                result["elapsed_batch_seconds"] = elapsed

        return {
            "request_ids": batch["request_ids"],
            "request_sizes": batch["request_sizes"],
            "flat_results": flat_results,
        }

    def unbatch(self, output: dict[str, Any]) -> list[dict[str, Any]]:
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
                    "milvus_recovered_this_predict": self.milvus_recovered_this_predict,
                    "last_milvus_error": self.last_milvus_error,
                }
            )

        return responses

    def encode_response(self, output: dict[str, Any], **kwargs) -> dict[str, Any]:
        return output

    def health(self) -> bool:
        try:
            return bool(
                self._with_milvus_retry(
                    lambda: utility.has_collection(self.cfg.collection_name),
                    op_name="health_check",
                    retries=1,
                )
            )
        except Exception:
            logger.exception("Health check failed.")
            return False

    def _process_decoded_images(
        self,
        decoded: list[dict[str, Any]],
        flat_results: list[dict[str, Any] | None],
    ) -> None:
        existing_by_hash = self._with_milvus_retry(
            lambda: self._query_existing_hashes([x["sha256"] for x in decoded]),
            op_name="query_existing_hashes",
        )

        candidates = []
        seen_in_current_batch: dict[str, dict[str, Any]] = {}

        for item in decoded:
            flat_idx = item["flat_idx"]
            image_id = item["image_id"]
            sha256 = item["sha256"]

            existing = existing_by_hash.get(sha256)
            if existing:
                flat_results[flat_idx] = {
                    "image_id": image_id,
                    "status": "duplicate",
                    "duplicate_type": "exact_sha256_existing_milvus",
                    "matched_image_id": existing["image_id"],
                    "matched_sha256": sha256,
                    "cosine_similarity": 1.0,
                    "distance": 0.0,
                }
                continue

            if sha256 in seen_in_current_batch:
                matched = seen_in_current_batch[sha256]
                flat_results[flat_idx] = {
                    "image_id": image_id,
                    "status": "duplicate",
                    "duplicate_type": "exact_sha256_current_batch",
                    "matched_image_id": matched["image_id"],
                    "matched_sha256": sha256,
                    "cosine_similarity": 1.0,
                    "distance": 0.0,
                }
                continue

            seen_in_current_batch[sha256] = item
            candidates.append(item)

        if not candidates:
            return

        images = [x["image"] for x in candidates]
        vectors = self._extract_features_images(images)

        existing_count = self._with_milvus_retry(
            lambda: int(self.collection.num_entities),
            op_name="num_entities",
        )

        if existing_count > 0:
            nearest_batch = self._with_milvus_retry(
                lambda: self._search_nearest_batch(vectors),
                op_name="search_nearest_batch",
            )
        else:
            nearest_batch = [None] * len(vectors)

        accepted = []
        pending_vectors: list[np.ndarray] = []
        pending_meta: list[dict[str, Any]] = []

        for item, vector, nearest in zip(candidates, vectors, nearest_batch):
            flat_idx = item["flat_idx"]
            image_id = item["image_id"]
            sha256 = item["sha256"]

            if nearest is not None:
                cosine_similarity = nearest["cosine_similarity"]

                if cosine_similarity >= self.cfg.dup_threshold:
                    flat_results[flat_idx] = {
                        "image_id": image_id,
                        "status": "duplicate",
                        "duplicate_type": "embedding_existing_milvus",
                        "matched_image_id": nearest["image_id"],
                        "matched_sha256": nearest.get("image_sha256"),
                        "cosine_similarity": cosine_similarity,
                        "distance": 1.0 - cosine_similarity,
                    }
                    continue

            mem_dup, mem_idx, mem_score = self._is_duplicate_in_memory(
                vector=vector,
                accepted_vectors=pending_vectors,
                threshold=self.cfg.dup_threshold,
            )

            if mem_dup:
                matched = pending_meta[mem_idx]
                flat_results[flat_idx] = {
                    "image_id": image_id,
                    "status": "duplicate",
                    "duplicate_type": "embedding_current_batch",
                    "matched_image_id": matched["image_id"],
                    "matched_sha256": matched["sha256"],
                    "cosine_similarity": mem_score,
                    "distance": 1.0 - mem_score,
                }
                continue

            accepted.append(
                {
                    "flat_idx": flat_idx,
                    "image_id": image_id,
                    "sha256": sha256,
                    "vector": vector,
                }
            )
            pending_vectors.append(vector)
            pending_meta.append(
                {
                    "image_id": image_id,
                    "sha256": sha256,
                }
            )

            flat_results[flat_idx] = {
                "image_id": image_id,
                "status": "unique",
                "sha256": sha256,
                "inserted": False,
            }

        if accepted:
            self._with_milvus_retry(
                lambda: self._insert_accepted(accepted, flat_results),
                op_name="insert_accepted",
            )

    def _connect_and_prepare_collection(self) -> Collection:
        with self._milvus_lock:
            try:
                connections.disconnect(alias="default")
            except Exception:
                pass

            connections.connect(
                alias="default",
                host=self.cfg.milvus_host,
                port=self.cfg.milvus_port,
            )

            if utility.has_collection(self.cfg.collection_name):
                collection = Collection(self.cfg.collection_name)
                collection.load()
                self.collection = collection
                self._wait_for_collection_loaded()
                logger.info("Loaded Milvus collection: %s", self.cfg.collection_name)
                return collection

            collection = self._create_collection()
            self.collection = collection
            logger.info("Created Milvus collection: %s", self.cfg.collection_name)
            return collection

    def _create_collection(self) -> Collection:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="image_id",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="image_sha256",
                dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.cfg.dim,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Base64 image dedup collection",
        )

        collection = Collection(
            name=self.cfg.collection_name,
            schema=schema,
        )

        index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
            },
        }

        collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        collection.load()
        self._wait_for_collection_loaded()
        return collection

    def _wait_for_collection_loaded(self):
        try:
            utility.wait_for_loading_complete(
                collection_name=self.cfg.collection_name,
                timeout=self.cfg.milvus_load_timeout,
            )
        except AttributeError:
            pass

    def _with_milvus_retry(self, fn, op_name: str, retries: int | None = None):
        retries = self.cfg.milvus_retries if retries is None else retries
        delay = self.cfg.milvus_retry_sleep

        for attempt in range(retries + 1):
            try:
                return fn()
            except Exception as exc:
                if not self._looks_like_milvus_failure(exc) or attempt >= retries:
                    raise

                self.last_milvus_error = repr(exc)
                self.milvus_recovered_this_predict += 1

                logger.exception(
                    "Milvus op failed: %s. Reconnecting. attempt=%s/%s",
                    op_name,
                    attempt + 1,
                    retries,
                )

                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                self._connect_and_prepare_collection()

        raise RuntimeError(f"Milvus operation failed after retries: {op_name}")

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

    @staticmethod
    def _decode_base64_to_bytes(image_base64: str) -> bytes:
        if "," in image_base64 and image_base64.lower().startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        try:
            return base64.b64decode(image_base64, validate=True)
        except binascii.Error:
            # Some clients send base64 with newlines/spaces.
            compact = "".join(image_base64.split())
            return base64.b64decode(compact, validate=True)

    @staticmethod
    def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.convert("RGB").copy()

    @torch.inference_mode()
    def _extract_features_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        features = outputs.last_hidden_state.mean(dim=1)
        vectors = features.detach().cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors.astype(np.float32)

    def _search_nearest_batch(
        self,
        vectors: np.ndarray,
    ) -> list[dict[str, Any] | None]:
        if len(vectors) == 0:
            return []

        results = self.collection.search(
            data=vectors.astype(np.float32).tolist(),
            anns_field="vector",
            param={
                "metric_type": "IP",
                "params": {
                    "ef": self.cfg.search_ef,
                },
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

    def _query_existing_hashes(
        self,
        hashes: list[str],
    ) -> dict[str, dict[str, str]]:
        hashes = sorted(set(h for h in hashes if h))
        if not hashes:
            return {}

        found: dict[str, dict[str, str]] = {}

        chunk_size = 100
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i : i + chunk_size]
            quoted = ",".join(f'"{h}"' for h in chunk)
            expr = f"image_sha256 in [{quoted}]"

            rows = self.collection.query(
                expr=expr,
                output_fields=["image_id", "image_sha256"],
            )

            for row in rows:
                found[row["image_sha256"]] = {
                    "image_id": row["image_id"],
                    "image_sha256": row["image_sha256"],
                }

        return found

    def _insert_accepted(
        self,
        accepted: list[dict[str, Any]],
        flat_results: list[dict[str, Any] | None],
    ) -> None:
        # Idempotency guard. If a previous insert partially succeeded before
        # a Milvus reconnect/retry, don't insert the same SHA again.
        existing_by_hash = self._query_existing_hashes([x["sha256"] for x in accepted])

        rows_to_insert = []
        for row in accepted:
            existing = existing_by_hash.get(row["sha256"])
            flat_idx = row["flat_idx"]

            if existing:
                if existing["image_id"] == row["image_id"]:
                    flat_results[flat_idx] = {
                        "image_id": row["image_id"],
                        "status": "unique",
                        "sha256": row["sha256"],
                        "inserted": True,
                        "already_existed_after_retry": True,
                    }
                else:
                    flat_results[flat_idx] = {
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

        mutation_result = self.collection.insert(
            [
                image_ids,
                image_hashes,
                vectors.tolist(),
            ]
        )

        if self.cfg.flush_on_insert:
            self.collection.flush()

        primary_keys = getattr(mutation_result, "primary_keys", None) or []

        for idx, row in enumerate(rows_to_insert):
            flat_idx = row["flat_idx"]
            result = flat_results[flat_idx] or {}
            result["inserted"] = True
            if idx < len(primary_keys):
                result["milvus_id"] = int(primary_keys[idx])
            flat_results[flat_idx] = result

    @staticmethod
    def _is_duplicate_in_memory(
        vector: np.ndarray,
        accepted_vectors: list[np.ndarray],
        threshold: float,
    ) -> tuple[bool, int | None, float | None]:
        if not accepted_vectors:
            return False, None, None

        matrix = np.stack(accepted_vectors)
        scores = matrix @ vector

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return True, best_idx, best_score

        return False, None, best_score


if __name__ == "__main__":
    api = Base64ImageDedupAPI()

    server = ls.LitServer(
        api,
        accelerator=os.getenv("LITSERVE_ACCELERATOR", "auto"),
        devices=os.getenv("LITSERVE_DEVICES", "auto"),
        workers_per_device=int(os.getenv("LITSERVE_WORKERS_PER_DEVICE", "1")),
        timeout=int(os.getenv("LITSERVE_REQUEST_TIMEOUT", "120")),
        track_requests=True,
        restart_workers=True,
        max_payload_size=os.getenv("LITSERVE_MAX_PAYLOAD_SIZE", "100MB"),
        callbacks=[DedupCallback()],
        middlewares=[(RequestIdMiddleware, {})],
    )

    server.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )