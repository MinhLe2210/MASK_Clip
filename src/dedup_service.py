import hashlib
import time
from typing import Any

import numpy as np

from src.embedder import ImageEmbedder
from src.image_io import bytes_from_ref, bytes_to_pil
from src.milvus_store import MilvusDedupStore


class ImageDedupService:
    def __init__(self, embedder: ImageEmbedder, store: MilvusDedupStore):
        self.embedder = embedder
        self.store = store

    def prepare(self) -> None:
        self.store.connect_and_prepare_collection()

    def reset_predict_state(self) -> None:
        self.store.reset_predict_state()

    @property
    def milvus_recovered_this_predict(self) -> int:
        return self.store.recovered_this_predict

    @property
    def last_milvus_error(self) -> str | None:
        return self.store.last_error

    def health(self) -> bool:
        return self.store.health()

    def predict_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
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
                image_bytes = bytes_from_ref(item["image_ref"])
                sha256 = hashlib.sha256(image_bytes).hexdigest()
                image = bytes_to_pil(image_bytes)

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
                    "error": f"invalid_image: {exc}",
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

    def _process_decoded_images(
        self,
        decoded: list[dict[str, Any]],
        flat_results: list[dict[str, Any] | None],
    ) -> None:
        existing_by_hash = self.store.with_retry(
            lambda: self.store.query_existing_hashes([x["sha256"] for x in decoded]),
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
        vectors = self.embedder.extract(images)

        existing_count = self.store.with_retry(
            self.store.num_entities,
            op_name="num_entities",
        )

        if existing_count > 0:
            nearest_batch = self.store.with_retry(
                lambda: self.store.search_nearest_batch(vectors),
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

                if cosine_similarity >= self.embedder.cfg.dup_threshold:
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
                threshold=self.embedder.cfg.dup_threshold,
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
            self.store.with_retry(
                lambda: self.store.insert_accepted(accepted, flat_results),
                op_name="insert_accepted",
            )

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
