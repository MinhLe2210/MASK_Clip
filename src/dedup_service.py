import hashlib
from typing import Any

import numpy as np

from src.image_io import bytes_from_ref, bytes_to_pil
from src.milvus_store import MilvusDedupStore
from src.triton_clients import TritonEmbeddingClient


class DedupService:
    def __init__(
        self,
        embedder: TritonEmbeddingClient,
        store: MilvusDedupStore,
        dup_threshold: float,
    ):
        self.embedder = embedder
        self.store = store
        self.dup_threshold = dup_threshold

    def prepare(self) -> None:
        self.store.connect_and_prepare_collection()

    def health(self) -> bool:
        return self.embedder.is_ready() and self.store.health()

    def deduplicate(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any] | None] = [None] * len(items)

        decoded = []
        for idx, item in enumerate(items):
            try:
                image_bytes = bytes_from_ref(item["image_ref"])
                sha256 = hashlib.sha256(image_bytes).hexdigest()
                image = bytes_to_pil(image_bytes)
                decoded.append(
                    {
                        "flat_idx": idx,
                        "image_id": item["image_id"],
                        "image_ref": item["image_ref"],
                        "sha256": sha256,
                        "image": image,
                    }
                )
            except Exception as exc:
                results[idx] = {
                    "image_id": item["image_id"],
                    "status": "error",
                    "error": f"invalid_image: {exc}",
                }

        unique_records: list[dict[str, Any]] = []
        if decoded:
            unique_records = self._process_decoded_images(decoded, results)

        return [result or {"status": "error", "error": "missing_result"} for result in results], unique_records

    def _process_decoded_images(
        self,
        decoded: list[dict[str, Any]],
        results: list[dict[str, Any] | None],
    ) -> list[dict[str, Any]]:
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
                results[flat_idx] = {
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
                results[flat_idx] = {
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
            return []

        vectors = self.embedder.embed([x["image"] for x in candidates])
        existing_count = self.store.with_retry(
            self.store.num_entities,
            op_name="num_entities",
        )
        nearest_batch = (
            self.store.with_retry(
                lambda: self.store.search_nearest_batch(vectors),
                op_name="search_nearest_batch",
            )
            if existing_count > 0
            else [None] * len(vectors)
        )

        accepted = []
        accepted_vectors: list[np.ndarray] = []
        accepted_meta: list[dict[str, Any]] = []

        for item, vector, nearest in zip(candidates, vectors, nearest_batch):
            flat_idx = item["flat_idx"]
            image_id = item["image_id"]
            sha256 = item["sha256"]

            if nearest is not None and nearest["cosine_similarity"] >= self.dup_threshold:
                results[flat_idx] = {
                    "image_id": image_id,
                    "status": "duplicate",
                    "duplicate_type": "embedding_existing_milvus",
                    "matched_image_id": nearest["image_id"],
                    "matched_sha256": nearest.get("image_sha256"),
                    "cosine_similarity": nearest["cosine_similarity"],
                    "distance": 1.0 - nearest["cosine_similarity"],
                }
                continue

            mem_dup, mem_idx, mem_score = self._is_duplicate_in_memory(
                vector=vector,
                accepted_vectors=accepted_vectors,
                threshold=self.dup_threshold,
            )
            if mem_dup:
                matched = accepted_meta[mem_idx]
                results[flat_idx] = {
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
                    "image_ref": item["image_ref"],
                    "image": item["image"],
                    "sha256": sha256,
                    "vector": vector,
                }
            )
            accepted_vectors.append(vector)
            accepted_meta.append({"image_id": image_id, "sha256": sha256})
            results[flat_idx] = {
                "image_id": image_id,
                "status": "unique",
                "sha256": sha256,
                "inserted": False,
            }

        if accepted:
            self.store.with_retry(
                lambda: self.store.insert_accepted(accepted, results),
                op_name="insert_accepted",
            )

        return [
            record
            for record in accepted
            if results[record["flat_idx"]] is not None
            and results[record["flat_idx"]].get("status") == "unique"
        ]

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
