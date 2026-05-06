import os
from dataclasses import dataclass


@dataclass
class Settings:
    model_path: str
    dim: int
    dup_threshold: float
    dedup_engine_path: str

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
        model_path = os.getenv("MODEL_PATH") or os.getenv("DEDUP_ENGINE_PATH")
        dedup_engine_path = os.getenv("DEDUP_ENGINE_PATH") or model_path
        milvus_host = os.getenv("MILVUS_HOST")
        milvus_port = os.getenv("MILVUS_PORT", "19530")

        if not model_path:
            raise ValueError("MODEL_PATH or DEDUP_ENGINE_PATH is missing.")
        if not milvus_host:
            raise ValueError("MILVUS_HOST is missing.")

        return cls(
            model_path=model_path,
            dim=int(os.getenv("EMBED_DIM", "384")),
            dup_threshold=float(os.getenv("DUP_THRESHOLD", "0.999995")),
            collection_name=os.getenv(
                "COLLECTION_NAME",
                "AI_detector_image_dedup_b64",
            ),
            dedup_engine_path=dedup_engine_path,
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
