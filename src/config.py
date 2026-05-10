import os
from dataclasses import dataclass
import json
from pathlib import Path


def load_openai_prompt() -> str:
    prompt_path = Path(__file__).with_name("prompt.json")

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("openai_prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "src/prompt.json must contain non-empty 'openai_prompt'."
        )

    return prompt.strip()

def _required_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or not value.strip():
        raise ValueError(f"{name} is missing.")
    return value.strip()


def _optional_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _optional_int_env(*names: str) -> int | None:
    value = _optional_env(*names)
    return int(value) if value is not None else None


def _optional_float_env(*names: str) -> float | None:
    value = _optional_env(*names)
    return float(value) if value is not None else None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_triton_http_url(raw: str) -> tuple[str, str]:
    value = raw.strip().rstrip("/")
    if value.startswith(("http://", "https://")):
        scheme, rest = value.split("://", 1)
        return f"{scheme}://{rest}", rest
    return f"http://{value}", value


@dataclass
class Settings:
    triton_http_url: str
    triton_client_url: str

    dedup_model_name: str
    dedup_input_name: str
    dedup_output_name: str
    dedup_image_size: int | None
    dedup_embedding_dim: int | None

    classification_model_name: str
    classification_input_name: str
    classification_output_name: str
    classification_labels: list[str]
    classification_image_size: int | None

    openai_model: str
    openai_proxy: str | None
    openai_timeout: float
    request_timeout: float
    vlm_prompt: str

    collection_name: str
    milvus_host: str
    milvus_port: str
    milvus_database: str | None

    dup_threshold: float
    search_ef: int
    milvus_retries: int
    milvus_retry_sleep: float
    milvus_load_timeout: float
    flush_on_insert: bool

    max_images_per_request: int

    @classmethod
    def from_env(cls) -> "Settings":
        triton_raw = _required_env("TRITON_URL", "http://127.0.0.1:8000")
        triton_http_url, triton_client_url = _normalize_triton_http_url(triton_raw)
        collection_name = _required_env(
            "COLLECTION_NAME",
            "ai_detector_images_deduplicate",
        )

        classification_labels = [
            x.strip()
            for x in _required_env("CLASSIFIER_LABELS", "fake,real").split(",")
            if x.strip()
        ]
        if len(classification_labels) < 2:
            raise ValueError("CLASSIFIER_LABELS must contain at least 2 labels.")

        return cls(
            triton_http_url=triton_http_url,
            triton_client_url=triton_client_url,
            dedup_model_name=_required_env("DEDUP_MODEL_NAME", "dedup_embedder"),
            dedup_input_name=_required_env("DEDUP_INPUT_NAME", "pixel_values"),
            dedup_output_name=_required_env("DEDUP_OUTPUT_NAME", "features"),
            dedup_image_size=_optional_int_env("DEDUP_IMAGE_SIZE"),
            dedup_embedding_dim=_optional_int_env("EMBED_DIM"),
            classification_model_name=_required_env(
                "CLASSIFIER_MODEL_NAME",
                _optional_env("AI_MODEL_NAME") or "AI_images_detector",
            ),
            classification_input_name=_required_env(
                "CLASSIFIER_INPUT_NAME",
                _optional_env("AI_INPUT_NAME") or "pixel_values",
            ),
            classification_output_name=_required_env(
                "CLASSIFIER_OUTPUT_NAME",
                _optional_env("AI_OUTPUT_NAME") or "logits",
            ),
            classification_labels=classification_labels,
            classification_image_size=_optional_int_env(
                "CLASSIFIER_INPUT_SIZE",
                "AI_IMAGE_SIZE",
                "AI_DEFAULT_IMAGE_SIZE",
            ),
            openai_model=_required_env("OPENAI_VLM_MODEL", "gpt-5-mini"),
            openai_proxy=_optional_env("OPENAI_PROXY"),
            openai_timeout=float(_required_env("OPENAI_TIMEOUT", "60")),
            request_timeout=float(
                _required_env(
                    "PIPELINE_REQUEST_TIMEOUT",
                    _optional_env("REQUEST_TIMEOUT") or "120",
                )
            ),
            vlm_prompt=load_openai_prompt(),
            collection_name=collection_name,
            milvus_host=_required_env("MILVUS_HOST"),
            milvus_port=_required_env("MILVUS_PORT", "19530"),
            milvus_database=_optional_env("MILVUS_DATABASE", "DATABASE"),
            dup_threshold=float(_required_env("DUP_THRESHOLD", "0.999995")),
            search_ef=int(_required_env("SEARCH_EF", "64")),
            milvus_retries=int(_required_env("MILVUS_RETRIES", "5")),
            milvus_retry_sleep=float(_required_env("MILVUS_RETRY_SLEEP", "1.0")),
            milvus_load_timeout=float(_required_env("MILVUS_LOAD_TIMEOUT", "120")),
            flush_on_insert=_bool_env("MILVUS_FLUSH_ON_INSERT", True),
            max_images_per_request=int(
                _required_env(
                    "MAX_IMAGES_PER_REQUEST",
                    _optional_env("CLIENT_BATCH_SIZE") or "64",
                )
            ),
        )
