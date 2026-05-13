import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

fastapi_module = types.ModuleType("fastapi")


class _FastAPIStub:
    def __init__(self, *args, **kwargs):
        self.state = SimpleNamespace()

    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn


class _HTTPExceptionStub(Exception):
    def __init__(self, status_code=None, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


fastapi_module.FastAPI = _FastAPIStub
fastapi_module.File = lambda *args, **kwargs: None
fastapi_module.Form = lambda *args, **kwargs: None
fastapi_module.HTTPException = _HTTPExceptionStub
fastapi_module.Request = type("Request", (), {})
fastapi_module.UploadFile = type("UploadFile", (), {})
sys.modules.setdefault("fastapi", fastapi_module)

starlette_module = types.ModuleType("starlette")
starlette_concurrency_module = types.ModuleType("starlette.concurrency")


async def _run_in_threadpool_stub(fn, *args, **kwargs):
    return fn(*args, **kwargs)


starlette_concurrency_module.run_in_threadpool = _run_in_threadpool_stub
sys.modules.setdefault("starlette", starlette_module)
sys.modules.setdefault("starlette.concurrency", starlette_concurrency_module)

milvus_store_module = types.ModuleType("src.milvus_store")
milvus_store_module.MilvusDedupStore = type("MilvusDedupStore", (), {})
sys.modules.setdefault("src.milvus_store", milvus_store_module)

triton_clients_module = types.ModuleType("src.triton_clients")
triton_clients_module.TritonClassificationClient = type(
    "TritonClassificationClient",
    (),
    {},
)
triton_clients_module.TritonEmbeddingClient = type("TritonEmbeddingClient", (), {})
sys.modules.setdefault("src.triton_clients", triton_clients_module)

nfa_vit_module = types.ModuleType("src.nfa_vit")
nfa_vit_module.TritonNfaVitClient = type("TritonNfaVitClient", (), {})
sys.modules.setdefault("src.nfa_vit", nfa_vit_module)

import pipeline_server


class FakeStore:
    def __init__(self, cached_results=None):
        self.cached_results = cached_results or {}
        self.insert_calls = []
        self.recovered_this_predict = 0
        self.last_error = None

    def with_retry(self, fn, op_name, retries=None):
        return fn()

    def query_cached_results(self, hashes):
        return {sha: self.cached_results[sha] for sha in hashes if sha in self.cached_results}

    def insert_processed_images(self, rows, results):
        self.insert_calls.append({"rows": rows, "results": results})


class FakeDedupService:
    def __init__(self, dedup_results, unique_records, store):
        self._dedup_results = dedup_results
        self._unique_records = unique_records
        self.store = store

    def deduplicate(self, items):
        return list(self._dedup_results), list(self._unique_records)


class FakeClassifier:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def classify(self, images):
        self.calls.append(images)
        return list(self.outputs)


class FakeNfaVit:
    def __init__(self, output=None):
        self.output = output or {"status": "classified", "final_label": "fake"}
        self.calls = []

    def infer(self, image):
        self.calls.append(image)
        return dict(self.output)


class FakeOpenAI:
    def __init__(self, output=None):
        self.output = output or {"model": "fake-vlm", "output_text": "{}"}
        self.calls = []

    def analyze(self, image_ref, prompt, detector_context):
        self.calls.append(
            {
                "image_ref": image_ref,
                "prompt": prompt,
                "detector_context": detector_context,
            }
        )
        return dict(self.output)


class PipelineRoutingTests(unittest.TestCase):
    def make_runtime(
        self,
        *,
        dedup_results,
        unique_records,
        classifier_outputs,
        nfa_output=None,
        cached_results=None,
    ):
        store = FakeStore(cached_results=cached_results)
        runtime = SimpleNamespace(
            cfg=SimpleNamespace(
                vlm_prompt="test prompt",
                openai_model="fake-vlm",
                nfa_model_name="nfa_vit",
            ),
            dedup_service=FakeDedupService(dedup_results, unique_records, store),
            classifier=FakeClassifier(classifier_outputs),
            nfa_vit_client=FakeNfaVit(nfa_output),
            openai_client=FakeOpenAI(),
        )
        return runtime, store

    def test_unique_real_skips_nfa_and_uses_classification_label(self):
        image = Image.new("RGB", (8, 8), "white")
        classification = {"status": "classified", "label": "real", "confidence": 0.99}
        runtime, store = self.make_runtime(
            dedup_results=[{"status": "unique", "sha256": "sha-1"}],
            unique_records=[
                {
                    "flat_idx": 0,
                    "image": image,
                    "image_ref": "memory://img-1",
                    "vector": np.array([0.1], dtype=np.float32),
                }
            ],
            classifier_outputs=[classification],
        )

        with patch("pipeline_server.image_ref_for_openai", side_effect=lambda ref: f"openai:{ref}"):
            results = pipeline_server.run_pipeline_on_items(
                runtime,
                [{"image_id": "img-1", "image_ref": "memory://img-1"}],
            )

        self.assertEqual(len(runtime.nfa_vit_client.calls), 0)
        self.assertEqual(results[0]["status"], "completed")
        self.assertIsNone(results[0]["nfa_vit"])
        self.assertEqual(results[0]["final_label"], "real")
        self.assertEqual(
            runtime.openai_client.calls[0]["detector_context"]["final_label"],
            "real",
        )
        self.assertIsNone(runtime.openai_client.calls[0]["detector_context"]["nfa_vit"])
        self.assertEqual(len(store.insert_calls), 1)
        self.assertIsNone(store.insert_calls[0]["rows"][0]["nfa_vit"])
        self.assertEqual(store.insert_calls[0]["rows"][0]["final_label"], "real")

    def test_unique_fake_still_runs_nfa_and_uses_nfa_final_label(self):
        image = Image.new("RGB", (8, 8), "white")
        classification = {"status": "classified", "label": "fake", "confidence": 0.97}
        runtime, store = self.make_runtime(
            dedup_results=[{"status": "unique", "sha256": "sha-2"}],
            unique_records=[
                {
                    "flat_idx": 0,
                    "image": image,
                    "image_ref": "memory://img-2",
                    "vector": np.array([0.2], dtype=np.float32),
                }
            ],
            classifier_outputs=[classification],
            nfa_output={"status": "classified", "final_label": "real"},
        )

        with patch("pipeline_server.image_ref_for_openai", side_effect=lambda ref: f"openai:{ref}"):
            results = pipeline_server.run_pipeline_on_items(
                runtime,
                [{"image_id": "img-2", "image_ref": "memory://img-2"}],
            )

        self.assertEqual(len(runtime.nfa_vit_client.calls), 1)
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[0]["nfa_vit"]["final_label"], "real")
        self.assertEqual(results[0]["final_label"], "real")
        self.assertEqual(
            runtime.openai_client.calls[0]["detector_context"]["final_label"],
            "real",
        )
        self.assertEqual(store.insert_calls[0]["rows"][0]["final_label"], "real")

    def test_duplicate_cache_miss_real_skips_nfa(self):
        image = Image.new("RGB", (8, 8), "white")
        classification = {"status": "classified", "label": "real", "confidence": 0.95}
        runtime, _ = self.make_runtime(
            dedup_results=[
                {
                    "status": "duplicate",
                    "matched_image_id": "orig-1",
                    "matched_sha256": "sha-orig-1",
                }
            ],
            unique_records=[],
            classifier_outputs=[classification],
        )

        with (
            patch("pipeline_server.bytes_from_ref", return_value=b"image-bytes"),
            patch("pipeline_server.bytes_to_pil", return_value=image),
            patch("pipeline_server.image_ref_for_openai", side_effect=lambda ref: f"openai:{ref}"),
        ):
            results = pipeline_server.run_pipeline_on_items(
                runtime,
                [{"image_id": "dup-1", "image_ref": "memory://dup-1"}],
            )

        self.assertEqual(len(runtime.nfa_vit_client.calls), 0)
        self.assertEqual(results[0]["status"], "completed")
        self.assertTrue(results[0]["cache_backfilled"])
        self.assertIsNone(results[0]["nfa_vit"])
        self.assertEqual(results[0]["final_label"], "real")
        self.assertEqual(
            runtime.openai_client.calls[0]["detector_context"]["final_label"],
            "real",
        )

    def test_duplicate_cache_hit_real_normalizes_old_nfa_payload(self):
        classification = {"status": "classified", "label": "real", "confidence": 0.94}
        runtime, _ = self.make_runtime(
            dedup_results=[
                {
                    "status": "duplicate",
                    "matched_image_id": "orig-2",
                    "matched_sha256": "sha-orig-2",
                }
            ],
            unique_records=[],
            classifier_outputs=[],
            cached_results={
                "sha-orig-2": {
                    "image_id": "orig-2",
                    "image_sha256": "sha-orig-2",
                    "pipeline_stage": "openai",
                    "classification": classification,
                    "nfa_vit": {"status": "classified", "final_label": "fake"},
                    "final_label": "fake",
                    "vlm": {"model": "old-vlm"},
                    "openai_skipped": False,
                    "skip_reason": None,
                }
            },
        )

        results = pipeline_server.run_pipeline_on_items(
            runtime,
            [{"image_id": "dup-2", "image_ref": "memory://dup-2"}],
        )

        self.assertEqual(len(runtime.classifier.calls), 0)
        self.assertEqual(len(runtime.nfa_vit_client.calls), 0)
        self.assertEqual(len(runtime.openai_client.calls), 0)
        self.assertTrue(results[0]["served_from_cache"])
        self.assertIsNone(results[0]["nfa_vit"])
        self.assertEqual(results[0]["final_label"], "real")


if __name__ == "__main__":
    unittest.main()
