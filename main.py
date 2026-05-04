import logging
import os
import uuid
from typing import Any

import litserve as ls
import torch
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import Settings
from src.dedup_service import ImageDedupService
from src.embedder import ImageEmbedder
from src.milvus_store import MilvusDedupStore
from src.request_parsing import (
    batch_dedup_requests,
    decode_dedup_request,
    unbatch_dedup_output,
)

try:
    from litserve.callbacks import Callback
except Exception:
    from litserve.callbacks.base import Callback


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("image-dedup-litserve")


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
        if lit_api is not None and getattr(lit_api, "service", None) is not None:
            lit_api.service.reset_predict_state()

    def on_after_predict(self, lit_api=None, *args, **kwargs):
        if lit_api is None or getattr(lit_api, "service", None) is None:
            return

        recovered = lit_api.service.milvus_recovered_this_predict
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
        self.cfg: Settings | None = None
        self.service: ImageDedupService | None = None

    def setup(self, device):
        self.cfg = Settings.from_env()

        torch_device = torch.device(
            device
            if isinstance(device, str)
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        embedder = ImageEmbedder(self.cfg, torch_device)
        store = MilvusDedupStore(self.cfg)
        self.service = ImageDedupService(embedder=embedder, store=store)
        self.service.prepare()

    def decode_request(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        return decode_dedup_request(request, self._cfg)

    def batch(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        return batch_dedup_requests(inputs)

    def predict(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        return self._service.predict_batch(batch)

    def unbatch(self, output: dict[str, Any]) -> list[dict[str, Any]]:
        service = self._service
        return unbatch_dedup_output(
            output,
            milvus_recovered_this_predict=service.milvus_recovered_this_predict,
            last_milvus_error=service.last_milvus_error,
        )

    def encode_response(self, output: dict[str, Any], **kwargs) -> dict[str, Any]:
        return output

    def health(self) -> bool:
        try:
            return self._service.health()
        except Exception:
            logger.exception("Health check failed.")
            return False

    @property
    def _cfg(self) -> Settings:
        if self.cfg is None:
            raise RuntimeError("LitAPI setup has not initialized settings.")
        return self.cfg

    @property
    def _service(self) -> ImageDedupService:
        if self.service is None:
            raise RuntimeError("LitAPI setup has not initialized dedup service.")
        return self.service


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
