import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.config import Settings


logger = logging.getLogger("image-dedup-litserve")


class ImageEmbedder:
    def __init__(self, cfg: Settings, device: torch.device):
        self.cfg = cfg
        self.device = device

        logger.info("Loading model from %s on %s", cfg.model_path, device)
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_path)
        self.model = AutoModel.from_pretrained(cfg.model_path).to(device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, images: list[Image.Image]) -> np.ndarray:
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
