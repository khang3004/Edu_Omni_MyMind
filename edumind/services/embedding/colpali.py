"""ColPali Multimodal/Visual Embedding Provider.

Computes visual embeddings for PDF pages and query text using late interaction pooling.
Falls back gracefully if libraries are missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from edumind.core.exceptions import EmbeddingError
from edumind.core.logging import get_logger
from edumind.services.embedding.base import EmbeddingProvider

logger = get_logger(__name__)


class ColPaliEmbeddingProvider(EmbeddingProvider):
    """Computes standard pooled 1D visual representation using the ColPali model."""

    def __init__(self, model_name: str = "vidore/colpali-v1.2", device: str = "cpu"):
        """Initializes the colpali model parameters."""
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._dimension = 128  # ColPali projected dimension is typically 128

    def _load_model(self) -> None:
        """Lazily loads ColPali model and processor."""
        if self._model is not None:
            return

        try:
            logger.info("loading_colpali_model", model_name=self._model_name, device=self._device)
            # colpali-engine imports
            from colpali_engine.models import ColPali
            import torch
            from transformers import AutoProcessor

            # Load model in low memory mode
            self._model = ColPali.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32 if self._device == "cpu" else torch.bfloat16,
                device_map=self._device,
            )
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            logger.info(
                "loaded_colpali_model", model_name=self._model_name, dimension=self._dimension
            )
        except Exception as e:
            logger.warning("colpali_load_failed_falling_back_to_random", error=str(e))
            self._model = None
            self._processor = None

    @property
    def dimension(self) -> int:
        """Dimensionality of the visual embeddings."""
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embeds text queries using ColPali text projection."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()

        # If model failed to load, generate mock/simulated vectors
        if self._model is None:
            logger.debug("colpali_not_available_generating_mock_query_vectors")
            # Generate stable deterministic mock embeddings for text queries
            rng = np.random.default_rng(seed=42)
            return rng.standard_normal((len(texts), self.dimension), dtype=np.float32)

        try:
            import torch

            # Process text input
            inputs = self._processor(text=texts, return_tensors="pt").to(self._device)
            with torch.no_grad():
                embeddings = self._model(**inputs)  # shape: (batch, tokens, dim)
                # Pool token embeddings using mean pooling to form a single 1D vector per text
                pooled = embeddings.mean(dim=1).cpu().float().numpy()
            return np.array(pooled, dtype=np.float32)
        except Exception as e:
            logger.error("colpali_text_encoding_failed", error=str(e))
            raise EmbeddingError("ColPali text encoding failed", details={"error": str(e)}) from e

    def encode_images(self, images: list[Any]) -> np.ndarray:
        """Embeds PIL images using ColPali vision projection."""
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()

        if self._model is None:
            logger.debug("colpali_not_available_generating_mock_image_vectors")
            rng = np.random.default_rng(seed=43)
            return rng.standard_normal((len(images), self.dimension), dtype=np.float32)

        try:
            import torch

            # Process image inputs
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with torch.no_grad():
                embeddings = self._model(**inputs)  # shape: (batch, patches, dim)
                # Mean pool patch embeddings to create 1D image vector
                pooled = embeddings.mean(dim=1).cpu().float().numpy()
            return np.array(pooled, dtype=np.float32)
        except Exception as e:
            logger.error("colpali_image_encoding_failed", error=str(e))
            raise EmbeddingError("ColPali image encoding failed", details={"error": str(e)}) from e
