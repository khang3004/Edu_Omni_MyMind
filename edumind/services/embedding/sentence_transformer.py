"""SentenceTransformer Embedding Provider.

Computes text embeddings locally using the HuggingFace ``sentence-transformers`` library.
"""

from __future__ import annotations

import numpy as np

from edumind.core.exceptions import EmbeddingError
from edumind.core.logging import get_logger
from edumind.services.embedding.base import EmbeddingProvider

logger = get_logger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Computes dense embeddings using a local SentenceTransformer model."""

    def __init__(self, model_name: str, device: str = "cpu"):
        """Initializes the provider.

        Args:
            model_name: HuggingFace model string.
            device: Computation device ("cpu", "cuda", or "mps").
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimension = 384  # Safe default, dynamically adjusted after load

    def _load_model(self) -> None:
        """Loads the SentenceTransformer model lazily."""
        if self._model is not None:
            return

        try:
            logger.info("loading_embedding_model", model_name=self._model_name, device=self._device)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            # Infer dimension
            test_emb = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = int(test_emb.shape[1])
            logger.info("loaded_embedding_model", model_name=self._model_name, dimension=self._dimension)
        except Exception as e:
            logger.error("embedding_model_load_failed", model_name=self._model_name, error=str(e))
            raise EmbeddingError(
                f"Failed to load sentence-transformer model '{self._model_name}'",
                details={"model_name": self._model_name, "original_error": str(e)},
            ) from e

    @property
    def dimension(self) -> int:
        """The output dimension of the embedding vectors."""
        self._load_model()
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Computes embeddings for the input texts.

        Args:
            texts: List of text strings.

        Returns:
            A 2D numpy array containing the computed embeddings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._load_model()
        assert self._model is not None

        try:
            logger.debug("encoding_texts", count=len(texts))
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=32,
            )
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error("text_encoding_failed", count=len(texts), error=str(e))
            raise EmbeddingError(
                "Error computing SentenceTransformer embeddings",
                details={"texts_count": len(texts), "original_error": str(e)},
            ) from e
