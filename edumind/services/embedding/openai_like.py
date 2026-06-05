"""OpenAI-Compatible Generic Embedding Provider.

Supports OpenAI, Ollama, custom local endpoints, and vLLM embedding models.
Uses dynamic API key rotation via KeyRotator.
"""

from __future__ import annotations

import numpy as np
import openai

from edumind.core.exceptions import EmbeddingError
from edumind.core.logging import get_logger
from edumind.services.embedding.base import EmbeddingProvider
from edumind.utils.rotator import KeyRotator
from edumind.utils.retry import retry_on_transient_error

logger = get_logger(__name__)


class OpenAILikeEmbeddingProvider(EmbeddingProvider):
    """Generic Embedding provider mapping to any OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        model_name: str,
        api_key_prefix: str,
        base_url: str | None = None,
    ):
        """Initializes provider parameters.

        Args:
            model_name: Name of the embedding target model.
            api_key_prefix: Environment variable prefix for keys (e.g. OPENAI_API_KEY_).
            base_url: Base endpoint URL (optional).
        """
        self._model_name = model_name
        self._api_key_prefix = api_key_prefix
        self._base_url = base_url.strip() if base_url and base_url.strip() else None
        self._dimension = None

    def _infer_dimension(self) -> None:
        """Infers the dimension by executing a quick single-item token encoding."""
        if self._dimension is not None:
            return

        try:
            logger.info("inferring_openai_like_embedding_dimension", model=self._model_name)
            test_emb = self.encode(["test"])
            self._dimension = int(test_emb.shape[1])
            logger.info("inferred_openai_like_embedding_dimension", dimension=self._dimension)
        except Exception as e:
            logger.warning("failed_to_infer_embedding_dimension_defaulting_to_1536", error=str(e))
            self._dimension = 1536

    @property
    def dimension(self) -> int:
        """Dimensionality of the computed vectors."""
        self._infer_dimension()
        return self._dimension

    @retry_on_transient_error(max_attempts=3)
    def encode(self, texts: list[str]) -> np.ndarray:
        """Computes embeddings for the input texts using the compatible provider.

        Args:
            texts: List of text strings.

        Returns:
            A 2D numpy array containing the computed embeddings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Resolve rotating API key
        api_key = KeyRotator.get_key(self._api_key_prefix)
        if not api_key:
            api_key = "dummy_key_placeholder"

        try:
            logger.debug(
                "requesting_openai_like_embeddings",
                model=self._model_name,
                base_url=self._base_url,
                count=len(texts),
            )
            
            client = openai.OpenAI(
                api_key=api_key,
                base_url=self._base_url,
            )

            response = client.embeddings.create(
                input=texts,
                model=self._model_name,
            )
            
            # Map response data objects to numpy float32 matrix
            vectors = [item.embedding for item in response.data]
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            logger.error("openai_like_embedding_failed", error=str(e))
            raise EmbeddingError(
                "Error computing OpenAI-compatible embeddings",
                details={"model": self._model_name, "base_url": self._base_url, "error": str(e)},
            ) from e
