"""Mock Embedding Provider.

Generates random vectors for text embeddings. Extremely useful for testing,
development, or as a CPU-friendly zero-dependency fallback.
"""

from __future__ import annotations

import numpy as np

from edumind.core.logging import get_logger
from edumind.services.embedding.base import EmbeddingProvider

logger = get_logger(__name__)


class MockEmbeddingProvider(EmbeddingProvider):
    """Fallback embedding provider that returns standard normal random vectors."""

    def __init__(self, dimension: int = 384):
        """Initializes the mock provider.

        Args:
            dimension: Fixed vector space dimensionality.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Fixed mock dimensionality."""
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generates random mock vectors.

        Args:
            texts: List of text strings.

        Returns:
            A 2D numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.debug("generating_mock_embeddings", count=len(texts), dimension=self.dimension)
        # Seed deterministic results per string to allow caching and consistency in mock testing
        embeddings = []
        for text in texts:
            # Simple deterministic hash seed
            hash_val = sum(ord(c) for c in text) % 10000
            rng = np.random.default_rng(hash_val)
            embeddings.append(rng.normal(0, 1, self.dimension))

        return np.array(embeddings, dtype=np.float32)
