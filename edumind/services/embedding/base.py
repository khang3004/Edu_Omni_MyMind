"""EduMIND Embedding Service — Base Protocol.

Defines the interface for text embedding models to ensure Strategy Pattern conformance.
"""

from __future__ import annotations

from typing import Protocol
import numpy as np


class EmbeddingProvider(Protocol):
    """Protocol defining the interface for document embedding computation."""

    @property
    def dimension(self) -> int:
        """The dimensionality of the computed embeddings."""
        ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Computes embeddings for a list of input texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            A 2D numpy array of shape (len(texts), dimension).
        """
        ...
