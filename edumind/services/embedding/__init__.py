"""EduMIND Embedding Services Package."""

from __future__ import annotations

from edumind.services.embedding.base import EmbeddingProvider
from edumind.services.embedding.mock import MockEmbeddingProvider
from edumind.services.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]
