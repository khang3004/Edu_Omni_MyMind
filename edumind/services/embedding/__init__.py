"""EduMIND Embedding Services Package."""

from __future__ import annotations

from edumind.services.embedding.base import EmbeddingProvider
from edumind.services.embedding.mock import MockEmbeddingProvider
from edumind.services.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from edumind.services.embedding.colpali import ColPaliEmbeddingProvider
from edumind.services.embedding.openai_like import OpenAILikeEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "ColPaliEmbeddingProvider",
    "OpenAILikeEmbeddingProvider",
]
