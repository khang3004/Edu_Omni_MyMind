"""EduMIND Core — Infrastructure layer.

Provides logging, exceptions, dependency injection, and cross-cutting concerns.
"""

from __future__ import annotations

from edumind.core.container import (
    get_embedding_provider,
    get_llm_provider,
    get_translation_provider,
    get_vectorstore,
    reset_container,
    set_override,
)
from edumind.core.exceptions import (
    ConfigurationError,
    EduMINDError,
    EmbeddingError,
    TranslationError,
    VectorStoreError,
)
from edumind.core.logging import get_logger

__all__ = [
    "get_embedding_provider",
    "get_llm_provider",
    "get_translation_provider",
    "get_vectorstore",
    "reset_container",
    "set_override",
    "ConfigurationError",
    "EmbeddingError",
    "EduMINDError",
    "TranslationError",
    "VectorStoreError",
    "get_logger",
]
