"""EduMIND Dependency Injection Container.

Provides thread-safe, lazily-initialized factories for core services.
Supports runtime overrides to facilitate mock testing.
"""

from __future__ import annotations

import threading
from typing import Any

from edumind.config import get_settings
from edumind.core.logging import get_logger
from edumind.services.embedding import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from edumind.services.llm import GeminiLLMProvider, LLMProvider, TemplateLLMProvider
from edumind.services.translation import HuggingFaceTranslationProvider, TranslationProvider
from edumind.services.vectorstore import QdrantVectorStore, VectorStore

logger = get_logger(__name__)

_lock = threading.RLock()

# Cached singleton service instances
_embedding_provider: EmbeddingProvider | None = None
_vectorstore: VectorStore | None = None
_llm_provider: LLMProvider | None = None
_translation_provider: TranslationProvider | None = None

# Testing override registry
_overrides: dict[str, Any] = {}


def reset_container() -> None:
    """Wipes the container cache and clears all testing overrides."""
    global _embedding_provider, _vectorstore, _llm_provider, _translation_provider
    with _lock:
        _embedding_provider = None
        _vectorstore = None
        _llm_provider = None
        _translation_provider = None
        _overrides.clear()
    logger.info("di_container_reset_complete")


def set_override(service_name: str, instance: Any) -> None:
    """Sets a testing override for a specific service factory.

    Args:
        service_name: The name of the service ("embedding", "vectorstore", "llm", "translation").
        instance: The mock/custom object to inject.
    """
    with _lock:
        _overrides[service_name] = instance
    logger.info("di_container_override_set", service=service_name, type=type(instance).__name__)


def get_embedding_provider() -> EmbeddingProvider:
    """Thread-safe factory retrieving the EmbeddingProvider.

    Resolves via setting: ``EMBEDDING_MODEL``. Falls back to ``MockEmbeddingProvider``
    on load failures or if explicitly disabled.
    """
    global _embedding_provider

    if "embedding" in _overrides:
        return _overrides["embedding"]  # type: ignore[no-any-return]

    with _lock:
        if _embedding_provider is None:
            settings = get_settings()
            model_name = settings.EMBEDDING_MODEL

            if model_name.lower() in ("none", "mock"):
                logger.info("using_mock_embedding_provider_by_config")
                _embedding_provider = MockEmbeddingProvider()
            else:
                try:
                    _embedding_provider = SentenceTransformerEmbeddingProvider(
                        model_name=model_name,
                        device=settings.DEVICE,
                    )
                except Exception as e:
                    logger.warning("sentence_transformer_failed_using_mock_fallback", error=str(e))
                    _embedding_provider = MockEmbeddingProvider()

        return _embedding_provider


def get_vectorstore() -> VectorStore:
    """Thread-safe factory retrieving the VectorStore.

    Resolves via connection settings: ``QDRANT_MODE``, ``QDRANT_HOST``, ``QDRANT_PORT``.
    """
    global _vectorstore

    if "vectorstore" in _overrides:
        return _overrides["vectorstore"]  # type: ignore[no-any-return]

    with _lock:
        if _vectorstore is None:
            settings = get_settings()
            emb_provider = get_embedding_provider()

            _vectorstore = QdrantVectorStore(
                mode=settings.QDRANT_MODE,
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY.get_secret_value(),
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embedding_dim=emb_provider.dimension,
            )

        return _vectorstore


def get_llm_provider() -> LLMProvider:
    """Thread-safe factory retrieving the LLMProvider.

    Resolves via ``GOOGLE_API_KEY``. Falls back to template string synthesizer
    if API key is missing.
    """
    global _llm_provider

    if "llm" in _overrides:
        return _overrides["llm"]  # type: ignore[no-any-return]

    with _lock:
        if _llm_provider is None:
            settings = get_settings()
            api_key = settings.GOOGLE_API_KEY.get_secret_value()

            if not api_key:
                logger.info("google_api_key_missing_using_template_llm")
                _llm_provider = TemplateLLMProvider()
            else:
                _llm_provider = GeminiLLMProvider(api_key=api_key)

        return _llm_provider


def get_translation_provider() -> TranslationProvider:
    """Thread-safe factory retrieving the TranslationProvider.

    Resolves via setting ``TRANSLATION_MODEL``. Defaults to Rule-based fallback.
    """
    global _translation_provider

    if "translation" in _overrides:
        return _overrides["translation"]  # type: ignore[no-any-return]

    with _lock:
        if _translation_provider is None:
            settings = get_settings()
            model_name = settings.TRANSLATION_MODEL

            _translation_provider = HuggingFaceTranslationProvider(model_name=model_name)

        return _translation_provider
