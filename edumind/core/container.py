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
    ColPaliEmbeddingProvider,
    EmbeddingProvider,
    MockEmbeddingProvider,
    OpenAILikeEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from edumind.services.graphstore import GraphStore, MockGraphStore, Neo4jGraphStore
from edumind.services.llm import (
    GeminiLLMProvider,
    LLMProvider,
    OpenAILikeLLMProvider,
    TemplateLLMProvider,
)
from edumind.services.translation import HuggingFaceTranslationProvider, TranslationProvider
from edumind.services.vectorstore import QdrantVectorStore, VectorStore
from edumind.utils.rotator import KeyRotator

logger = get_logger(__name__)

_lock = threading.RLock()

# Cached singleton service instances
_embedding_provider: EmbeddingProvider | None = None
_vectorstore: VectorStore | None = None
_llm_provider: LLMProvider | None = None
_translation_provider: TranslationProvider | None = None
_graph_store: GraphStore | None = None
_colpali_provider: ColPaliEmbeddingProvider | None = None

# Testing override registry
_overrides: dict[str, Any] = {}


def reset_container() -> None:
    """Wipes the container cache and clears all testing overrides."""
    global _embedding_provider, _vectorstore, _llm_provider, _translation_provider, _graph_store, _colpali_provider
    with _lock:
        _embedding_provider = None
        _vectorstore = None
        _llm_provider = None
        _translation_provider = None
        _graph_store = None
        _colpali_provider = None
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

    Resolves via setting: ``EMBEDDING_PROVIDER``, ``EMBEDDING_MODEL``, ``EMBEDDING_BASE_URL``, ``EMBEDDING_API_KEY_PREFIX``.
    Falls back to ``MockEmbeddingProvider`` on load failures or if explicitly disabled.
    """
    global _embedding_provider

    if "embedding" in _overrides:
        return _overrides["embedding"]  # type: ignore[no-any-return]

    with _lock:
        if _embedding_provider is None:
            settings = get_settings()
            provider = settings.EMBEDDING_PROVIDER.lower()
            model_name = settings.EMBEDDING_MODEL

            if provider in ("none", "mock") or model_name.lower() in ("none", "mock"):
                logger.info("using_mock_embedding_provider_by_config")
                _embedding_provider = MockEmbeddingProvider()
            elif provider == "local":
                try:
                    _embedding_provider = SentenceTransformerEmbeddingProvider(
                        model_name=model_name,
                        device=settings.DEVICE,
                    )
                except Exception as e:
                    logger.warning("sentence_transformer_failed_using_mock_fallback", error=str(e))
                    _embedding_provider = MockEmbeddingProvider()
            else:
                logger.info("using_openai_like_embedding_provider", provider=provider, model=model_name)
                try:
                    _embedding_provider = OpenAILikeEmbeddingProvider(
                        model_name=model_name,
                        api_key_prefix=settings.EMBEDDING_API_KEY_PREFIX,
                        base_url=settings.EMBEDDING_BASE_URL,
                    )
                except Exception as e:
                    logger.warning("openai_like_embedding_provider_failed_using_mock_fallback", error=str(e))
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
                path=str(settings.DATA_DIR / "qdrant_db"),
            )

        return _vectorstore


def get_llm_provider() -> LLMProvider:
    """Thread-safe factory retrieving the LLMProvider.

    Resolves via settings: ``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_BASE_URL``, ``LLM_API_KEY_PREFIX``.
    Falls back to template string synthesizer if no API key or provider is missing.
    """
    global _llm_provider

    if "llm" in _overrides:
        return _overrides["llm"]  # type: ignore[no-any-return]

    with _lock:
        if _llm_provider is None:
            settings = get_settings()
            provider = settings.LLM_PROVIDER.lower()

            if provider in ("mock", "none", "template"):
                logger.info("using_template_llm_provider_by_config")
                _llm_provider = TemplateLLMProvider()
            elif provider == "google":
                rotator_key = KeyRotator.get_key(settings.LLM_API_KEY_PREFIX)
                static_key = settings.GOOGLE_API_KEY.get_secret_value()
                if not rotator_key and not static_key:
                    logger.info("google_api_key_missing_using_template_llm")
                    _llm_provider = TemplateLLMProvider()
                else:
                    _llm_provider = GeminiLLMProvider(
                        api_key=static_key,
                        model_name=settings.LLM_MODEL,
                        api_key_prefix=settings.LLM_API_KEY_PREFIX,
                    )
            else:
                logger.info("using_openai_like_llm_provider", provider=provider, model=settings.LLM_MODEL)
                _llm_provider = OpenAILikeLLMProvider(
                    model_name=settings.LLM_MODEL,
                    api_key_prefix=settings.LLM_API_KEY_PREFIX,
                    base_url=settings.LLM_BASE_URL,
                )

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


def get_graph_store() -> GraphStore:
    """Thread-safe factory retrieving the GraphStore.

    Falls back to MockGraphStore on failures.
    """
    global _graph_store

    if "graph_store" in _overrides:
        return _overrides["graph_store"]  # type: ignore[no-any-return]

    with _lock:
        if _graph_store is None:
            settings = get_settings()
            try:
                _graph_store = Neo4jGraphStore(
                    uri=settings.NEO4J_URI,
                    user=settings.NEO4J_USER,
                    password=settings.NEO4J_PASSWORD.get_secret_value(),
                )
                if not _graph_store.is_ready:
                    logger.warning("neo4j_initialization_failed_using_mock_fallback")
                    _graph_store = MockGraphStore()
            except Exception as e:
                logger.warning("neo4j_factory_failed_using_mock_fallback", error=str(e))
                _graph_store = MockGraphStore()

        return _graph_store


def get_colpali_provider() -> ColPaliEmbeddingProvider:
    """Thread-safe factory retrieving the ColPaliEmbeddingProvider."""
    global _colpali_provider

    if "colpali" in _overrides:
        return _overrides["colpali"]  # type: ignore[no-any-return]

    with _lock:
        if _colpali_provider is None:
            settings = get_settings()
            _colpali_provider = ColPaliEmbeddingProvider(
                model_name=settings.COLPALI_MODEL,
                device=settings.DEVICE,
            )
        return _colpali_provider


