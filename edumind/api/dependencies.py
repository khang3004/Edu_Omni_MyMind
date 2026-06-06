"""EduMIND FastAPI — Dependency Injection helpers.

Each function returns a cached module instance via functools.lru_cache,
keeping the same singleton pattern as the Streamlit app.
FastAPI routes declare these via Depends() for clean injection.
"""

from __future__ import annotations

from functools import lru_cache

from edumind.modules.rag_engine import MultimodalRAG
from edumind.modules.speech_processor import CodeSwitchedASR
from edumind.modules.vietmix_translator import VietMixTranslator


@lru_cache(maxsize=1)
def _asr_singleton() -> CodeSwitchedASR:
    from edumind.config import get_settings

    settings = get_settings()
    return CodeSwitchedASR(model_name=settings.WHISPER_MODEL)


@lru_cache(maxsize=1)
def _translator_singleton() -> VietMixTranslator:
    return VietMixTranslator()


@lru_cache(maxsize=1)
def _rag_singleton() -> MultimodalRAG:
    return MultimodalRAG()


# ---------------------------------------------------------------------------
# FastAPI Depends() callables
# ---------------------------------------------------------------------------


def get_asr() -> CodeSwitchedASR:
    """FastAPI dependency — returns the shared ASR module instance."""
    return _asr_singleton()


def get_translator() -> VietMixTranslator:
    """FastAPI dependency — returns the shared translator instance."""
    return _translator_singleton()


def get_rag() -> MultimodalRAG:
    """FastAPI dependency — returns the shared RAG engine instance."""
    return _rag_singleton()
