"""EduMIND Translation Services Package."""

from __future__ import annotations

from edumind.services.translation.api import GeminiTranslationProvider, OpenAILikeTranslationProvider
from edumind.services.translation.base import TranslationProvider
from edumind.services.translation.huggingface import HuggingFaceTranslationProvider
from edumind.services.translation.rule_based import RuleBasedTranslationProvider

__all__ = [
    "TranslationProvider",
    "HuggingFaceTranslationProvider",
    "RuleBasedTranslationProvider",
    "GeminiTranslationProvider",
    "OpenAILikeTranslationProvider",
]
