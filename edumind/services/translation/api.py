"""API-based Machine Translation Providers (Google Gemini & OpenAI).

Delegates bilingual translation queries to cloud LLM endpoints.
"""

from __future__ import annotations

import openai

try:
    from google import genai as _google_genai
    from google.genai import types as _genai_types
except ImportError:
    _google_genai = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]

from edumind.core.logging import get_logger
from edumind.services.translation.base import TranslationProvider
from edumind.services.translation.rule_based import RuleBasedTranslationProvider
from edumind.utils.rotator import KeyRotator

logger = get_logger(__name__)


class GeminiTranslationProvider(TranslationProvider):
    """Google Gemini translation provider using official SDK."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key_prefix: str = "GEMINI_API_KEY_",
        api_key: str = "",
    ):
        """Initializes the provider."""
        self._model_name = model_name or "gemini-2.0-flash"
        self._api_key_prefix = api_key_prefix
        self._api_key = api_key
        self._fallback = RuleBasedTranslationProvider()

    def translate_to_english(self, text: str) -> str:
        """Translates text to standard English using Gemini API."""
        return self._translate(text, target="English")

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates text to standard Vietnamese using Gemini API."""
        return self._translate(text, target="Vietnamese")

    def _translate(self, text: str, target: str) -> str:
        if not text or not text.strip():
            return ""

        key = KeyRotator.get_key(self._api_key_prefix) or self._api_key
        if not key:
            try:
                from edumind.core.container import get_settings
                key = get_settings().GOOGLE_API_KEY.get_secret_value()
            except Exception:
                pass

        if not key:
            logger.warning("gemini_translation_missing_key_using_fallback")
            if target == "English":
                return self._fallback.translate_to_english(text)
            return self._fallback.translate_to_vietnamese(text)

        try:
            if _google_genai is None:
                raise ImportError("google-genai package not installed")

            client = _google_genai.Client(api_key=key)
            prompt = (
                f"Translate the following code-mixed AI/Data Science text into standard {target}. "
                "Preserve technical terms and professional meaning. Output ONLY the translated text, nothing else.\n\n"
                f"Text: {text}\n"
                "Translation:"
            )
            logger.info("requesting_gemini_translation", model=self._model_name, target=target)
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error("gemini_translation_failed_falling_back", error=str(e))
            if target == "English":
                return self._fallback.translate_to_english(text)
            return self._fallback.translate_to_vietnamese(text)


class OpenAILikeTranslationProvider(TranslationProvider):
    """Generic OpenAI-compatible API translation provider (OpenAI, Groq, local vLLM)."""

    def __init__(
        self,
        model_name: str,
        api_key_prefix: str,
        base_url: str | None = None,
    ):
        """Initializes the provider."""
        self._model_name = model_name
        self._api_key_prefix = api_key_prefix
        self._base_url = base_url.strip() if base_url and base_url.strip() else None
        self._fallback = RuleBasedTranslationProvider()

    def translate_to_english(self, text: str) -> str:
        """Translates text to standard English using OpenAI-like API."""
        return self._translate(text, target="English")

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates text to standard Vietnamese using OpenAI-like API."""
        return self._translate(text, target="Vietnamese")

    def _translate(self, text: str, target: str) -> str:
        if not text or not text.strip():
            return ""

        key = KeyRotator.get_key(self._api_key_prefix)
        if not key:
            key = "dummy_key_placeholder"

        try:
            client = openai.OpenAI(
                api_key=key,
                base_url=self._base_url,
            )
            prompt = (
                f"Translate the following code-mixed AI/Data Science text into standard {target}. "
                "Preserve technical terms and professional meaning. Output ONLY the translated text, nothing else.\n\n"
                f"Text: {text}\n"
                "Translation:"
            )
            logger.info("requesting_openai_like_translation", model=self._model_name, target=target)
            response = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            res = response.choices[0].message.content
            if not res:
                raise ValueError("Empty response received from LLM")
            return res.strip()
        except Exception as e:
            logger.error("openai_like_translation_failed_falling_back", error=str(e))
            if target == "English":
                return self._fallback.translate_to_english(text)
            return self._fallback.translate_to_vietnamese(text)
