"""Google Gemini LLM Provider.

Synthesizes context-grounded answers utilizing the official Google Gen AI SDK.
Supports dynamic API key rotation via KeyRotator.

NOTE: Migrated from deprecated `google.generativeai` to `google.genai` (google-genai package).
"""

from __future__ import annotations

try:
    from google import genai as _google_genai
    from google.genai import types as _genai_types
except ImportError:
    _google_genai = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]

from edumind.core.exceptions import LLMError
from edumind.core.logging import get_logger
from edumind.models.chunks import RetrievedChunk
from edumind.services.llm.base import LLMProvider
from edumind.utils.retry import retry_on_transient_error
from edumind.utils.rotator import KeyRotator

logger = get_logger(__name__)


class GeminiLLMProvider(LLMProvider):
    """Google Gemini model client implementing context-aware answer synthesis."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gemini-2.0-flash",
        api_key_prefix: str = "GEMINI_API_KEY_",
    ):
        """Initializes the Gemini provider.

        Args:
            api_key: The fallback static Google API Key.
            model_name: The Gemini model variation (e.g. gemini-2.0-flash).
            api_key_prefix: Tiền tố xoay vòng key.
        """
        self._api_key = api_key
        self._model_name = model_name
        self._api_key_prefix = api_key_prefix
        self._configured = True

    @property
    def is_configured(self) -> bool:
        """True if configuration parameters are present."""
        return self._configured

    @retry_on_transient_error(max_attempts=3)
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Sends the question and context block to Gemini to synthesize a grounded answer.

        Args:
            question: User question string.
            contexts: List of RetrievedChunk matched items.

        Returns:
            The synthesized answer string.
        """
        # Resolve rotating API key
        key = KeyRotator.get_key(self._api_key_prefix)
        if not key:
            key = self._api_key

        if not key:
            raise LLMError("Gemini LLM Provider has no valid API key configured.")

        if not contexts:
            return "No relevant contexts available to generate an answer."

        # Construct a grounded RAG prompt
        context_str = ""
        for i, ctx in enumerate(contexts, start=1):
            source = ctx.metadata.get("source_file", "Unknown Source")
            page = ctx.metadata.get("page_number", "Unknown Page")
            section = ctx.metadata.get("section_header", "")
            loc = f"Page {page}, {source}"
            if section and section != "Untitled Section":
                loc += f" — §{section}"
            context_str += f"--- Context Chunk {i} [{loc}] ---\n{ctx.text}\n\n"

        prompt = (
            "You are EduMIND, an expert educational teaching assistant.\n"
            "Your task is to answer the student's question "
            "based strictly on the provided context sections.\n"
            "If the answer cannot be found in the context, "
            "clearly state that you do not have enough information.\n"
            "Provide professional, clear, and comprehensive explanations. "
            "Cite your source chunks using standard notations.\n\n"
            f"--- Context ---\n{context_str}"
            f"--- Student Question ---\n{question}\n\n"
            "--- Your Educational Answer ---"
        )

        try:
            logger.info(
                "requesting_gemini_generation", model=self._model_name, chunks_count=len(contexts)
            )
            if _google_genai is None:
                raise LLMError(
                    "google-genai package not installed. Run: uv add google-genai"
                )
            client = _google_genai.Client(api_key=key)
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=_genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            text_result = response.text.strip()
            logger.info("gemini_generation_success")
            return text_result
        except Exception as e:
            logger.error("gemini_generation_failed", error=str(e))
            raise LLMError(
                "Failed to generate grounded answer from Google Gemini API",
                details={"model": self._model_name, "error": str(e)},
            ) from e
