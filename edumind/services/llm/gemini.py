"""Google Gemini LLM Provider.

Synthesizes context-grounded answers utilizing the official Google Generative AI SDK.
"""

from __future__ import annotations

from edumind.core.exceptions import LLMError
from edumind.core.logging import get_logger
from edumind.models.chunks import RetrievedChunk
from edumind.services.llm.base import LLMProvider
from edumind.utils.retry import retry_on_transient_error

logger = get_logger(__name__)


class GeminiLLMProvider(LLMProvider):
    """Google Gemini model client implementing context-aware answer synthesis."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initializes the Gemini provider.

        Args:
            api_key: The Google API Key.
            model_name: The Gemini model variation (e.g. gemini-1.5-flash).
        """
        self._api_key = api_key
        self._model_name = model_name
        self._configured = False

        self._configure()

    def _configure(self) -> None:
        """Configures the google-generativeai SDK."""
        if not self._api_key:
            logger.warning("gemini_api_key_missing_or_empty")
            self._configured = False
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._configured = True
            logger.info("gemini_sdk_configured", model=self._model_name)
        except Exception as e:
            logger.error("gemini_configuration_failed", error=str(e))
            self._configured = False

    @property
    def is_configured(self) -> bool:
        """True if the API key was successfully verified and configured."""
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
        if not self.is_configured:
            raise LLMError("Gemini LLM Provider is not configured or lacks API key.")

        if not contexts:
            return "No relevant contexts available to generate an answer."

        import google.generativeai as genai

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
            "Your task is to answer the student's question based strictly on the provided context sections.\n"
            "If the answer cannot be found in the context, clearly state that you do not have enough information.\n"
            "Provide professional, clear, and comprehensive explanations. Cite your source chunks using standard notations.\n\n"
            f"--- Context ---\n{context_str}"
            f"--- Student Question ---\n{question}\n\n"
            "--- Your Educational Answer ---"
        )

        try:
            logger.info("requesting_gemini_generation", model=self._model_name, chunks_count=len(contexts))
            model = genai.GenerativeModel(self._model_name)
            response = model.generate_content(prompt)
            text_result = response.text.strip()
            logger.info("gemini_generation_success")
            return text_result
        except Exception as e:
            logger.error("gemini_generation_failed", error=str(e))
            raise LLMError(
                "Failed to generate grounded answer from Google Gemini API",
                details={"model": self._model_name, "error": str(e)},
            ) from e
