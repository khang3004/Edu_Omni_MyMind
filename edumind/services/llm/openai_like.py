"""OpenAI-Compatible Generic LLM Provider.

Supports Groq, OpenAI, Ollama, custom PEFT local endpoints, and vLLM.
Uses dynamic API key rotation via KeyRotator.
"""

from __future__ import annotations

import openai

from edumind.core.exceptions import LLMError
from edumind.core.logging import get_logger
from edumind.models.chunks import RetrievedChunk
from edumind.services.llm.base import LLMProvider
from edumind.utils.retry import retry_on_transient_error
from edumind.utils.rotator import KeyRotator

logger = get_logger(__name__)


class OpenAILikeLLMProvider(LLMProvider):
    """Generic LLM provider mapping to any OpenAI-compatible completions endpoint."""

    def __init__(
        self,
        model_name: str,
        api_key_prefix: str,
        base_url: str | None = None,
    ):
        """Initializes client attributes.

        Args:
            model_name: Name of the LLM target model.
            api_key_prefix: Environment variable prefix for keys (e.g. GROQ_API_KEY_).
            base_url: Base endpoint URL (optional).
        """
        self._model_name = model_name
        self._api_key_prefix = api_key_prefix
        # Normalize base URL (strip empty spaces)
        self._base_url = base_url.strip() if base_url and base_url.strip() else None

    @retry_on_transient_error(max_attempts=3)
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Formulates a standard prompt and invokes the compatible chat completion API.

        Args:
            question: Original student query.
            contexts: List of retrieved context chunks.

        Returns:
            The synthesized answer.
        """
        # Resolve rotating API key
        api_key = KeyRotator.get_key(self._api_key_prefix)

        # Ollama or local endpoints might not require keys, set dummy placeholder
        if not api_key:
            api_key = "dummy_key_placeholder"

        if not contexts:
            return "No relevant contexts available to generate an answer."

        # Format context strings
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
            logger.info(
                "requesting_openai_like_generation",
                model=self._model_name,
                base_url=self._base_url,
                chunks_count=len(contexts),
            )

            # Instantiate fresh client for each request to easily apply rotating keys
            client = openai.OpenAI(
                api_key=api_key,
                base_url=self._base_url,
            )

            response = client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            text_result = response.choices[0].message.content
            if not text_result:
                raise LLMError("Received empty response from OpenAI-compatible endpoint.")

            logger.info("openai_like_generation_success")
            return text_result.strip()
        except Exception as e:
            logger.error("openai_like_generation_failed", error=str(e))
            raise LLMError(
                "Failed to generate answer from OpenAI-compatible API",
                details={"model": self._model_name, "base_url": self._base_url, "error": str(e)},
            ) from e
