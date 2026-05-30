"""Template-based Grounded Mock LLM Provider.

Formulates educational answers via clean string formatting and metadata citations,
acting as a zero-cost local fallback when external LLM APIs are disabled or offline.
"""

from __future__ import annotations

from edumind.core.logging import get_logger
from edumind.models.chunks import RetrievedChunk
from edumind.services.llm.base import LLMProvider

logger = get_logger(__name__)


class TemplateLLMProvider(LLMProvider):
    """Fallback LLM provider that formats context chunks directly into a readable list."""

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Synthesizes an answer using the top retrieved contexts directly.

        Args:
            question: Original user query.
            contexts: List of RetrievedChunk matched items.

        Returns:
            A formatted string containing answer synthesis and citations.
        """
        logger.info("generating_template_fallback_answer", contexts_count=len(contexts))

        if not contexts:
            return (
                "❌ No relevant information found in the uploaded documents. "
                "Please try asking another question or uploading more files."
            )

        answer_parts: list[str] = []
        answer_parts.append(f"📋 **Question:** {question}\n")
        answer_parts.append("📖 **Synthesized Answer from Documents (Local Fallback):**\n")

        # Synthesize answers from top-3 relevant context chunks
        for i, ctx in enumerate(contexts[:3], start=1):
            page = ctx.metadata.get("page_number", "?")
            source = ctx.metadata.get("source_file", "Unknown")
            section = ctx.metadata.get("section_header", "")

            citation = f"[Page {page}, {source}]"
            if section and section != "Untitled Section":
                citation = f"[Page {page}, {source} — §{section}]"

            text_preview = ctx.text[:300] + "..." if len(ctx.text) > 300 else ctx.text
            answer_parts.append(
                f"**{i}.** {text_preview}\n   — *{citation}* (Relevance: {ctx.score:.2f})\n"
            )

        # Footer
        answer_parts.append(
            "\n💡 *Note: The answers above are synthesized directly from your matched "
            "document contexts. Verify with original sources for maximum accuracy.*"
        )

        return "\n".join(answer_parts)
