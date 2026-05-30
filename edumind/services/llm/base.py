"""EduMIND LLM Service — Base Protocol.

Defines the interface for language model generation to support modular synthesis
of vector search contexts.
"""

from __future__ import annotations

from typing import Protocol

from edumind.models.chunks import RetrievedChunk


class LLMProvider(Protocol):
    """Protocol for interacting with generative language model APIs or local backends."""

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Generates a contextual response based on the question and retrieved document contexts.

        Args:
            question: Original user query.
            contexts: Checked retrieved context documents with scores and metadata.

        Returns:
            The synthesized text answer.
        """
        ...
