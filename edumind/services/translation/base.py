"""EduMIND Translation Service — Base Protocol.

Defines the abstract interface for language translation providers (HuggingFace,
Rule-based, etc.) to support the Strategy Pattern.
"""

from __future__ import annotations

from typing import Protocol


class TranslationProvider(Protocol):
    """Protocol for translating code-mixed text between English and Vietnamese."""

    def translate_to_english(self, text: str) -> str:
        """Translates a bilingual code-mixed text into clean standard English.

        Args:
            text: Raw bilingual sentence.

        Returns:
            Clean English translation.
        """
        ...

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates a bilingual code-mixed text into clean standard Vietnamese.

        Args:
            text: Raw bilingual sentence.

        Returns:
            Clean Vietnamese translation.
        """
        ...
