"""EduMIND Core — Custom Exception Hierarchy.

All application-specific exceptions inherit from ``EduMINDError`` so that callers
can catch a single base class for broad error handling while still distinguishing
sub-types when finer granularity is needed.

Usage::

    from edumind.core.exceptions import EmbeddingError
    raise EmbeddingError("Model download failed", model_name="all-MiniLM-L6-v2")
"""

from __future__ import annotations


class EduMINDError(Exception):
    """Base exception for all EduMIND application errors.

    Attributes:
        message: Human-readable error description.
        context: Arbitrary key-value pairs for structured logging.
    """

    def __init__(self, message: str = "", **context: object) -> None:
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{details}]"
        return base


class ConfigurationError(EduMINDError):
    """Raised when application configuration is invalid or missing."""


class EmbeddingError(EduMINDError):
    """Raised when text embedding generation fails."""


class VectorStoreError(EduMINDError):
    """Raised when vector database operations fail."""


class TranscriptionError(EduMINDError):
    """Raised when speech-to-text transcription fails."""


class TranslationError(EduMINDError):
    """Raised when machine translation fails."""


class LLMError(EduMINDError):
    """Raised when LLM answer generation fails."""


class InputValidationError(EduMINDError):
    """Raised when user input fails sanitization or validation."""
