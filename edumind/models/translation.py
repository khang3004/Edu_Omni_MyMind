"""EduMIND Domain Models — Code-Mixing and Translation.

Defines Pydantic models for language labels and code-mixing index (CMI) results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TokenLabel(BaseModel):
    """Represents a language label mapped to a specific token.

    Attributes:
        token: The original token string.
        language: Identified language code ("vi", "en", or "other").
        confidence: Confidence score of the classification (0.0 to 1.0).
    """

    token: str = Field(..., description="The original token text")
    language: str = Field(..., description='Identified language ("vi", "en", or "other")')
    confidence: float = Field(default=1.0, description="Confidence score (0.0 to 1.0)")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamps the confidence score to [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class CMIResult(BaseModel):
    """Stores computed Code-Mixing Index (CMI) metrics.

    Attributes:
        score: The normalized CMI value (0.0 to 1.0).
        total_tokens: Total tokens analyzed (N).
        vi_count: Count of Vietnamese tokens.
        en_count: Count of English tokens.
        other_count: Count of non-alphabetic/punctuation tokens.
        dominant_language: Identified dominant language code ("vi" or "en").
        token_labels: Token-by-token language label mapping.
    """

    score: float = Field(..., description="Normalized CMI score (0.0 to 1.0)")
    total_tokens: int = Field(..., ge=0, description="Total tokens analyzed (N)")
    vi_count: int = Field(..., ge=0, description="Count of Vietnamese tokens")
    en_count: int = Field(..., ge=0, description="Count of English tokens")
    other_count: int = Field(default=0, ge=0, description="Count of other/punctuation tokens")
    dominant_language: str = Field(default="vi", description="Dominant language code")
    token_labels: list[TokenLabel] = Field(default_factory=list, description="Token labels list")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamps/validates the CMI score to be within range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))
