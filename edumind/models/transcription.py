"""EduMIND Domain Models — Audio Transcription.

Defines Pydantic models for transcription segments and complete results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    """Represents a single segment of transcribed text with start and end timestamps.

    Attributes:
        start: Start time of the segment in seconds.
        end: End time of the segment in seconds.
        text: Transcribed text content for this segment.
    """

    start: float = Field(..., ge=0.0, description="Start time in seconds")
    end: float = Field(..., ge=0.0, description="End time in seconds")
    text: str = Field(..., description="Transcribed text content")

    @model_validator(mode="after")
    def validate_timestamps(self) -> TranscriptSegment:
        """Validates that start time is before or equal to end time."""
        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must be less than or equal to end time ({self.end})")
        return self


class TranscriptResult(BaseModel):
    """Represents the complete transcription result from an audio file.

    Attributes:
        text: Unified text transcription of the entire audio.
        segments: Chronological list of transcribed segments.
        language: Automatically detected language code (e.g., "vi", "en").
        is_mock: Flag indicating whether mock simulation data was used.
    """

    text: str = Field(..., description="Unified full text transcription")
    segments: list[TranscriptSegment] = Field(default_factory=list, description="Chronological segments list")
    language: str = Field(default="vi", description="Detected language code")
    is_mock: bool = Field(default=False, description="Flag indicating if mock/simulation was used")
