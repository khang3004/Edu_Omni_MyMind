"""EduMIND Domain Models — Document Chunking and Retrieval.

Defines Pydantic models for layout-aware PDF chunks and retrieved vector search results.
"""

from __future__ import annotations

import hashlib
from typing import Any
import uuid

from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """Represents a single parsed text chunk with metadata.

    Attributes:
        text: The textual content of the chunk.
        metadata: Associated metadata dictionary (e.g., page, source, section).
        chunk_id: A unique identifier for the chunk.
    """

    text: str = Field(..., min_length=1, description="Textual content of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")
    chunk_id: str = Field(default="", description="Unique identifier for the chunk")

    def model_post_init(self, __context: Any) -> None:
        """Generates a unique chunk_id if not already provided."""
        if not self.chunk_id:
            content_hash = hashlib.md5(self.text.encode("utf-8")).hexdigest()[:8]
            self.chunk_id = f"chunk_{content_hash}_{uuid.uuid4().hex[:6]}"


class RetrievedChunk(BaseModel):
    """Represents a matched chunk retrieved from vector search.

    Attributes:
        text: Retrieved text content.
        metadata: Original chunk metadata.
        score: Cosine similarity score (0.0 to 1.0).
    """

    text: str = Field(..., min_length=1, description="Retrieved text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Original chunk metadata")
    score: float = Field(default=0.0, description="Cosine similarity score (0.0 to 1.0)")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamps/validates the similarity score to be within range [0.0, 1.0]."""
        # Allow slight float precision overflow (e.g. 1.0001 or -0.0001) but clamp it
        return max(0.0, min(1.0, v))
