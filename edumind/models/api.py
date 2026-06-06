"""EduMIND — FastAPI Request/Response Schemas.

Defines Pydantic models used as FastAPI endpoint contracts.
These are *transport* models (API layer), separate from domain models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ServiceStatus(BaseModel):
    """Status of a single internal service."""

    name: str
    ready: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """Minimal health-check response."""

    status: str = "ok"


class DetailedHealthResponse(BaseModel):
    """Full health-check response with per-service status."""

    status: str
    services: list[ServiceStatus]
    config: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


class TranslateRequest(BaseModel):
    """Request body for translation and CMI endpoints."""

    text: str = Field(..., min_length=1, description="Input text (code-mixed Vi/En)")


class TranslateResponse(BaseModel):
    """Response body for translation results."""

    result: str = Field(..., description="Translated output text")
    mode: str = Field(default="rule-based", description="Translation mode ('model' or 'rule-based')")


class TokenLabelDTO(BaseModel):
    """DTO for token-level language classification."""

    token: str
    language: str  # "vi" | "en" | "other"
    confidence: float = 1.0


class CMIResponse(BaseModel):
    """Response body for Code-Mixing Index analysis."""

    score: float = Field(..., description="Normalized CMI score (0.0–1.0)")
    total_tokens: int
    vi_count: int
    en_count: int
    other_count: int
    dominant_language: str
    token_labels: list[TokenLabelDTO]


class TranslatorStatusResponse(BaseModel):
    """Response body for translator status check."""

    mode: str  # "model" | "rule-based"
    is_model_loaded: bool


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


class TranscriptSegmentDTO(BaseModel):
    """DTO for a single timestamped transcript segment."""

    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    """Response body for audio transcription."""

    text: str = Field(..., description="Full transcription text")
    corrected_text: str = Field(..., description="Post-processed / teencode-corrected text")
    segments: list[TranscriptSegmentDTO] = Field(default_factory=list)
    language: str = "vi"
    is_mock: bool = False
    corrections: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {'original': ..., 'corrected': ...} changes made",
    )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


class RAGIngestTextRequest(BaseModel):
    """Request body for ingesting plain text into the vector store."""

    text: str = Field(..., min_length=1)
    source_name: str = Field(default="manual_input")
    metadata_extra: dict[str, Any] = Field(default_factory=dict)


class RAGIngestResponse(BaseModel):
    """Response body for document ingestion."""

    chunks_indexed: int
    source_files: list[str] = Field(default_factory=list)


class RAGQueryRequest(BaseModel):
    """Request body for semantic QA."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    generate_answer: bool = Field(default=True, description="If True, also generate a synthesized answer via LLM")


class RetrievedChunkDTO(BaseModel):
    """DTO for a single matched vector-search result."""

    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Response body for RAG semantic search."""

    query: str
    answer: str = Field(default="", description="LLM-synthesized answer (empty if generate_answer=False)")
    results: list[RetrievedChunkDTO]


class RAGStatusResponse(BaseModel):
    """Response body for RAG collection status."""

    collection_name: str
    points_count: int
    status: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PipelineResponse(BaseModel):
    """Response body for the end-to-end pipeline run."""

    raw_transcript: str
    corrected_transcript: str
    chunks_indexed: int
    transcript_chunks: int
    pdf_chunks: int
    sources: list[str] = Field(default_factory=list)
