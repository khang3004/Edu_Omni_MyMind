"""EduMIND API — End-to-End Pipeline Endpoint.

Routes:
    POST /api/v1/pipeline/run   → Upload PDF + audio → full ingestion pipeline
    POST /api/v1/pipeline/mock  → Run pipeline with mock data (no files needed)

The pipeline:
    1. Transcribe audio (Whisper)
    2. Post-process / correct teencode
    3. Parse PDF slides (Docling, layout-aware)
    4. Embed + store both transcript chunks and slide chunks in Qdrant

Usage (Postman / curl):
    # With real files
    curl -X POST http://localhost:8000/api/v1/pipeline/run \\
         -F "pdf=@slides.pdf" \\
         -F "audio=@lecture.mp3"

    # With mock data (no files needed)
    curl -X POST http://localhost:8000/api/v1/pipeline/mock
"""

from __future__ import annotations

import gc
from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from edumind.api.dependencies import get_asr, get_rag
from edumind.models.api import PipelineResponse
from edumind.modules.rag_engine import MultimodalRAG
from edumind.modules.speech_processor import CodeSwitchedASR
from edumind.utils.data_manager import get_raw_dir

router = APIRouter(prefix="/pipeline", tags=["Pipeline — End-to-End"])

_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


async def _run_pipeline(
    asr: CodeSwitchedASR,
    rag: MultimodalRAG,
    pdf_file: UploadFile | None,
    audio_file: UploadFile | None,
    use_mock_audio: bool = False,
) -> PipelineResponse:
    """Shared pipeline logic for real and mock runs."""
    # ── Step 1: Transcription ─────────────────────────────────────────────────
    if use_mock_audio or audio_file is None:
        transcript_result = asr._mock_transcribe()
    else:
        suffix = Path(audio_file.filename or "audio.mp3").suffix.lower()
        if suffix not in _AUDIO_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported audio format '{suffix}'.",
            )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name
        try:
            transcript_result = asr.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Step 2: Post-process transcript ──────────────────────────────────────
    corrected = asr.post_process(transcript_result.text)

    # ── Step 3: Parse PDF ─────────────────────────────────────────────────────
    all_chunks = []
    sources: list[str] = []

    if pdf_file is not None:
        dest_path = get_raw_dir() / (pdf_file.filename or "upload.pdf")
        with open(dest_path, "wb") as f:
            f.write(await pdf_file.read())

        pdf_chunks = rag.ingest_pdf(dest_path)
        for chunk in pdf_chunks:
            chunk.metadata["source_type"] = "slide"
            chunk.metadata["source_file"] = pdf_file.filename or "upload.pdf"
        all_chunks.extend(pdf_chunks)
        sources.append(pdf_file.filename or "upload.pdf")

    # ── Step 4: Ingest transcript ─────────────────────────────────────────────
    transcript_chunks = rag.ingest_text(
        corrected,
        source_name="transcript",
        metadata_extra={"source_type": "transcript"},
    )
    all_chunks.extend(transcript_chunks)
    sources.append("transcript")

    # ── Step 5: Embed + store all chunks ──────────────────────────────────────
    stored_count = rag.embed_and_store(all_chunks)
    gc.collect()

    pdf_chunk_count = len(all_chunks) - len(transcript_chunks)

    return PipelineResponse(
        raw_transcript=transcript_result.text,
        corrected_transcript=corrected,
        chunks_indexed=stored_count,
        transcript_chunks=len(transcript_chunks),
        pdf_chunks=pdf_chunk_count,
        sources=sources,
    )


@router.post(
    "/run",
    response_model=PipelineResponse,
    summary="Run full E2E pipeline",
    description=(
        "Upload a lecture PDF and/or an audio recording. "
        "Runs the full pipeline: ASR → text correction → PDF parsing → embedding → Qdrant storage. "
        "At least one file (pdf or audio) is required."
    ),
)
async def run_pipeline(
    pdf: UploadFile | None = File(default=None, description="Lecture PDF slides"),
    audio: UploadFile | None = File(default=None, description="Lecture audio recording"),
    asr: CodeSwitchedASR = Depends(get_asr),
    rag: MultimodalRAG = Depends(get_rag),
) -> PipelineResponse:
    """Run the full ingestion pipeline with real files."""
    if pdf is None and audio is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'pdf' or 'audio' file is required.",
        )
    return await _run_pipeline(asr, rag, pdf_file=pdf, audio_file=audio)


@router.post(
    "/mock",
    response_model=PipelineResponse,
    summary="Run mock pipeline",
    description=(
        "Runs the full pipeline using mock/simulated audio data. "
        "Optionally accepts a real PDF slide file. "
        "Useful for testing the pipeline without a real audio recording."
    ),
)
async def run_mock_pipeline(
    pdf: UploadFile | None = File(default=None, description="Optional lecture PDF slides"),
    asr: CodeSwitchedASR = Depends(get_asr),
    rag: MultimodalRAG = Depends(get_rag),
) -> PipelineResponse:
    """Run the pipeline with simulated mock audio."""
    return await _run_pipeline(asr, rag, pdf_file=pdf, audio_file=None, use_mock_audio=True)
