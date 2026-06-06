"""EduMIND API — ASR (Speech-to-Text) Endpoints.

Routes:
    POST /api/v1/asr/transcribe   Upload an audio file → full transcript
    POST /api/v1/asr/mock         Return simulated mock transcript (no file needed)

Usage (Postman / curl):
    curl -X POST http://localhost:8000/api/v1/asr/transcribe \\
         -F "file=@lecture.mp3"
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from edumind.api.dependencies import get_asr
from edumind.models.api import TranscribeResponse, TranscriptSegmentDTO
from edumind.modules.speech_processor import CodeSwitchedASR

router = APIRouter(prefix="/asr", tags=["ASR — Speech to Text"])

_SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _build_transcribe_response(asr: CodeSwitchedASR, result) -> TranscribeResponse:
    """Shared helper: build TranscribeResponse from a TranscriptResult."""
    raw_text = result.text
    corrected_text = asr.post_process(raw_text)
    corrections = asr.get_corrections(raw_text, corrected_text)

    segments = [
        TranscriptSegmentDTO(start=seg.start, end=seg.end, text=seg.text)
        for seg in result.segments
    ]

    return TranscribeResponse(
        text=raw_text,
        corrected_text=corrected_text,
        segments=segments,
        language=result.language,
        is_mock=result.is_mock,
        corrections=corrections,
    )


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe audio file",
    description=(
        "Upload an audio file (WAV/MP3/FLAC/M4A/OGG). "
        "Returns the full transcription, teencode-corrected text, timestamped segments, "
        "and the list of corrections applied."
    ),
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    asr: CodeSwitchedASR = Depends(get_asr),
) -> TranscribeResponse:
    """Transcribe an uploaded audio file using OpenAI Whisper."""
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in _SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported audio format '{suffix}'. Supported: {sorted(_SUPPORTED_FORMATS)}",
        )

    # Write to a temp file and transcribe
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = asr.transcribe(tmp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return _build_transcribe_response(asr, result)


@router.post(
    "/mock",
    response_model=TranscribeResponse,
    summary="Simulated mock transcript",
    description=(
        "Returns a pre-built mock transcript without requiring an audio file. "
        "Useful for testing downstream pipeline steps."
    ),
)
def transcribe_mock(asr: CodeSwitchedASR = Depends(get_asr)) -> TranscribeResponse:
    """Return a simulated mock transcription result."""
    result = asr._mock_transcribe()
    return _build_transcribe_response(asr, result)
