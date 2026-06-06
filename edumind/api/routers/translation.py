"""EduMIND API — Translation Endpoints.

Routes:
    GET  /api/v1/translate/status    → Current translator mode
    POST /api/v1/translate/analyze   → CMI score + token-level language breakdown
    POST /api/v1/translate/to-en     → Translate code-mixed text → English
    POST /api/v1/translate/to-vi     → Translate code-mixed text → Vietnamese

Usage (Postman / curl):
    curl -X POST http://localhost:8000/api/v1/translate/analyze \\
         -H "Content-Type: application/json" \\
         -d '{"text": "Hôm nay mình sẽ discuss về loss function trong deep learning"}'

    curl -X POST http://localhost:8000/api/v1/translate/to-en \\
         -H "Content-Type: application/json" \\
         -d '{"text": "Hôm nay mình sẽ discuss về loss function trong deep learning"}'
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from edumind.api.dependencies import get_translator
from edumind.models.api import (
    CMIResponse,
    TokenLabelDTO,
    TranslateRequest,
    TranslateResponse,
    TranslatorStatusResponse,
)
from edumind.modules.vietmix_translator import VietMixTranslator

router = APIRouter(prefix="/translate", tags=["Translation — VietMix"])


@router.get(
    "/status",
    response_model=TranslatorStatusResponse,
    summary="Translator status",
    description="Returns whether the translator is running in neural-model mode or rule-based mode.",
)
def translation_status(
    translator: VietMixTranslator = Depends(get_translator),
) -> TranslatorStatusResponse:
    """Return current translator operational mode."""
    return TranslatorStatusResponse(
        mode=translator.mode,
        is_model_loaded=translator.is_model_loaded,
    )


@router.post(
    "/analyze",
    response_model=CMIResponse,
    summary="Code-Mixing Index analysis",
    description=(
        "Analyzes a code-mixed Vietnamese-English sentence. "
        "Returns the CMI score (0.0 = monolingual, 1.0 = fully mixed), "
        "token counts per language, and token-level language labels."
    ),
)
def analyze_cmi(
    body: TranslateRequest,
    translator: VietMixTranslator = Depends(get_translator),
) -> CMIResponse:
    """Compute Code-Mixing Index and per-token language labels."""
    result = translator.calculate_cmi(body.text)

    token_labels = [
        TokenLabelDTO(token=tl.token, language=tl.language, confidence=tl.confidence)
        for tl in result.token_labels
    ]

    return CMIResponse(
        score=result.score,
        total_tokens=result.total_tokens,
        vi_count=result.vi_count,
        en_count=result.en_count,
        other_count=result.other_count,
        dominant_language=result.dominant_language,
        token_labels=token_labels,
    )


@router.post(
    "/to-en",
    response_model=TranslateResponse,
    summary="Translate to English",
    description=(
        "Translates code-mixed Vietnamese-English text into standard English. "
        "Technical/domain terms are preserved as-is."
    ),
)
def translate_to_english(
    body: TranslateRequest,
    translator: VietMixTranslator = Depends(get_translator),
) -> TranslateResponse:
    """Translate input text to clean English."""
    result = translator.translate_to_english(body.text)
    return TranslateResponse(result=result, mode=translator.mode)


@router.post(
    "/to-vi",
    response_model=TranslateResponse,
    summary="Translate to Vietnamese",
    description=(
        "Translates code-mixed Vietnamese-English text into standard Vietnamese. "
        "Technical/domain English terms are preserved as-is in the output."
    ),
)
def translate_to_vietnamese(
    body: TranslateRequest,
    translator: VietMixTranslator = Depends(get_translator),
) -> TranslateResponse:
    """Translate input text to clean Vietnamese."""
    result = translator.translate_to_vietnamese(body.text)
    return TranslateResponse(result=result, mode=translator.mode)
