"""EduMIND Modules — Core NLP Processing Pipelines

Exposes:
  - CodeSwitchedASR: Bilingual note-taker and speech transcription.
  - VietMixTranslator: Code-mixed machine translation & CMI analytics.
  - MultimodalRAG: Vector store indexing and semantic QA.
"""

from edumind.modules.rag_engine import MultimodalRAG
from edumind.modules.speech_processor import CodeSwitchedASR
from edumind.modules.vietmix_translator import VietMixTranslator

__all__ = [
    "CodeSwitchedASR",
    "VietMixTranslator",
    "MultimodalRAG",
]
