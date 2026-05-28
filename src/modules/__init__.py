"""
EduMIND Modules — Các pipeline xử lý NLP chính
================================================
  - speech_processor: Nhận dạng giọng nói song ngữ (Code-Switched ASR)
  - vietmix_translator: Dịch thuật Vi-En pha trộn ngôn ngữ + CMI
  - rag_engine: RAG đa phương thức với Qdrant
"""

from src.modules.speech_processor import CodeSwitchedASR
from src.modules.vietmix_translator import VietMixTranslator
from src.modules.rag_engine import MultimodalRAG

__all__ = [
    "CodeSwitchedASR",
    "VietMixTranslator",
    "MultimodalRAG",
]
