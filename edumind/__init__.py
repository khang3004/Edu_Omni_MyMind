"""EduMIND — All-in-One Multimodal Lecture Assistant.

Integrated multimodal lecture assistant system:
  1. Bilingual Note-Taker (Code-Switched ASR).
  2. VietMix Translator (Vi-En Code-Mixed Translation + CMI computation).
  3. Anti-Forget Multimodal RAG (Intelligent chunking + Qdrant Vector search).

Developed by HCMUS Underdogs.
"""

from __future__ import annotations

from pathlib import Path

# Load environment variables early so Hugging Face and other libraries see them
try:
    from dotenv import load_dotenv
    proj_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=proj_root / ".env")
except ImportError:
    pass

# Suppress annoying/verbose Hugging Face transformers warnings.
try:
    import transformers

    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass

from edumind.modules.rag_engine import MultimodalRAG
from edumind.modules.speech_processor import CodeSwitchedASR
from edumind.modules.vietmix_translator import VietMixTranslator

__version__ = "2.0.0"
__app_name__ = "EduMIND"

__all__ = [
    "MultimodalRAG",
    "CodeSwitchedASR",
    "VietMixTranslator",
]
