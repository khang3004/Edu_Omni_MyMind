"""EduMIND — All-in-One Multimodal Lecture Assistant

Integrated multimodal lecture assistant system:
  1. Bilingual Note-Taker (Code-Switched ASR).
  2. VietMix Translator (Vi-En Code-Mixed Translation + CMI computation).
  3. Anti-Forget Multimodal RAG (Intelligent chunking + Qdrant Vector search).

Developed by HCMUS Underdogs.
"""

# Suppress annoying/verbose Hugging Face transformers warnings.
# Streamlit's file watcher inspects modules dynamically, which triggers warning
# logs from transformers' dynamic alias module attributes (like __path__).
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass

__version__ = "1.0.0"
__app_name__ = "EduMIND"

