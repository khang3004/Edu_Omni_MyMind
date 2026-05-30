"""EduMIND — Central Configuration.

Loads configuration from environment variables (.env) with smart default values.
Automatically detects the optimal computation device (CUDA → MPS → CPU).

Usage:
    from edumind.config import settings
    print(settings.DEVICE)  # torch.device
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import torch

# ----------------------------------------------------------------------
# Load environment variables from .env file at the project root
# ----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")


# ----------------------------------------------------------------------
# Helper function to detect optimal computation device
# ----------------------------------------------------------------------
def _detect_device(preference: str = "auto") -> torch.device:
    """Detects the optimal computation device available on the system.

    Args:
        preference: Device preference. Can be "auto" (automatic detection),
            "cuda", "mps", or "cpu".

    Returns:
        The most suitable torch.device instance.
    """
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        logger.info("🚀 NVIDIA CUDA GPU detected — using GPU acceleration.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("🍎 Apple Silicon MPS detected — using Metal acceleration.")
        return torch.device("mps")
    else:
        logger.info("💻 Using CPU — model inference will be slower but functional.")
        return torch.device("cpu")


# ----------------------------------------------------------------------
# Central Settings Class for EduMIND
# ----------------------------------------------------------------------
class EduMINDSettings:
    """Singleton configuration manager for the EduMIND application.

    Reads configuration values from environment variables and sets safe CPU-friendly
    defaults where variables are missing.
    """

    def __init__(self):
        # --- Computation Device ---
        self.DEVICE: torch.device = _detect_device(
            os.getenv("EDUMIND_DEVICE", "auto")
        )

        # --- ASR (Speech Recognition) Model ---
        self.WHISPER_MODEL: str = os.getenv("EDUMIND_WHISPER_MODEL", "tiny")

        # --- Embedding Model (for RAG) ---
        self.EMBEDDING_MODEL: str = os.getenv(
            "EDUMIND_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- Machine Translation Model ---
        self.TRANSLATION_MODEL: str = os.getenv("EDUMIND_TRANSLATION_MODEL", "none")

        # --- Qdrant Vector Database ---
        self.QDRANT_MODE: str = os.getenv("QDRANT_MODE", "memory")
        self.QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
        self.QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
        self.QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
        self.QDRANT_COLLECTION_NAME: str = os.getenv(
            "QDRANT_COLLECTION_NAME", "edumind_documents"
        )

        # --- Google Gemini API (optional, for RAG answer generation) ---
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

        # --- Project Paths ---
        self.PROJ_ROOT: Path = PROJ_ROOT
        self.DATA_DIR: Path = PROJ_ROOT / "data"
        self.MODELS_DIR: Path = PROJ_ROOT / "models"

        # --- Teencode / Abbreviations Mapping (Code-Switching) ---
        # Maps common abbreviations used in bilingual Vietnamese-English lectures
        self.TEENCODE_MAP: dict[str, str] = {
            # Vietnamese slang / abbreviations
            "ko": "không",
            "k": "không",
            "dc": "được",
            "đc": "được",
            "j": "gì",
            "z": "gì",
            "ns": "nói sao",
            "bđ": "bắt đầu",
            "bt": "bình thường",
            "ck": "chồng",
            "vk": "vợ",
            "trc": "trước",
            "nc": "nói chuyện",
            "r": "rồi",
            "đag": "đang",
            "mk": "mình",
            "mn": "mọi người",
            "cx": "cũng",
            "ms": "mới",
            "ib": "inbox",
            "rep": "reply",
            "fb": "Facebook",
            "acc": "account",
            # Technical abbreviations (Tech/Academic slang)
            "loss fn": "loss function",
            "act fn": "activation function",
            "dl": "deadline",
            "lr": "learning rate",
            "bs": "batch size",
            "grad": "gradient",
            "backprop": "backpropagation",
            "conv": "convolution",
            "fc": "fully connected",
            "bn": "batch normalization",
            "ln": "layer normalization",
            "attn": "attention",
            "tfm": "transformer",
            "enc": "encoder",
            "dec": "decoder",
            "emb": "embedding",
            "vocab": "vocabulary",
            "eval": "evaluation",
            "cfg": "configuration",
            "ckpt": "checkpoint",
            "ep": "epoch",
            "iter": "iteration",
            # Academic/AI domain acronyms
            "ML": "Machine Learning",
            "DL": "Deep Learning",
            "NLP": "Natural Language Processing",
            "CV": "Computer Vision",
            "RL": "Reinforcement Learning",
            "GAN": "Generative Adversarial Network",
            "VAE": "Variational Autoencoder",
            "RNN": "Recurrent Neural Network",
            "CNN": "Convolutional Neural Network",
            "LSTM": "Long Short-Term Memory",
            "GRU": "Gated Recurrent Unit",
            "BERT": "Bidirectional Encoder Representations from Transformers",
            "GPT": "Generative Pre-trained Transformer",
            "LLM": "Large Language Model",
            "RAG": "Retrieval-Augmented Generation",
            "ASR": "Automatic Speech Recognition",
            "TTS": "Text-to-Speech",
            "OCR": "Optical Character Recognition",
            "API": "Application Programming Interface",
        }

    def summary(self) -> dict:
        """Returns a summarized dictionary of settings (for UI display).

        Returns:
            A dictionary containing key configuration flags and statuses.
        """
        return {
            "device": str(self.DEVICE),
            "whisper_model": self.WHISPER_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "translation_model": self.TRANSLATION_MODEL,
            "qdrant_mode": self.QDRANT_MODE,
            "has_google_api": bool(self.GOOGLE_API_KEY),
        }


# ----------------------------------------------------------------------
# Globally initialized Settings singleton
# ----------------------------------------------------------------------
settings = EduMINDSettings()
logger.info(f"⚙️ EduMIND Config: {settings.summary()}")
