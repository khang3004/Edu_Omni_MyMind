"""
EduMIND — Cấu hình trung tâm (Central Configuration)
=====================================================
Tải cấu hình từ biến môi trường (.env) với giá trị mặc định thông minh.
Tự động phát hiện thiết bị tính toán (CUDA → MPS → CPU).

Sử dụng:
    from src.config import settings
    print(settings.DEVICE)  # torch.device
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import torch

# ──────────────────────────────────────────────────────────────────────
# Load biến môi trường từ file .env tại thư mục gốc dự án
# ──────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")


# ──────────────────────────────────────────────────────────────────────
# Hàm tiện ích phát hiện thiết bị tính toán tối ưu
# ──────────────────────────────────────────────────────────────────────
def _detect_device(preference: str = "auto") -> torch.device:
    """
    Phát hiện thiết bị tính toán tối ưu nhất có sẵn trên hệ thống.

    Args:
        preference: "auto" (tự động), "cuda", "mps", hoặc "cpu".

    Returns:
        torch.device phù hợp nhất.
    """
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        logger.info("🚀 Phát hiện GPU NVIDIA CUDA — sử dụng GPU acceleration.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("🍎 Phát hiện Apple Silicon MPS — sử dụng Metal acceleration.")
        return torch.device("mps")
    else:
        logger.info("💻 Sử dụng CPU — các mô hình sẽ chạy chậm hơn nhưng vẫn hoạt động.")
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────
# Lớp Settings chứa toàn bộ cấu hình EduMIND
# ──────────────────────────────────────────────────────────────────────
class EduMINDSettings:
    """
    Singleton chứa toàn bộ cấu hình cho EduMIND.
    Đọc từ biến môi trường với giá trị mặc định an toàn cho CPU.
    """

    def __init__(self):
        # --- Thiết bị tính toán ---
        self.DEVICE: torch.device = _detect_device(
            os.getenv("EDUMIND_DEVICE", "auto")
        )

        # --- Mô hình ASR (Nhận dạng giọng nói) ---
        self.WHISPER_MODEL: str = os.getenv("EDUMIND_WHISPER_MODEL", "tiny")

        # --- Mô hình Embedding (cho RAG) ---
        self.EMBEDDING_MODEL: str = os.getenv(
            "EDUMIND_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- Mô hình dịch thuật ---
        self.TRANSLATION_MODEL: str = os.getenv("EDUMIND_TRANSLATION_MODEL", "none")

        # --- Qdrant Vector Database ---
        self.QDRANT_MODE: str = os.getenv("QDRANT_MODE", "memory")
        self.QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
        self.QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
        self.QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
        self.QDRANT_COLLECTION_NAME: str = os.getenv(
            "QDRANT_COLLECTION_NAME", "edumind_documents"
        )

        # --- Google Gemini API (tùy chọn, cho RAG generation) ---
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

        # --- Đường dẫn dự án ---
        self.PROJ_ROOT: Path = PROJ_ROOT
        self.DATA_DIR: Path = PROJ_ROOT / "data"
        self.MODELS_DIR: Path = PROJ_ROOT / "models"

        # --- Từ điển sửa lỗi teencode / viết tắt (Code-Switch) ---
        # Ánh xạ các từ viết tắt phổ biến trong giảng dạy song ngữ Vi-En
        self.TEENCODE_MAP: dict[str, str] = {
            # Viết tắt tiếng Việt phổ biến
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
            # Viết tắt kỹ thuật (Tech/Academic slang)
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
            # Từ viết tắt chuyên ngành AI/ML
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
        """Trả về bản tóm tắt cấu hình dưới dạng dict (để hiển thị trên UI)."""
        return {
            "device": str(self.DEVICE),
            "whisper_model": self.WHISPER_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "translation_model": self.TRANSLATION_MODEL,
            "qdrant_mode": self.QDRANT_MODE,
            "has_google_api": bool(self.GOOGLE_API_KEY),
        }


# ──────────────────────────────────────────────────────────────────────
# Khởi tạo settings toàn cục (singleton)
# ──────────────────────────────────────────────────────────────────────
settings = EduMINDSettings()
logger.info(f"⚙️ EduMIND Config: {settings.summary()}")
