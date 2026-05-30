"""Rule-based Translation Provider.

Performs translation of code-mixed Vietnamese-English text utilizing direct dictionary mappings.
"""

from __future__ import annotations

import re

from edumind.core.logging import get_logger
from edumind.services.translation.base import TranslationProvider

logger = get_logger(__name__)

_VI_TO_EN_DICT: dict[str, str] = {
    "hôm nay": "today",
    "mình": "I",
    "chúng ta": "we",
    "sẽ": "will",
    "về": "about",
    "trong": "in",
    "của": "of",
    "cho": "for",
    "với": "with",
    "và": "and",
    "là": "is",
    "có": "have",
    "được": "can",
    "cần": "need",
    "phải": "must",
    "nhớ": "remember",
    "ôn": "review",
    "tập": "practice",
    "bài": "lesson",
    "trước": "before",
    "sau": "after",
    "buổi": "session",
    "tiếp": "next",
    "các bạn": "you all",
    "bắt đầu": "begin",
    "đầu tiên": "first",
    "bây giờ": "now",
    "khoảng": "about",
    "nên": "should",
    "nhé": "",
    "nha": "",
    "ạ": "",
    "cái": "the",
    "này": "this",
    "đó": "that",
}

_EN_TO_VI_DICT: dict[str, str] = {
    "discuss": "thảo luận",
    "explain": "giải thích",
    "submit": "nộp",
    "review": "ôn tập",
    "model": "mô hình",
    "function": "hàm",
    "loss": "hàm mất mát",
    "deep": "sâu",
    "learning": "học",
    "attention": "cơ chế chú ý",
    "mechanism": "cơ chế",
    "fine-tuning": "tinh chỉnh",
    "set": "đặt",
    "training": "huấn luyện",
    "dataset": "tập dữ liệu",
    "token": "đơn vị từ vựng",
    "embedding": "biểu diễn nhúng",
    "layer": "tầng",
    "output": "đầu ra",
    "input": "đầu vào",
    "batch": "lô",
    "epoch": "vòng lặp",
    "gradient": "đạo hàm",
    "weight": "trọng số",
    "bias": "độ lệch",
    "accuracy": "độ chính xác",
    "precision": "độ chính xác dương",
    "recall": "độ triệu hồi",
    "score": "điểm số",
}


class RuleBasedTranslationProvider(TranslationProvider):
    """Provides dictionary-based, fast token-mapping translation without neural models."""

    def translate_to_english(self, text: str) -> str:
        """Maps Vietnamese words to English equivalents using pre-defined mapping.

        Replaces longer phrases first to preserve nested mappings.
        """
        if not text or not text.strip():
            return ""

        result = text
        sorted_phrases = sorted(
            _VI_TO_EN_DICT.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for vi_phrase, en_phrase in sorted_phrases:
            pattern = r"\b" + re.escape(vi_phrase) + r"\b"
            result = re.sub(pattern, en_phrase, result, flags=re.IGNORECASE)

        result = re.sub(r"\s+", " ", result).strip()

        if result:
            result = result[0].upper() + result[1:]

        return result

    def translate_to_vietnamese(self, text: str) -> str:
        """Maps English words to Vietnamese equivalents using pre-defined mapping."""
        if not text or not text.strip():
            return ""

        result = text
        sorted_words = sorted(
            _EN_TO_VI_DICT.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for en_word, vi_word in sorted_words:
            pattern = r"\b" + re.escape(en_word) + r"\b"
            result = re.sub(pattern, vi_word, result, flags=re.IGNORECASE)

        result = re.sub(r"\s+", " ", result).strip()
        return result
