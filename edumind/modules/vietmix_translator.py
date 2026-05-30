"""EduMIND — VietMix Bilingual Machine Translator (Vi-En Code-Mixed Translation)

This module handles machine translation of code-mixed (Vietnamese-English) text.
Features:
  1. Computes the Code-Mixing Index (CMI) to evaluate the linguistic mixing level.
  2. Translates code-mixed text into standard English.
  3. Translates code-mixed text into standard Vietnamese.
  4. Provides token-level language identification.

CMI Formula:
    CMI = (N - max(w_L)) / N
    Where:
        N = total tokens (excluding symbols and punctuation)
        w_L = token count of the dominant language
    CMI = 0.0 → fully monolingual
    CMI → 1.0 → highly code-mixed / language-switched
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger


# ----------------------------------------------------------------------
# Set of common Vietnamese words (used for token-level language ID)
# ----------------------------------------------------------------------
_VIETNAMESE_COMMON_WORDS: set[str] = {
    # Pronouns
    "tôi", "mình", "tớ", "ta", "chúng", "các", "bạn", "anh", "chị", "em",
    "ông", "bà", "cô", "chú", "họ", "nó", "hắn",
    # Function words
    "là", "có", "được", "và", "của", "cho", "với", "trong", "từ", "để",
    "này", "đỏ", "đó", "kia", "ấy", "nào", "gì", "sao", "nên", "vì", "nhưng",
    "nếu", "thì", "mà", "hay", "hoặc", "cũng", "đã", "đang", "sẽ",
    "rồi", "chưa", "không", "chẳng", "rất", "lắm", "quá", "hơn",
    # Common verbs
    "làm", "đi", "đến", "về", "biết", "hiểu", "nói", "xem", "học",
    "viết", "đọc", "nghe", "thấy", "cần", "muốn", "phải", "dùng",
    "chạy", "lấy", "gọi", "hỏi", "trả", "lời", "bắt", "đầu",
    # Common academic / classroom terms
    "bài", "môn", "lớp", "điểm", "thi", "kiểm", "tra", "hôm", "nay",
    "mai", "tuần", "tháng", "năm", "giờ", "phút", "thời", "gian",
    "nhớ", "ôn", "tập", "nộp", "trước", "sau", "trên", "dưới",
    "nhé", "nha", "ạ", "nhỉ", "ơi", "hả", "chứ",
}


# ----------------------------------------------------------------------
# Data Classes for CMI and Translation results
# ----------------------------------------------------------------------
@dataclass
class TokenLabel:
    """Represents a language label mapped to a specific token.

    Attributes:
        token: The original token string.
        language: Identified language code ("vi", "en", or "other").
        confidence: Confidence score of the classification (0.0 to 1.0).
    """
    token: str
    language: str
    confidence: float = 1.0


@dataclass
class CMIResult:
    """Stores computed Code-Mixing Index (CMI) metrics.

    Attributes:
        score: The normalized CMI value (0.0 to 1.0).
        total_tokens: Total tokens analyzed (N).
        vi_count: Count of Vietnamese tokens.
        en_count: Count of English tokens.
        other_count: Count of non-alphabetic/punctuation tokens.
        dominant_language: Identified dominant language code ("vi" or "en").
        token_labels: Token-by-token language label mapping.
    """
    score: float
    total_tokens: int
    vi_count: int
    en_count: int
    other_count: int = 0
    dominant_language: str = "vi"
    token_labels: list[TokenLabel] = field(default_factory=list)


# ----------------------------------------------------------------------
# Fallback Dictionary Mapping for Rule-based Translation
# ----------------------------------------------------------------------
_VI_TO_EN_DICT: dict[str, str] = {
    "hôm nay": "today", "mình": "I", "chúng ta": "we", "sẽ": "will",
    "về": "about", "trong": "in", "của": "of", "cho": "for", "với": "with",
    "và": "and", "là": "is", "có": "have", "được": "can", "cần": "need",
    "phải": "must", "nhớ": "remember", "ôn": "review", "tập": "practice",
    "bài": "lesson", "trước": "before", "sau": "after", "buổi": "session",
    "tiếp": "next", "các bạn": "you all", "bắt đầu": "begin",
    "đầu tiên": "first", "bây giờ": "now", "khoảng": "about",
    "nên": "should", "nhé": "", "nha": "", "ạ": "",
    "cái": "the", "này": "this", "đó": "that",
}

_EN_TO_VI_DICT: dict[str, str] = {
    "discuss": "thảo luận", "explain": "giải thích", "submit": "nộp",
    "review": "ôn tập", "model": "mô hình", "function": "hàm",
    "loss": "hàm mất mát", "deep": "sâu", "learning": "học",
    "attention": "cơ chế chú ý", "mechanism": "cơ chế",
    "fine-tuning": "tinh chỉnh", "set": "đặt", "training": "huấn luyện",
    "dataset": "tập dữ liệu", "token": "đơn vị từ vựng",
    "embedding": "biểu diễn nhúng", "layer": "tầng", "output": "đầu ra",
    "input": "đầu vào", "batch": "lô", "epoch": "vòng lặp",
    "gradient": "đạo hàm", "weight": "trọng số", "bias": "độ lệch",
    "accuracy": "độ chính xác", "precision": "độ chính xác dương",
    "recall": "độ triệu hồi", "score": "điểm số",
}


# ----------------------------------------------------------------------
# Main Class: VietMixTranslator
# ----------------------------------------------------------------------
class VietMixTranslator:
    """Translates code-mixed Vietnamese-English text and calculates mixing metrics.

    Uses deep learning sequence-to-sequence models from HuggingFace when provided,
    otherwise falls back gracefully to a dictionary-based translation system.
    """

    def __init__(self, model_name: str = "none"):
        """Initializes the VietMixTranslator.

        Args:
            model_name: HuggingFace translation model name. Set to "none" to bypass
                deep learning models and use rule-based fallback exclusively.
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._use_model = False

        # Load machine translation model if requested
        if model_name and model_name.lower() != "none":
            self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Loads a seq2seq translation model from Hugging Face.

        Falls back automatically to rule-based translations if loading fails.
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"🔄 Loading translation model: {model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._use_model = True
            logger.success(f"✅ Loaded translation model: {model_name}")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not load translation model '{model_name}': {e}. "
                "Defaulting to rule-based translation mode."
            )
            self._use_model = False

    # ------------------------------------------------------------------
    # Token-Level Language Identification
    # ------------------------------------------------------------------
    def _detect_token_language(self, token: str) -> str:
        """Determines the language of a single token.

        Algorithm:
            1. Non-alphabetic/empty tokens → "other"
            2. Vietnamese diacritics match → "vi"
            3. Common Vietnamese word lookup (unaccented) → "vi"
            4. Default fallback (Latin characters) → "en"

        Args:
            token: String token to examine.

        Returns:
            Language label string ("vi", "en", or "other").
        """
        clean = token.strip().lower()

        # Skip empty strings or non-alphabetical punctuation/numbers
        if not clean or not any(c.isalpha() for c in clean):
            return "other"

        # Check for unique Vietnamese unicode diacritics
        vietnamese_diacritics_pattern = re.compile(
            r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
            r"ùúủũụưứừửữựỳýỷỹỵđ]",
            re.IGNORECASE,
        )
        if vietnamese_diacritics_pattern.search(clean):
            return "vi"

        # Check unaccented common Vietnamese dictionary list
        if clean in _VIETNAMESE_COMMON_WORDS:
            return "vi"

        # Default to English
        return "en"

    def _label_tokens(self, text: str) -> list[TokenLabel]:
        """Classifies all tokens in a sentence.

        Args:
            text: Sentence to tokenize and classify.

        Returns:
            A list of TokenLabel objects mapping text tokens to languages.
        """
        tokens = text.split()
        return [
            TokenLabel(token=t, language=self._detect_token_language(t))
            for t in tokens
        ]

    # ------------------------------------------------------------------
    # Code-Mixing Index (CMI) Computation
    # ------------------------------------------------------------------
    def calculate_cmi(self, text: str) -> CMIResult:
        """Computes the Code-Mixing Index (CMI) for a sentence.

        Formula:
            CMI = (N - max(w_Vi, w_En)) / N
            Where N = vi_count + en_count (excluding punctuation/numbers).

        Args:
            text: String sentence to evaluate.

        Returns:
            CMIResult storing mixing score, token-level classifications, and counts.
        """
        if not text or not text.strip():
            return CMIResult(
                score=0.0, total_tokens=0,
                vi_count=0, en_count=0, other_count=0,
                dominant_language="unknown", token_labels=[],
            )

        # Label tokens
        token_labels = self._label_tokens(text)

        # Compute linguistic counts
        vi_count = sum(1 for tl in token_labels if tl.language == "vi")
        en_count = sum(1 for tl in token_labels if tl.language == "en")
        other_count = sum(1 for tl in token_labels if tl.language == "other")

        # N is the total count of alphabetic (language) tokens
        n = vi_count + en_count

        # Determine the dominant language
        if vi_count >= en_count:
            dominant = "vi"
            max_w = vi_count
        else:
            dominant = "en"
            max_w = en_count

        # Calculate standard CMI
        if n == 0:
            cmi_score = 0.0
        else:
            cmi_score = (n - max_w) / n

        return CMIResult(
            score=round(cmi_score, 4),
            total_tokens=len(token_labels),
            vi_count=vi_count,
            en_count=en_count,
            other_count=other_count,
            dominant_language=dominant,
            token_labels=token_labels,
        )

    # ------------------------------------------------------------------
    # Translation Interfaces
    # ------------------------------------------------------------------
    def translate_to_english(self, text: str) -> str:
        """Translates code-mixed Vietnamese-English text into clean English.

        Args:
            text: Raw bilingual sentence to translate.

        Returns:
            Standardized English sentence.
        """
        if self._use_model and self._model is not None:
            return self._model_translate(text, target="en")
        return self._rule_based_translate_to_english(text)

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates code-mixed Vietnamese-English text into clean Vietnamese.

        Args:
            text: Raw bilingual sentence to translate.

        Returns:
            Standardized Vietnamese sentence.
        """
        if self._use_model and self._model is not None:
            return self._model_translate(text, target="vi")
        return self._rule_based_translate_to_vietnamese(text)

    def _model_translate(self, text: str, target: str = "en") -> str:
        """Executes translation using the HuggingFace Seq2Seq model.

        Gracefully defaults to rule-based fallback if execution fails.
        """
        try:
            prefix = f"translate to {target}: "
            inputs = self._tokenizer(
                prefix + text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            outputs = self._model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(
                f"⚠️ Deep learning model translation failed: {e}. "
                "Defaulting to rule-based fallback translation."
            )
            if target == "en":
                return self._rule_based_translate_to_english(text)
            return self._rule_based_translate_to_vietnamese(text)

    def _rule_based_translate_to_english(self, text: str) -> str:
        """Maps Vietnamese words to English equivalents using pre-defined mapping.

        Replaces longer phrases first to preserve nested mappings.

        Args:
            text: Code-mixed sentence.

        Returns:
            Rough translated English string.
        """
        result = text

        # Sort mapping by key length in descending order to prioritize multi-word mappings
        sorted_phrases = sorted(
            _VI_TO_EN_DICT.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for vi_phrase, en_phrase in sorted_phrases:
            pattern = r"\b" + re.escape(vi_phrase) + r"\b"
            result = re.sub(pattern, en_phrase, result, flags=re.IGNORECASE)

        # Remove duplicate spaces
        result = re.sub(r"\s+", " ", result).strip()

        # Sentence casing
        if result:
            result = result[0].upper() + result[1:]

        return result

    def _rule_based_translate_to_vietnamese(self, text: str) -> str:
        """Maps English words to Vietnamese equivalents using pre-defined mapping.

        Args:
            text: Code-mixed sentence.

        Returns:
            Rough translated Vietnamese string.
        """
        result = text

        # Replace English terms with Vietnamese equivalents
        sorted_words = sorted(
            _EN_TO_VI_DICT.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for en_word, vi_word in sorted_words:
            pattern = r"\b" + re.escape(en_word) + r"\b"
            result = re.sub(pattern, vi_word, result, flags=re.IGNORECASE)

        # Standardize whitespace
        result = re.sub(r"\s+", " ", result).strip()

        return result

    @property
    def is_model_loaded(self) -> bool:
        """Checks if a neural sequence-to-sequence translation model is active."""
        return self._use_model and self._model is not None

    @property
    def mode(self) -> str:
        """Returns current operational mode: 'model' or 'rule-based'."""
        return "model" if self.is_model_loaded else "rule-based"
