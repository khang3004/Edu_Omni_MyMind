"""EduMIND — VietMix Bilingual Machine Translator.

Analyzes code-mixing index (CMI) and translates code-mixed (Vietnamese-English)
text utilizing Strategy-based translation providers.
"""

from __future__ import annotations

import re

from edumind.core.logging import get_logger
from edumind.models.translation import CMIResult, TokenLabel
from edumind.services.translation.base import TranslationProvider
from edumind.services.translation.huggingface import HuggingFaceTranslationProvider

logger = get_logger(__name__)

# Set of common Vietnamese words (used for token-level language identification)
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


class VietMixTranslator:
    """Translates code-mixed Vietnamese-English text and calculates mixing metrics.

    Uses an injected TranslationProvider strategy to delegate actual translation
    operations, adhering to Single Responsibility and Open-Closed principles.
    """

    def __init__(self, translation_provider: TranslationProvider | None = None):
        """Initializes the VietMixTranslator.

        Args:
            translation_provider: Injected translation strategy. If None, resolves
                lazily via the DI Container.
        """
        self._provider = translation_provider

    def _get_provider(self) -> TranslationProvider:
        """Retrieves the active provider, resolving from container if not injected."""
        if self._provider is None:
            from edumind.core.container import get_translation_provider
            self._provider = get_translation_provider()
        return self._provider

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
                score=0.0,
                total_tokens=0,
                vi_count=0,
                en_count=0,
                other_count=0,
                dominant_language="unknown",
                token_labels=[],
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
        provider = self._get_provider()
        return provider.translate_to_english(text)

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates code-mixed Vietnamese-English text into clean Vietnamese.

        Args:
            text: Raw bilingual sentence to translate.

        Returns:
            Standardized Vietnamese sentence.
        """
        provider = self._get_provider()
        return provider.translate_to_vietnamese(text)

    @property
    def is_model_loaded(self) -> bool:
        """Checks if a neural sequence-to-sequence translation model is active."""
        provider = self._get_provider()
        if isinstance(provider, HuggingFaceTranslationProvider):
            return provider.is_model_loaded
        return False

    @property
    def mode(self) -> str:
        """Returns current operational mode: 'model' or 'rule-based'."""
        return "model" if self.is_model_loaded else "rule-based"
