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
    "tôi",
    "mình",
    "tớ",
    "ta",
    "chúng",
    "các",
    "bạn",
    "anh",
    "chị",
    "em",
    "ông",
    "bà",
    "cô",
    "chú",
    "họ",
    "nó",
    "là",
    "có",
    "được",
    "và",
    "của",
    "cho",
    "với",
    "trong",
    "từ",
    "để",
    "này",
    "đỏ",
    "đó",
    "kia",
    "ấy",
    "nào",
    "gì",
    "sao",
    "nên",
    "vì",
    "nhưng",
    "nếu",
    "thì",
    "mà",
    "hay",
    "hoặc",
    "cũng",
    "đã",
    "đang",
    "sẽ",
    "rồi",
    "chưa",
    "không",
    "chẳng",
    "rất",
    "lắm",
    "quá",
    "hơn",
    "làm",
    "đi",
    "đến",
    "về",
    "biết",
    "hiểu",
    "nói",
    "xem",
    "học",
    "viết",
    "đọc",
    "nghe",
    "thấy",
    "cần",
    "muốn",
    "phải",
    "dùng",
    "bài",
    "môn",
    "lớp",
    "điểm",
    "thi",
    "hôm",
    "nay",
    "tuần",
    "tháng",
    "năm",
    "giờ",
    "phút",
    "nhớ",
    "ôn",
    "tập",
    "nộp",
    "trước",
    "sau",
    "nhé",
    "nha",
    "ạ",
    "nhỉ",
    "ơi",
    "hả",
    "chứ",
}


# Set of common English words (used to immediately filter out common English stopwords)
_ENGLISH_COMMON_WORDS: set[str] = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "to",
    "in",
    "is",
    "you",
    "that",
    "it",
    "he",
    "was",
    "for",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "i",
    "at",
    "be",
    "this",
    "have",
    "from",
    "or",
    "one",
    "had",
    "by",
    "word",
    "but",
    "not",
    "what",
    "all",
    "were",
    "we",
    "when",
    "your",
    "can",
    "said",
    "there",
    "use",
    "an",
    "each",
    "which",
    "she",
    "do",
    "how",
    "their",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "him",
    "into",
    "time",
    "has",
    "look",
    "two",
    "more",
    "write",
    "go",
    "see",
    "number",
    "no",
    "way",
    "could",
    "people",
    "my",
    "than",
    "first",
    "water",
    "been",
    "call",
    "who",
    "oil",
    "its",
    "now",
    "find",
    "input",
    "output",
    "dataset",
    "system",
    "model",
}


def _is_vietnamese_syllable(word: str) -> bool:
    """Checks if a word conforms to standard Vietnamese syllable structure rules.

    Helps identify unaccented Vietnamese words versus English terminology.
    """
    clean = word.lower().strip()
    if not clean:
        return False

    # Vietnamese syllables do not use w, f, j, z
    if any(c in clean for c in "wfjz"):
        return False

    # Maximum length of a Vietnamese syllable is 7 characters
    if len(clean) > 7:
        return False

    # English words are often multisyllabic, having multiple non-contiguous vowel groups.
    # A single Vietnamese syllable can only have at most 1 contiguous vowel cluster.
    vowels = set("aeiouy")
    vowel_clusters = 0
    in_vowel = False
    for char in clean:
        if char in vowels:
            if not in_vowel:
                vowel_clusters += 1
                in_vowel = True
        else:
            in_vowel = False

    if vowel_clusters > 1:
        return False

    # Consonant clusters at the start that are common in English but impossible in Vietnamese
    invalid_starts = [
        "bl",
        "br",
        "cl",
        "cr",
        "dr",
        "fl",
        "fr",
        "gl",
        "gr",
        "pl",
        "pr",
        "sc",
        "sk",
        "sl",
        "sm",
        "sn",
        "sp",
        "st",
        "str",
        "sw",
        "tw",
    ]
    for start in invalid_starts:
        if clean.startswith(start):
            return False

    # Final consonants / clusters check
    if len(clean) > 1:
        end_char = clean[-1]
        penultimate = clean[-2]

        # In Vietnamese, 'g' at the end is only valid as 'ng'
        if end_char == "g" and penultimate != "n":
            return False
        # In Vietnamese, 'h' at the end is only valid as 'nh' or 'ch'
        if end_char == "h" and penultimate not in ("n", "c"):
            return False

        # Invalid final single letters
        if end_char in ("d", "s", "l", "r", "b", "f", "k", "v", "x", "z"):
            return False

        # Common English ending clusters that are impossible in Vietnamese
        invalid_end_clusters = [
            "ct",
            "nd",
            "nt",
            "mp",
            "rt",
            "st",
            "ld",
            "lf",
            "lk",
            "lp",
            "lt",
            "mb",
            "pt",
            "ft",
            "th",
            "rd",
            "rk",
            "rn",
            "ss",
            "sh",
        ]
        for end in invalid_end_clusters:
            if clean.endswith(end):
                return False

    # Double vowels common in English but impossible in Vietnamese
    invalid_double_vowels = ["ee", "oo", "ea", "ou", "ae"]
    for dv in invalid_double_vowels:
        if dv in clean:
            return False

    return True


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
            2. Common English words lookup → "en"
            3. Vietnamese diacritics match → "vi"
            4. Common Vietnamese word lookup (unaccented) → "vi"
            5. Vietnamese syllable rules validation → "vi"
            6. Default fallback (Latin characters) → "en"

        Args:
            token: String token to examine.

        Returns:
            Language label string ("vi", "en", or "other").
        """
        clean = token.strip().lower()

        # Skip empty strings or non-alphabetical punctuation/numbers
        if not clean or not any(c.isalpha() for c in clean):
            return "other"

        # Fast path check for common English words / stopwords
        if clean in _ENGLISH_COMMON_WORDS:
            return "en"

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

        # Validate against Vietnamese syllable rules (heuristic check)
        if _is_vietnamese_syllable(clean):
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
        return [TokenLabel(token=t, language=self._detect_token_language(t)) for t in tokens]

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
