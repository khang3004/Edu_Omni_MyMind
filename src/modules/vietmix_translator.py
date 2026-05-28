"""
EduMIND — Bộ Dịch Thuật Song Ngữ VietMix (Vi-En Code-Mixed Translation)
========================================================================
Module xử lý dịch thuật văn bản pha trộn ngôn ngữ Việt-Anh với:
  1. Tính chỉ số Code-Mixing Index (CMI) đánh giá mức độ pha trộn
  2. Dịch code-mixed → tiếng Anh chuẩn
  3. Dịch code-mixed → tiếng Việt chuẩn
  4. Nhận diện ngôn ngữ ở cấp độ token (token-level language ID)

Công thức CMI:
    CMI = (N - max(w_L)) / N
    Trong đó:
        N = tổng số token
        w_L = số token thuộc ngôn ngữ chiếm đa số (dominant language)
    CMI = 0.0 → đơn ngữ hoàn toàn
    CMI → 1.0 → pha trộn ngôn ngữ cao

Sử dụng:
    translator = VietMixTranslator()
    cmi = translator.calculate_cmi("Hôm nay mình discuss về loss function")
    en_text = translator.translate_to_english("Hôm nay mình discuss về loss function")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# Tập từ vựng tiếng Việt phổ biến (dùng cho nhận diện ngôn ngữ)
# ──────────────────────────────────────────────────────────────────────
_VIETNAMESE_COMMON_WORDS: set[str] = {
    # Đại từ nhân xưng
    "tôi", "mình", "tớ", "ta", "chúng", "các", "bạn", "anh", "chị", "em",
    "ông", "bà", "cô", "chú", "họ", "nó", "hắn",
    # Từ chức năng (function words)
    "là", "có", "được", "và", "của", "cho", "với", "trong", "từ", "để",
    "này", "đó", "kia", "ấy", "nào", "gì", "sao", "nên", "vì", "nhưng",
    "nếu", "thì", "mà", "hay", "hoặc", "cũng", "đã", "đang", "sẽ",
    "rồi", "chưa", "không", "chẳng", "rất", "lắm", "quá", "hơn",
    # Động từ phổ biến
    "làm", "đi", "đến", "về", "biết", "hiểu", "nói", "xem", "học",
    "viết", "đọc", "nghe", "thấy", "cần", "muốn", "phải", "dùng",
    "chạy", "lấy", "gọi", "hỏi", "trả", "lời", "bắt", "đầu",
    # Danh từ / trạng từ phổ biến trong giảng dạy
    "bài", "môn", "lớp", "điểm", "thi", "kiểm", "tra", "hôm", "nay",
    "mai", "tuần", "tháng", "năm", "giờ", "phút", "thời", "gian",
    "nhớ", "ôn", "tập", "nộp", "trước", "sau", "trên", "dưới",
    "nhé", "nha", "ạ", "nhỉ", "ơi", "hả", "chứ",
}


# ──────────────────────────────────────────────────────────────────────
# Data Classes cho kết quả CMI và dịch thuật
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TokenLabel:
    """Nhãn ngôn ngữ cho một token."""
    token: str               # Token gốc
    language: str            # "vi", "en", hoặc "other" (số, ký tự đặc biệt)
    confidence: float = 1.0  # Độ tin cậy của nhận diện (0.0 → 1.0)


@dataclass
class CMIResult:
    """Kết quả tính Code-Mixing Index."""
    score: float                                    # Giá trị CMI (0.0 → 1.0)
    total_tokens: int                               # Tổng số token (N)
    vi_count: int                                   # Số token tiếng Việt
    en_count: int                                   # Số token tiếng Anh
    other_count: int = 0                            # Số token khác (số, ký hiệu)
    dominant_language: str = "vi"                    # Ngôn ngữ chiếm đa số
    token_labels: list[TokenLabel] = field(default_factory=list)  # Nhãn từng token


# ──────────────────────────────────────────────────────────────────────
# Từ điển dịch thuật đơn giản (rule-based fallback)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# Lớp chính: VietMixTranslator
# ──────────────────────────────────────────────────────────────────────
class VietMixTranslator:
    """
    Bộ dịch thuật văn bản pha trộn ngôn ngữ Việt-Anh (Code-Mixed Translation).

    Hỗ trợ:
        - Tính Code-Mixing Index (CMI) đo mức độ pha trộn ngôn ngữ
        - Nhận diện ngôn ngữ ở cấp token (Vietnamese vs English)
        - Dịch code-mixed → tiếng Anh / tiếng Việt chuẩn
        - Sử dụng mô hình HuggingFace (nếu có) hoặc rule-based fallback

    Args:
        model_name: Tên mô hình HuggingFace seq2seq ("none" = rule-based fallback)
    """

    def __init__(self, model_name: str = "none"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._use_model = False

        # Tải mô hình nếu được yêu cầu
        if model_name and model_name.lower() != "none":
            self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """
        Tải mô hình seq2seq từ HuggingFace để dịch thuật.
        Fallback sang rule-based nếu tải thất bại.
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"🔄 Đang tải mô hình dịch thuật: {model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._use_model = True
            logger.success(f"✅ Đã tải mô hình dịch thuật: {model_name}")
        except Exception as e:
            logger.warning(
                f"⚠️ Không thể tải mô hình {model_name}: {e}. "
                "Sử dụng rule-based translation."
            )
            self._use_model = False

    # ──────────────────────────────────────────────────────────────────
    # Nhận diện ngôn ngữ ở cấp token
    # ──────────────────────────────────────────────────────────────────
    def _detect_token_language(self, token: str) -> str:
        """
        Nhận diện ngôn ngữ của một token đơn lẻ.

        Thuật toán:
            1. Bỏ qua token rỗng, số, hoặc ký hiệu đặc biệt → "other"
            2. Kiểm tra dấu tiếng Việt (Unicode range) → "vi"
            3. Kiểm tra trong từ điển tiếng Việt phổ biến → "vi"
            4. Mặc định → "en" (giả sử từ Latin không dấu là tiếng Anh)

        Args:
            token: Một token (từ đơn) cần nhận diện.

        Returns:
            "vi", "en", hoặc "other".
        """
        clean = token.strip().lower()

        # Bỏ qua token rỗng hoặc chỉ chứa ký hiệu
        if not clean or not any(c.isalpha() for c in clean):
            return "other"

        # Kiểm tra dấu tiếng Việt (các ký tự Unicode đặc trưng)
        # Dải Unicode chứa dấu tiếng Việt: ắ, ằ, ẳ, ẵ, ặ, ơ, ư, đ, ...
        vietnamese_diacritics_pattern = re.compile(
            r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
            r"ùúủũụưứừửữựỳýỷỹỵđ]",
            re.IGNORECASE,
        )
        if vietnamese_diacritics_pattern.search(clean):
            return "vi"

        # Kiểm tra trong danh sách từ tiếng Việt phổ biến (không dấu)
        if clean in _VIETNAMESE_COMMON_WORDS:
            return "vi"

        # Mặc định: coi là tiếng Anh
        return "en"

    def _label_tokens(self, text: str) -> list[TokenLabel]:
        """
        Gán nhãn ngôn ngữ cho từng token trong câu.

        Args:
            text: Câu văn bản cần gán nhãn.

        Returns:
            Danh sách TokenLabel cho từng token.
        """
        tokens = text.split()
        return [
            TokenLabel(token=t, language=self._detect_token_language(t))
            for t in tokens
        ]

    # ──────────────────────────────────────────────────────────────────
    # Tính Code-Mixing Index (CMI)
    # ──────────────────────────────────────────────────────────────────
    def calculate_cmi(self, text: str) -> CMIResult:
        """
        Tính Code-Mixing Index (CMI) cho một câu.

        Công thức:
            CMI = (N - max(w_Vi, w_En)) / N

        Trong đó:
            - N: tổng số token có nghĩa (loại bỏ "other")
            - w_Vi: số token tiếng Việt
            - w_En: số token tiếng Anh
            - max(w_Vi, w_En): số token thuộc ngôn ngữ chiếm đa số

        Ý nghĩa:
            CMI = 0.0 → câu hoàn toàn đơn ngữ (chỉ Vi hoặc chỉ En)
            CMI = 0.5 → câu pha trộn 50/50
            CMI → 1.0 → câu pha trộn rất nhiều ngôn ngữ

        Args:
            text: Câu văn bản cần tính CMI.

        Returns:
            CMIResult với điểm CMI, số liệu thống kê, và nhãn token.
        """
        if not text or not text.strip():
            return CMIResult(
                score=0.0, total_tokens=0,
                vi_count=0, en_count=0, other_count=0,
                dominant_language="unknown", token_labels=[],
            )

        # Gán nhãn ngôn ngữ cho từng token
        token_labels = self._label_tokens(text)

        # Đếm số token theo ngôn ngữ
        vi_count = sum(1 for tl in token_labels if tl.language == "vi")
        en_count = sum(1 for tl in token_labels if tl.language == "en")
        other_count = sum(1 for tl in token_labels if tl.language == "other")

        # N = tổng token có nghĩa (bỏ "other")
        n = vi_count + en_count

        # Xác định ngôn ngữ chiếm đa số
        if vi_count >= en_count:
            dominant = "vi"
            max_w = vi_count
        else:
            dominant = "en"
            max_w = en_count

        # Tính CMI theo công thức
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

    # ──────────────────────────────────────────────────────────────────
    # Dịch thuật: Code-Mixed → English / Vietnamese chuẩn
    # ──────────────────────────────────────────────────────────────────
    def translate_to_english(self, text: str) -> str:
        """
        Dịch văn bản code-mixed Vi-En sang tiếng Anh chuẩn.

        Sử dụng mô hình HuggingFace nếu có, hoặc rule-based fallback.

        Args:
            text: Câu code-mixed cần dịch.

        Returns:
            Câu tiếng Anh chuẩn.
        """
        if self._use_model and self._model is not None:
            return self._model_translate(text, target="en")
        return self._rule_based_translate_to_english(text)

    def translate_to_vietnamese(self, text: str) -> str:
        """
        Dịch văn bản code-mixed Vi-En sang tiếng Việt chuẩn.

        Args:
            text: Câu code-mixed cần dịch.

        Returns:
            Câu tiếng Việt chuẩn.
        """
        if self._use_model and self._model is not None:
            return self._model_translate(text, target="vi")
        return self._rule_based_translate_to_vietnamese(text)

    def _model_translate(self, text: str, target: str = "en") -> str:
        """
        Dịch thuật sử dụng mô hình HuggingFace seq2seq.

        Args:
            text: Văn bản nguồn.
            target: Ngôn ngữ đích ("en" hoặc "vi").

        Returns:
            Văn bản đã dịch.
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
            logger.warning(f"⚠️ Model translation failed: {e}. Using rule-based fallback.")
            if target == "en":
                return self._rule_based_translate_to_english(text)
            return self._rule_based_translate_to_vietnamese(text)

    def _rule_based_translate_to_english(self, text: str) -> str:
        """
        Dịch rule-based: thay thế các từ tiếng Việt bằng tiếng Anh tương đương.

        Thuật toán:
            1. Thay thế cụm từ (multi-word) trước
            2. Thay thế từ đơn
            3. Giữ nguyên từ tiếng Anh đã có trong câu

        Args:
            text: Câu code-mixed.

        Returns:
            Câu tiếng Anh (gần đúng — rule-based).
        """
        result = text

        # Thay thế cụm từ (ưu tiên cụm dài hơn)
        sorted_phrases = sorted(
            _VI_TO_EN_DICT.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for vi_phrase, en_phrase in sorted_phrases:
            pattern = r"\b" + re.escape(vi_phrase) + r"\b"
            result = re.sub(pattern, en_phrase, result, flags=re.IGNORECASE)

        # Loại bỏ khoảng trắng thừa
        result = re.sub(r"\s+", " ", result).strip()

        # Viết hoa chữ cái đầu câu
        if result:
            result = result[0].upper() + result[1:]

        return result

    def _rule_based_translate_to_vietnamese(self, text: str) -> str:
        """
        Dịch rule-based: thay thế các từ tiếng Anh bằng tiếng Việt tương đương.

        Args:
            text: Câu code-mixed.

        Returns:
            Câu tiếng Việt (gần đúng — rule-based).
        """
        result = text

        # Thay thế từ tiếng Anh → tiếng Việt
        sorted_words = sorted(
            _EN_TO_VI_DICT.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for en_word, vi_word in sorted_words:
            pattern = r"\b" + re.escape(en_word) + r"\b"
            result = re.sub(pattern, vi_word, result, flags=re.IGNORECASE)

        # Loại bỏ khoảng trắng thừa
        result = re.sub(r"\s+", " ", result).strip()

        return result

    @property
    def is_model_loaded(self) -> bool:
        """Kiểm tra mô hình dịch thuật đã tải chưa."""
        return self._use_model and self._model is not None

    @property
    def mode(self) -> str:
        """Trả về chế độ dịch hiện tại: 'model' hoặc 'rule-based'."""
        return "model" if self.is_model_loaded else "rule-based"
