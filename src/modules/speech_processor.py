"""
EduMIND — Bộ Xử Lý Giọng Nói Song Ngữ (Code-Switched ASR)
============================================================
Module nhận dạng giọng nói sử dụng Whisper với khả năng:
  1. Phiên âm (transcribe) file audio đa định dạng (WAV, MP3, FLAC, M4A)
  2. Hậu xử lý (post-processing) sửa lỗi teencode & viết tắt
  3. Fallback mock mode khi không tải được mô hình Whisper

Kiến trúc:
    CodeSwitchedASR
    ├── transcribe(audio_path) → TranscriptResult
    ├── post_process(text) → str
    └── _mock_transcribe() → TranscriptResult  (fallback)

Sử dụng:
    asr = CodeSwitchedASR()
    result = asr.transcribe("lecture.wav")
    corrected = asr.post_process(result.text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# Data Classes cho kết quả phiên âm
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TranscriptSegment:
    """Một đoạn phiên âm có mốc thời gian."""
    start: float       # Thời điểm bắt đầu (giây)
    end: float         # Thời điểm kết thúc (giây)
    text: str           # Nội dung văn bản của đoạn


@dataclass
class TranscriptResult:
    """Kết quả phiên âm hoàn chỉnh từ một file audio."""
    text: str                                       # Toàn bộ văn bản phiên âm
    segments: list[TranscriptSegment] = field(default_factory=list)  # Danh sách các đoạn
    language: str = "vi"                           # Ngôn ngữ được phát hiện
    is_mock: bool = False                          # True nếu dùng dữ liệu mock


# ──────────────────────────────────────────────────────────────────────
# Lớp chính: CodeSwitchedASR
# ──────────────────────────────────────────────────────────────────────
class CodeSwitchedASR:
    """
    Bộ nhận dạng giọng nói song ngữ (Code-Switched ASR) sử dụng OpenAI Whisper.

    Tính năng:
        - Tải mô hình Whisper (mặc định: tiny cho CPU nhanh)
        - Phiên âm audio với timestamps cho từng đoạn
        - Hậu xử lý sửa teencode/viết tắt chuyên ngành
        - Mock mode tự động khi mô hình không khả dụng

    Args:
        model_name: Tên mô hình Whisper ("tiny", "base", "small", "medium", "large")
        teencode_map: Từ điển ánh xạ teencode → từ chuẩn (mặc định từ config)
    """

    def __init__(
        self,
        model_name: str = "tiny",
        teencode_map: dict[str, str] | None = None,
    ):
        # Lưu tên mô hình và trạng thái
        self.model_name = model_name
        self._model = None
        self._is_mock_mode = False

        # Tải từ điển teencode từ config nếu không được cung cấp
        if teencode_map is not None:
            self.teencode_map = teencode_map
        else:
            from src.config import settings
            self.teencode_map = settings.TEENCODE_MAP

        # Thử tải mô hình Whisper
        self._load_model()

    def _load_model(self) -> None:
        """
        Tải mô hình Whisper vào bộ nhớ.
        Nếu thất bại (thiếu thư viện, lỗi mạng, ...), tự động bật mock mode.
        """
        try:
            import whisper
            logger.info(f"🎤 Đang tải mô hình Whisper-{self.model_name}...")
            self._model = whisper.load_model(self.model_name)
            logger.success(f"✅ Đã tải Whisper-{self.model_name} thành công!")
        except ImportError:
            logger.warning(
                "⚠️ Thư viện 'openai-whisper' chưa được cài đặt. "
                "Chuyển sang mock mode."
            )
            self._is_mock_mode = True
        except Exception as e:
            logger.warning(
                f"⚠️ Không thể tải mô hình Whisper-{self.model_name}: {e}. "
                "Chuyển sang mock mode."
            )
            self._is_mock_mode = True

    # ──────────────────────────────────────────────────────────────────
    # Phương thức chính: Phiên âm
    # ──────────────────────────────────────────────────────────────────
    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """
        Phiên âm file audio thành văn bản với timestamps.

        Args:
            audio_path: Đường dẫn tới file audio (WAV, MP3, FLAC, M4A).

        Returns:
            TranscriptResult chứa văn bản, đoạn phiên âm, và ngôn ngữ.

        Raises:
            FileNotFoundError: Nếu file audio không tồn tại.
        """
        audio_path = Path(audio_path)

        # Kiểm tra file tồn tại
        if not audio_path.exists():
            raise FileNotFoundError(f"File audio không tồn tại: {audio_path}")

        # Nếu mock mode → trả kết quả giả lập
        if self._is_mock_mode or self._model is None:
            logger.info("🎭 Sử dụng mock transcription (mô hình chưa tải).")
            return self._mock_transcribe(source_file=str(audio_path))

        # Phiên âm bằng Whisper thật
        logger.info(f"🎤 Đang phiên âm: {audio_path.name}...")
        try:
            result = self._model.transcribe(
                str(audio_path),
                language=None,  # Tự phát hiện ngôn ngữ
                verbose=False,
            )

            # Chuyển đổi segments sang dataclass
            segments = [
                TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                )
                for seg in result.get("segments", [])
            ]

            detected_lang = result.get("language", "vi")
            full_text = result.get("text", "").strip()

            logger.success(
                f"✅ Phiên âm hoàn tất! Ngôn ngữ: {detected_lang}, "
                f"Số đoạn: {len(segments)}"
            )

            return TranscriptResult(
                text=full_text,
                segments=segments,
                language=detected_lang,
                is_mock=False,
            )

        except Exception as e:
            logger.error(f"❌ Lỗi phiên âm: {e}. Chuyển sang mock mode.")
            return self._mock_transcribe(source_file=str(audio_path))

    # ──────────────────────────────────────────────────────────────────
    # Hậu xử lý: Sửa lỗi teencode / viết tắt
    # ──────────────────────────────────────────────────────────────────
    def post_process(self, text: str) -> str:
        """
        Hậu xử lý văn bản phiên âm: sửa teencode và viết tắt chuyên ngành.

        Thuật toán:
            1. Sắp xếp từ điển theo độ dài giảm dần (ưu tiên cụm từ dài hơn)
            2. Thay thế từng mục bằng regex word-boundary (\b)
            3. Chuẩn hóa khoảng trắng thừa

        Args:
            text: Văn bản thô từ ASR cần sửa lỗi.

        Returns:
            Văn bản đã được sửa teencode/viết tắt.
        """
        if not text or not text.strip():
            return text

        corrected = text

        # Sắp xếp theo độ dài giảm dần để tránh thay thế sai
        # VD: "loss fn" phải được xử lý trước "fn"
        sorted_terms = sorted(
            self.teencode_map.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for abbr, full_form in sorted_terms:
            # Tạo regex pattern với word boundary
            # re.escape để xử lý ký tự đặc biệt trong abbreviation
            pattern = r"\b" + re.escape(abbr) + r"\b"
            corrected = re.sub(pattern, full_form, corrected, flags=re.IGNORECASE)

        # Chuẩn hóa khoảng trắng thừa
        corrected = re.sub(r"\s+", " ", corrected).strip()

        return corrected

    def get_corrections(self, original: str, corrected: str) -> list[dict]:
        """
        So sánh văn bản gốc và đã sửa, trả về danh sách các thay đổi.

        Args:
            original: Văn bản gốc từ ASR.
            corrected: Văn bản đã qua post_process.

        Returns:
            Danh sách dict {"original": ..., "corrected": ..., "position": ...}
        """
        changes = []
        orig_words = original.split()
        corr_words = corrected.split()

        # So sánh đơn giản theo từng từ
        max_len = max(len(orig_words), len(corr_words))
        for i in range(min(len(orig_words), max_len)):
            if i < len(corr_words) and orig_words[i].lower() != corr_words[i].lower():
                changes.append({
                    "original": orig_words[i],
                    "corrected": corr_words[i] if i < len(corr_words) else "",
                    "position": i,
                })

        return changes

    # ──────────────────────────────────────────────────────────────────
    # Mock Mode: Dữ liệu giả lập cho demo
    # ──────────────────────────────────────────────────────────────────
    def _mock_transcribe(self, source_file: str = "demo.wav") -> TranscriptResult:
        """
        Tạo kết quả phiên âm giả lập mô phỏng bài giảng song ngữ Vi-En.
        Sử dụng khi mô hình Whisper chưa được tải.

        Returns:
            TranscriptResult với dữ liệu mô phỏng bài giảng NLP thực tế.
        """
        logger.info("🎭 Generating mock transcription — bài giảng NLP song ngữ...")

        # Mô phỏng một đoạn bài giảng code-switched Vi-En thực tế
        mock_segments = [
            TranscriptSegment(
                start=0.0, end=5.2,
                text="Xin chào các bạn, hôm nay chúng ta sẽ discuss về NLP"
            ),
            TranscriptSegment(
                start=5.2, end=12.8,
                text="Đầu tiên mình sẽ explain về loss fn trong deep learning model"
            ),
            TranscriptSegment(
                start=12.8, end=20.1,
                text="Các bạn cần submit bài trc dl nhé, ko là bị trừ điểm"
            ),
            TranscriptSegment(
                start=20.1, end=28.5,
                text="Bây giờ mình sẽ bđ phần attention mechanism trong transformer"
            ),
            TranscriptSegment(
                start=28.5, end=35.0,
                text="Cái lr nên set khoảng 2e-5 cho fine-tuning BERT"
            ),
            TranscriptSegment(
                start=35.0, end=42.3,
                text="Mn nhớ review lại backprop và gradient descent trước buổi sau"
            ),
        ]

        full_text = " ".join(seg.text for seg in mock_segments)

        return TranscriptResult(
            text=full_text,
            segments=mock_segments,
            language="vi",
            is_mock=True,
        )

    @property
    def is_ready(self) -> bool:
        """Kiểm tra mô hình đã sẵn sàng chưa (True nếu Whisper đã tải)."""
        return self._model is not None and not self._is_mock_mode

    @property
    def is_mock(self) -> bool:
        """Kiểm tra có đang ở mock mode không."""
        return self._is_mock_mode
