"""EduMIND — Bilingual Speech Processor (Code-Switched ASR)

This module handles speech-to-text recognition using OpenAI's Whisper model,
specifically optimized for code-switched (bilingual Vietnamese-English) lecture speech.

Features:
  1. Transcribes multi-format audio files (WAV, MP3, FLAC, M4A) with timestamps.
  2. Post-processes text by correcting abbreviations, teencode, and domain slang.
  3. Automatically falls back to a smart mock mode if Whisper cannot be loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


# ----------------------------------------------------------------------
# Data Classes for Transcription Outputs
# ----------------------------------------------------------------------
@dataclass
class TranscriptSegment:
    """Represents a single segment of transcribed text with start and end timestamps.

    Attributes:
        start: Start time of the segment in seconds.
        end: End time of the segment in seconds.
        text: Transcribed text content for this segment.
    """
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    """Represents the complete transcription result from an audio file.

    Attributes:
        text: Unified text transcription of the entire audio.
        segments: Chronological list of transcribed segments.
        language: Automatically detected language code (e.g., "vi", "en").
        is_mock: Flag indicating whether mock simulation data was used.
    """
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "vi"
    is_mock: bool = False


# ----------------------------------------------------------------------
# Main Class: CodeSwitchedASR
# ----------------------------------------------------------------------
class CodeSwitchedASR:
    """Bilingual speech recognizer using OpenAI's Whisper.

    Manages model loading, automatic transcription, post-processing for
    academic/technical Vietnamese teencode, and mock fallbacks.
    """

    def __init__(
        self,
        model_name: str = "tiny",
        teencode_map: dict[str, str] | None = None,
    ):
        """Initializes the bilingual ASR processor.

        Args:
            model_name: Whisper model version ("tiny", "base", "small", etc.).
            teencode_map: Dictionary mapping abbreviations/teencode to full terms.
                If None, defaults to `settings.TEENCODE_MAP`.
        """
        self.model_name = model_name
        self._model = None
        self._is_mock_mode = False

        # Load teencode mapping from configuration if not explicitly provided
        if teencode_map is not None:
            self.teencode_map = teencode_map
        else:
            from edumind.config import settings
            self.teencode_map = settings.TEENCODE_MAP

        # Initialize the Whisper model
        self._load_model()

    def _load_model(self) -> None:
        """Loads the Whisper model into memory.

        If loading fails (due to missing libraries, hardware limits, or network errors),
        the engine gracefully transitions to mock mode.
        """
        try:
            import whisper
            logger.info(f"🎤 Loading Whisper-{self.model_name} model...")
            self._model = whisper.load_model(self.model_name)
            logger.success(f"✅ Loaded Whisper-{self.model_name} successfully!")
        except ImportError:
            logger.warning(
                "⚠️ Library 'openai-whisper' is not installed. "
                "Switching to mock mode."
            )
            self._is_mock_mode = True
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load Whisper-{self.model_name} model: {e}. "
                "Switching to mock mode."
            )
            self._is_mock_mode = True

    # ------------------------------------------------------------------
    # Core Transcription Methods
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribes the input audio file into text with segment timestamps.

        Args:
            audio_path: Path to the target audio file (WAV, MP3, FLAC, M4A).

        Returns:
            A TranscriptResult containing text, segments, and language.

        Raises:
            FileNotFoundError: If the specified audio file does not exist.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        # Fallback to mock data if in mock mode
        if self._is_mock_mode or self._model is None:
            logger.info("🎭 Using mock transcription (ASR model bypassed or unavailable).")
            return self._mock_transcribe(source_file=str(audio_path))

        logger.info(f"🎤 Transcribing audio: {audio_path.name}...")
        try:
            result = self._model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect language
                verbose=False,
            )

            # Map dict segments to TranscriptSegment objects
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
                f"✅ Transcription complete! Language: {detected_lang}, "
                f"Segments: {len(segments)}"
            )

            return TranscriptResult(
                text=full_text,
                segments=segments,
                language=detected_lang,
                is_mock=False,
            )

        except Exception as e:
            logger.error(f"❌ Transcription error: {e}. Falling back to mock mode.")
            return self._mock_transcribe(source_file=str(audio_path))

    # ------------------------------------------------------------------
    # Text Correction & Post-Processing
    # ------------------------------------------------------------------
    def post_process(self, text: str) -> str:
        """Corrects teencode and domain-specific abbreviations in the text.

        The correction algorithm:
            1. Sorts the teencode map by key length in descending order to avoid
               nested abbreviation substitution bugs (e.g., replacing "loss fn"
               before "fn").
            2. Uses word boundary regex patterns to perform safe substitutions.
            3. Standardizes redundant whitespaces.

        Args:
            text: Raw input text transcribed by ASR.

        Returns:
            Fully corrected, clean text.
        """
        if not text or not text.strip():
            return text

        corrected = text

        # Sort abbreviations in descending order of length to avoid nested replacement errors
        sorted_terms = sorted(
            self.teencode_map.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for abbr, full_form in sorted_terms:
            # Construct a safe regex pattern with word boundaries (\b)
            pattern = r"\b" + re.escape(abbr) + r"\b"
            corrected = re.sub(pattern, full_form, corrected, flags=re.IGNORECASE)

        # Standardize multiple spaces into single spaces
        corrected = re.sub(r"\s+", " ", corrected).strip()

        return corrected

    def get_corrections(self, original: str, corrected: str) -> list[dict]:
        """Compares original and corrected texts and extracts individual changes.

        Args:
            original: Raw transcription from ASR.
            corrected: Corrected post-processed text.

        Returns:
            A list of dictionary objects tracking changes, for example:
            [{"original": "fn", "corrected": "function", "position": 3}]
        """
        changes = []
        orig_words = original.split()
        corr_words = corrected.split()

        max_len = max(len(orig_words), len(corr_words))
        for i in range(min(len(orig_words), max_len)):
            if i < len(corr_words) and orig_words[i].lower() != corr_words[i].lower():
                changes.append({
                    "original": orig_words[i],
                    "corrected": corr_words[i] if i < len(corr_words) else "",
                    "position": i,
                })

        return changes

    # ------------------------------------------------------------------
    # Mock ASR Mode (Simulation & Demos)
    # ------------------------------------------------------------------
    def _mock_transcribe(self, source_file: str = "demo.wav") -> TranscriptResult:
        """Generates a mock transcription of a bilingual NLP lecture.

        Used when the physical Whisper model is bypassed or fails to load.

        Returns:
            A TranscriptResult preloaded with structured code-switched data.
        """
        logger.info("🎭 Generating mock transcription — bilingual NLP lecture...")

        # Simulating a realistic code-switched NLP lecture
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
        """Checks if the actual Whisper model is loaded and functional."""
        return self._model is not None and not self._is_mock_mode

    @property
    def is_mock(self) -> bool:
        """Checks if the speech processor is running in simulated mock mode."""
        return self._is_mock_mode
