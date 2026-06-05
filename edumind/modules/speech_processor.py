"""EduMIND — Code-Switched Automatic Speech Recognition (ASR).

Transcribes bilingual academic/technical lectures and resolves Vietnamese teencode.
Supports OpenAI's Whisper model with dynamic fallback capabilities.
"""

from __future__ import annotations

from pathlib import Path
import re

import threading
import json

from edumind.config import get_settings
from edumind.core.logging import get_logger
from edumind.models.transcription import TranscriptResult, TranscriptSegment

logger = get_logger(__name__)


class CodeSwitchedASR:
    """Bilingual speech recognizer using OpenAI's Whisper.

    Manages model loading, automatic transcription, post-processing for
    academic/technical Vietnamese teencode, and mock fallbacks.
    """

    def __init__(
        self,
        model_name: str | None = None,
        teencode_map: dict[str, str] | None = None,
    ):
        """Initializes the bilingual ASR processor.

        Args:
            model_name: Whisper model version ("tiny", "base", "small", etc.).
                If None, resolves from config settings.
            teencode_map: Dictionary mapping abbreviations/teencode to full terms.
                If None, defaults to ``settings.TEENCODE_MAP``.
        """
        settings = get_settings()
        self.model_name = model_name or settings.WHISPER_MODEL
        self._model = None
        self._is_mock_mode = False
        self._lock = threading.Lock()
        self.dynamic_teencode_path = settings.DATA_DIR / "processed" / "dynamic_teencode.json"

        # Load teencode mapping from configuration if not explicitly provided
        self.teencode_map = teencode_map if teencode_map is not None else dict(settings.TEENCODE_MAP)

        if teencode_map is None:
            self._load_dynamic_teencode()

        # Initialize the Whisper model
        self._load_model()

    def _load_model(self) -> None:
        """Loads the Whisper model into memory.

        If loading fails (due to missing libraries, hardware limits, or network errors),
        the engine gracefully transitions to mock mode.
        """
        try:
            import whisper

            logger.info("loading_whisper_model", model=self.model_name)
            self._model = whisper.load_model(self.model_name)
            logger.info("loaded_whisper_model", model=self.model_name)
        except ImportError:
            logger.warning(
                "whisper_library_not_installed_mock_mode_fallback",
                reason="Library 'openai-whisper' is not installed",
            )
            self._is_mock_mode = True
        except Exception as e:
            logger.warning(
                "whisper_model_load_failed_mock_mode_fallback",
                model=self.model_name,
                error=str(e),
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
            logger.info("using_mock_transcription_fallback", source_file=audio_path.name)
            return self._mock_transcribe(source_file=str(audio_path))

        logger.info("transcribing_audio_file", file_name=audio_path.name)
        try:
            result = self._model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect language
                verbose=False,
            )

            # Map dict segments to TranscriptSegment objects
            segments = [
                TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=str(seg["text"]).strip(),
                )
                for seg in result.get("segments", [])
            ]

            detected_lang = result.get("language", "vi")
            full_text = result.get("text", "").strip()

            logger.info("transcription_completed", language=detected_lang, segments_count=len(segments))

            return TranscriptResult(
                text=full_text,
                segments=segments,
                language=detected_lang,
                is_mock=False,
            )

        except Exception as e:
            logger.error("transcription_execution_failed_falling_back", error=str(e))
            return self._mock_transcribe(source_file=str(audio_path))

    # ------------------------------------------------------------------
    # Text Correction & Post-Processing
    # ------------------------------------------------------------------
    def post_process(self, text: str) -> str:
        """Corrects teencode and domain-specific abbreviations in the text.

        The correction algorithm:
            1. Sorts the teencode map by key length in descending order to avoid
               nested abbreviation substitution bugs.
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

    def get_corrections(self, original: str, corrected: str) -> list[dict[str, int | str]]:
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

        Args:
            source_file: Name of the mock source file.

        Returns:
            A TranscriptResult preloaded with structured code-switched data.
        """
        logger.info("generating_mock_bilingual_nlp_lecture_transcription", file=source_file)

        # Simulating a realistic code-switched NLP lecture
        mock_segments = [
            TranscriptSegment(
                start=0.0,
                end=5.2,
                text="Xin chào các bạn, hôm nay chúng ta sẽ discuss về NLP",
            ),
            TranscriptSegment(
                start=5.2,
                end=12.8,
                text="Đầu tiên mình sẽ explain về loss fn trong deep learning model",
            ),
            TranscriptSegment(
                start=12.8,
                end=20.1,
                text="Các bạn cần submit bài trc dl nhé, ko là bị trừ điểm",
            ),
            TranscriptSegment(
                start=20.1,
                end=28.5,
                text="Bây giờ mình sẽ bđ phần attention mechanism trong transformer",
            ),
            TranscriptSegment(
                start=28.5,
                end=35.0,
                text="Cái lr nên set khoảng 2e-5 cho fine-tuning BERT",
            ),
            TranscriptSegment(
                start=35.0,
                end=42.3,
                text="Mn nhớ review lại backprop và gradient descent trước buổi sau",
            ),
        ]

        full_text = " ".join(seg.text for seg in mock_segments)

        return TranscriptResult(
            text=full_text,
            segments=mock_segments,
            language="vi",
            is_mock=True,
        )

    def _load_dynamic_teencode(self) -> None:
        """Loads dynamic teencode corrections from local JSON files."""
        if self.dynamic_teencode_path.exists():
            try:
                with open(self.dynamic_teencode_path, encoding="utf-8") as f:
                    dyn_map = json.load(f)
                    if isinstance(dyn_map, dict):
                        self.teencode_map.update(dyn_map)
                        logger.info("loaded_dynamic_teencode_mapping", count=len(dyn_map))
            except Exception as e:
                logger.warning("failed_to_load_dynamic_teencode", error=str(e))

    def update_teencode(self, orig: str, corr: str) -> None:
        """Dynamically updates the teencode map and serializes to disk.

        Thread-safe execution ensures multiple webhooks do not cause race conditions.
        """
        orig_clean = orig.strip().lower()
        corr_clean = corr.strip().lower()

        if not orig_clean or not corr_clean or orig_clean == corr_clean:
            return

        # Avoid learning numbers or extremely long phrases as shorthand keys
        if not orig_clean.isalpha() or len(orig_clean) >= len(corr_clean):
            return

        with self._lock:
            self.teencode_map[orig_clean] = corr_clean

            # Read existing dynamic teencode to avoid overwriting other values
            dyn_map = {}
            if self.dynamic_teencode_path.exists():
                try:
                    with open(self.dynamic_teencode_path, encoding="utf-8") as f:
                        dyn_map = json.load(f)
                except Exception:
                    dyn_map = {}

            dyn_map[orig_clean] = corr_clean

            try:
                self.dynamic_teencode_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.dynamic_teencode_path, "w", encoding="utf-8") as f:
                    json.dump(dyn_map, f, ensure_ascii=False, indent=4)
                logger.info("dynamic_teencode_map_updated", original=orig_clean, corrected=corr_clean)
            except Exception as e:
                logger.error("failed_to_write_dynamic_teencode", error=str(e))

    @property
    def is_ready(self) -> bool:
        """Checks if the actual Whisper model is loaded and functional."""
        return self._model is not None and not self._is_mock_mode

    @property
    def is_mock(self) -> bool:
        """Checks if the speech processor is running in simulated mock mode."""
        return self._is_mock_mode
