"""EduMIND × Label Studio — ML Backend Model.

This module implements the ``EduMINDMLBackend``, a ``LabelStudioMLBase``
subclass that bridges Label Studio human-annotation workflows with the three
core EduMIND services:

    1. ``CodeSwitchedASR``     — Whisper-based bilingual speech recognition.
    2. ``VietMixTranslator``   — Code-mixing analysis and Vi→En translation.
    3. ``MultimodalRAG``       — Qdrant-backed retrieval-augmented generation.

Prediction flow (``predict``):
    - Audio tasks → Whisper transcription with confidence scoring.
    - Image/PDF tasks → OCR-based text extraction via pypdf / Tesseract fallback.

Fit / Active-Learning flow (``fit``):
    - On each confirmed annotation the gold-standard text is:
        a. Appended to ``corpus.jsonl`` (append-only audit log).
        b. Indexed into Qdrant via ``MultimodalRAG.embed_and_store`` for RAG.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``edumind`` is importable
# whether the backend is launched directly or via Docker with a bind-mount.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from label_studio_ml.model import LabelStudioMLBase  # noqa: E402  (import after sys.path fix)

from edumind.core.logging import get_logger  # noqa: E402
from edumind.models.chunks import DocumentChunk  # noqa: E402
from edumind.modules.rag_engine import MultimodalRAG  # noqa: E402
from edumind.modules.speech_processor import CodeSwitchedASR  # noqa: E402
from edumind.modules.vietmix_translator import VietMixTranslator  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DEFAULT_CORPUS_PATH = str(
    _PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
)

# Label Studio label-config names used in the EduMIND annotation template.
# Change these if the project configuration defines different from_name values.
_AUDIO_FROM_NAME = "transcript"
_AUDIO_TO_NAME = "audio"
_IMAGE_FROM_NAME = "ocr_text"
_IMAGE_TO_NAME = "image"

# Minimum confidence score below which predictions are flagged as low-quality.
_LOW_CONFIDENCE_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _logprob_to_confidence(segments: list[dict[str, Any]]) -> float:
    """Converts Whisper segment log-probabilities into a [0, 1] confidence score.

    Args:
        segments: List of segment dictionaries returned by ``whisper.transcribe``.
            Each segment should have an ``avg_logprob`` field.

    Returns:
        Mean sigmoid-transformed confidence across all segments.
        Returns 0.5 as a neutral default when no segments are provided.
    """
    if not segments:
        return 0.5

    logprobs = [
        float(seg.get("avg_logprob", -1.0))
        for seg in segments
        if "avg_logprob" in seg
    ]
    if not logprobs:
        return 0.5

    avg_logprob = sum(logprobs) / len(logprobs)
    # Map log-probability to (0, 1) via sigmoid: σ(x) = 1 / (1 + e^(-x))
    # Whisper log-probs are negative; typical range ≈ [-2, 0].
    # Shift by +1 so that avg_logprob = -1.0 → σ(0) = 0.5.
    shifted = avg_logprob + 1.0
    confidence = 1.0 / (1.0 + math.exp(-shifted))
    return round(confidence, 4)


def _compute_cmi(text: str) -> float:
    """Estimates the Code-Mixing Index (CMI) of a text string.

    CMI is defined here as the fraction of tokens that appear to belong to
    the non-dominant language (English tokens in a predominantly Vietnamese
    context).  Uses a simple ASCII heuristic: tokens whose characters are
    all ASCII are treated as English.

    Args:
        text: Input text string (may be code-mixed Vi/En).

    Returns:
        CMI score in [0.0, 1.0].  1.0 means fully English, 0.0 means fully
        Vietnamese.
    """
    tokens = text.split()
    if not tokens:
        return 0.0

    english_count = sum(1 for t in tokens if t.isascii() and t.isalpha())
    return round(english_count / len(tokens), 4)


class _AnnotationWriter:
    """Thread-safe append-only writer for the gold-standard corpus JSONL file.

    Each call to ``write`` atomically appends a single JSON record to the file,
    creating the file and its parent directories if necessary.
    """

    def __init__(self, corpus_path: str | Path) -> None:
        """Initializes the writer.

        Args:
            corpus_path: Absolute path to the target JSONL file.
        """
        self._path = Path(corpus_path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("corpus_writer_initialized", path=str(self._path))

    def write(self, record: dict[str, Any]) -> None:
        """Appends a single JSON record to the corpus file.

        Args:
            record: A dictionary representing one annotation record.
        """
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("corpus_record_written", annotation_id=record.get("annotation_id"))


# ---------------------------------------------------------------------------
# Main ML Backend
# ---------------------------------------------------------------------------

class EduMINDMLBackend(LabelStudioMLBase):
    """Label Studio ML Backend for the EduMIND educational AI platform.

    Provides automatic pre-annotation (``predict``) for audio transcription
    and PDF/image OCR tasks, and integrates human-verified annotations into
    the Qdrant vector store for continuous RAG improvement (``fit``).

    Environment variables read at startup:
        CORPUS_JSONL_PATH: Override path for the gold-standard corpus file.
    """

    def setup(self) -> None:
        """Initializes all EduMIND core services.

        Failures in any single service are caught and logged; the backend
        remains operational with graceful fallbacks so that Label Studio can
        still serve annotation UIs even if a model fails to load.
        """
        logger.info("edumind_backend_setup_started")

        # ---- 1. ASR: Whisper ------------------------------------------------
        try:
            self._asr = CodeSwitchedASR()
            logger.info(
                "asr_initialized",
                model=self._asr.model_name,
                mock_mode=self._asr.is_mock,
            )
        except Exception as exc:
            logger.error("asr_initialization_failed", error=str(exc))
            self._asr = None  # type: ignore[assignment]

        # ---- 2. Translation: VietMix ----------------------------------------
        try:
            self._translator = VietMixTranslator()
            logger.info("translator_initialized")
        except Exception as exc:
            logger.error("translator_initialization_failed", error=str(exc))
            self._translator = None  # type: ignore[assignment]

        # ---- 3. RAG Engine: Qdrant ------------------------------------------
        try:
            self._rag = MultimodalRAG()
            logger.info("rag_engine_initialized")
        except Exception as exc:
            logger.error("rag_engine_initialization_failed", error=str(exc))
            self._rag = None  # type: ignore[assignment]

        # ---- 4. Corpus writer -----------------------------------------------
        corpus_path = os.environ.get("CORPUS_JSONL_PATH", _DEFAULT_CORPUS_PATH)
        self._writer = _AnnotationWriter(corpus_path)

        logger.info("edumind_backend_setup_complete")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        tasks: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generates pre-annotations for a batch of Label Studio tasks.

        Supported task types:
            - **Audio**: Runs Whisper ASR and returns a textarea with the
              normalized transcript and a confidence score.
            - **Image / PDF**: Extracts text via OCR and returns a textarea.

        Args:
            tasks: List of Label Studio task dictionaries.
            context: Optional metadata context from Label Studio.
            **kwargs: Additional keyword arguments (forwarded from LS SDK).

        Returns:
            A list of prediction dictionaries, one per input task.
        """
        predictions: list[dict[str, Any]] = []

        for task in tasks:
            task_data: dict[str, Any] = task.get("data", {})

            try:
                if "audio" in task_data:
                    prediction = self._predict_audio(task_data)
                elif "image" in task_data:
                    prediction = self._predict_image(task_data)
                else:
                    logger.warning(
                        "unknown_task_type_skipping",
                        keys=list(task_data.keys()),
                    )
                    prediction = self._empty_prediction()
            except Exception as exc:
                logger.error(
                    "predict_task_failed_returning_empty",
                    error=str(exc),
                    task_id=task.get("id"),
                )
                prediction = self._empty_prediction()

            predictions.append(prediction)

        return predictions

    def _predict_audio(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Generates an ASR pre-annotation for an audio task.

        The audio path may be expressed as:
            - An absolute filesystem path.
            - A Label Studio ``/data/local-files/`` relative URI.

        Args:
            task_data: The ``data`` field of a single Label Studio task.

        Returns:
            A prediction dictionary compatible with the Label Studio API.
        """
        audio_ref: str = task_data.get("audio", "")
        audio_path = self._resolve_local_path(audio_ref)

        # ---- Transcribe -----------------------------------------------------
        if self._asr is None or not Path(audio_path).exists():
            logger.warning(
                "asr_unavailable_or_file_missing_using_placeholder",
                path=audio_path,
            )
            return self._placeholder_audio_prediction(audio_ref)

        transcript_result = self._asr.transcribe(audio_path)
        normalized_text = self._asr.post_process(transcript_result.text)

        # ---- Confidence from Whisper segments --------------------------------
        raw_segments: list[dict[str, Any]] = [
            seg.model_dump()
            for seg in transcript_result.segments
        ]
        confidence = _logprob_to_confidence(raw_segments)

        # ---- Translation (best-effort) --------------------------------------
        translation: str = ""
        if self._translator is not None:
            try:
                translation = self._translator.translate_to_english(normalized_text)
            except Exception as exc:
                logger.warning("translation_failed_in_predict", error=str(exc))

        # ---- Build Label Studio result dict ---------------------------------
        return {
            "result": [
                {
                    "from_name": _AUDIO_FROM_NAME,
                    "to_name": _AUDIO_TO_NAME,
                    "type": "textarea",
                    "value": {
                        "text": [normalized_text],
                    },
                },
                {
                    "from_name": "translation_hint",
                    "to_name": _AUDIO_TO_NAME,
                    "type": "textarea",
                    "value": {
                        "text": [translation],
                    },
                },
            ],
            "score": confidence,
            "model_version": f"edumind-asr-{self._asr.model_name}",
        }

    def _predict_image(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Generates an OCR pre-annotation for an image / PDF task utilizing IBM Docling.

        Args:
            task_data: The ``data`` field of a single Label Studio task.

        Returns:
            A prediction dictionary compatible with the Label Studio API.
        """
        image_ref: str = task_data.get("image", "")
        image_path = self._resolve_local_path(image_ref)
        extracted_text = ""
        confidence = 0.75  # Higher default confidence due to Docling's layout parsing

        path_obj = Path(image_path)
        if not path_obj.exists():
            return self._placeholder_image_prediction(image_ref)

        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(path_obj))
            doc = result.document

            text_parts = []
            for item, level in doc.iterate_items():
                if hasattr(item, "text") and item.text:
                    text_parts.append(item.text.strip())

            extracted_text = "\n\n".join(text_parts)
        except Exception as exc:
            logger.warning("docling_ocr_failed", error=str(exc), path=image_path)
            # Fallback to simple txt file reading if it's plain text
            try:
                with open(path_obj, encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
            except Exception:
                extracted_text = ""

        if not extracted_text.strip():
            return self._placeholder_image_prediction(image_ref)

        return {
            "result": [
                {
                    "from_name": _IMAGE_FROM_NAME,
                    "to_name": _IMAGE_TO_NAME,
                    "type": "textarea",
                    "value": {
                        "text": [extracted_text.strip()],
                    },
                }
            ],
            "score": confidence,
            "model_version": "edumind-docling-ocr-v1",
        }

    # -----------------------------------------------------------------------
    # Fit / Active Learning
    # -----------------------------------------------------------------------

    def fit(
        self,
        event: str,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Processes a completed annotation and updates the EduMIND knowledge base.

        Called by Label Studio on ``ANNOTATION_CREATED`` and
        ``ANNOTATION_UPDATED`` events.  The gold-standard text is:

            1. Extracted from the annotation payload.
            2. Written to the append-only ``corpus.jsonl``.
            3. Indexed into Qdrant asynchronously (non-blocking background thread).

        Args:
            event: Label Studio webhook event name (e.g., ``"ANNOTATION_CREATED"``).
            data: Full event payload from Label Studio.
            **kwargs: Additional keyword arguments.

        Returns:
            A status dictionary indicating success or partial failure.
        """
        logger.info("fit_event_received", event=event)

        if event not in {"ANNOTATION_CREATED", "ANNOTATION_UPDATED"}:
            logger.info("fit_event_ignored", event=event)
            return {"status": "ignored", "event": event}

        try:
            annotation = data.get("annotation", {})
            task = data.get("task", {})
            task_data: dict[str, Any] = task.get("data", {})

            annotation_id: int | str = annotation.get("id", "unknown")
            task_id: int | str = task.get("id", "unknown")

            # ---- Extract gold-standard text from annotation -----------------
            gold_text = self._extract_gold_text(annotation)
            if not gold_text:
                logger.warning(
                    "fit_empty_gold_text_skipping",
                    annotation_id=annotation_id,
                )
                return {"status": "skipped", "reason": "empty_gold_text"}

            # ---- Translate gold text (best-effort) --------------------------
            gold_translation: str = ""
            cmi_score: float = _compute_cmi(gold_text)  # fallback heuristic
            if self._translator is not None:
                try:
                    cmi_result = self._translator.calculate_cmi(gold_text)
                    cmi_score = cmi_result.score
                    gold_translation = self._translator.translate_to_english(gold_text)
                except Exception as exc:
                    logger.warning("fit_translation_failed", error=str(exc))

            # ---- Dynamic Teencode Learning ----------------------------------
            if self._asr is not None:
                try:
                    original_transcript = self._extract_prediction_text(task)
                    if original_transcript:
                        changes = self._asr.get_corrections(original_transcript, gold_text)
                        for ch in changes:
                            orig = str(ch.get("original", ""))
                            corr = str(ch.get("corrected", ""))
                            if orig and corr:
                                self._asr.update_teencode(orig, corr)
                except Exception as exc:
                    logger.warning("fit_teencode_learning_failed", error=str(exc))

            # ---- Build corpus record ----------------------------------------
            audio_ref: str = task_data.get("audio", task_data.get("image", ""))
            record: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "source_ref": audio_ref,
                "gold_transcript": gold_text,
                "translation": gold_translation,
                "cmi_score": cmi_score,
                "annotation_id": annotation_id,
                "task_id": task_id,
                "event": event,
            }

            # ---- Persist to corpus.jsonl ------------------------------------
            self._writer.write(record)

            # ---- Index into Qdrant (non-blocking) ---------------------------
            if self._rag is not None:
                indexing_thread = threading.Thread(
                    target=self._index_annotation,
                    args=(gold_text, record),
                    daemon=True,
                    name=f"qdrant-index-{annotation_id}",
                )
                indexing_thread.start()
                logger.info("qdrant_indexing_thread_started", annotation_id=annotation_id)

            return {
                "status": "success",
                "annotation_id": annotation_id,
                "task_id": task_id,
                "indexed_to_qdrant": self._rag is not None,
            }

        except Exception as exc:
            logger.error("fit_unexpected_failure", error=str(exc))
            return {"status": "error", "detail": str(exc)}

    def _index_annotation(self, gold_text: str, record: dict[str, Any]) -> None:
        """Indexes a gold-standard annotation into Qdrant.

        Runs inside a daemon thread to avoid blocking the Label Studio webhook
        response.  All exceptions are caught and logged to prevent thread crashes.

        Args:
            gold_text: The verified/corrected transcript text.
            record: Full corpus record dictionary (used as chunk metadata).
        """
        try:
            chunks = self._rag.ingest_text(  # type: ignore[union-attr]
                text=gold_text,
                source_name=record.get("source_ref", "label_studio_annotation"),
                metadata_extra={
                    "annotation_id": record.get("annotation_id"),
                    "task_id": record.get("task_id"),
                    "cmi_score": record.get("cmi_score", 0.0),
                    "gold_annotation": True,
                    "timestamp": record.get("timestamp"),
                },
            )
            count = self._rag.embed_and_store(chunks)  # type: ignore[union-attr]
            logger.info(
                "qdrant_indexing_complete",
                chunks_indexed=count,
                annotation_id=record.get("annotation_id"),
            )
        except Exception as exc:
            logger.error(
                "qdrant_indexing_thread_failed",
                error=str(exc),
                annotation_id=record.get("annotation_id"),
            )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_local_path(self, uri: str) -> str:
        """Converts a Label Studio URI or relative path to an absolute filesystem path.

        Label Studio prefixes local file URIs with ``/data/local-files/?d=<path>``.
        This helper strips that prefix and prepends ``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT``.

        Args:
            uri: Raw URI string from the Label Studio task data field.

        Returns:
            Absolute filesystem path string.
        """
        if not uri:
            return uri

        # Handle LS local-file URI format:  /data/local-files/?d=<relative-path>
        if "/data/local-files/" in uri:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(uri)
            params = parse_qs(parsed.query)
            relative = params.get("d", [""])[0]
            document_root = os.environ.get(
                "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT",
                str(_PROJECT_ROOT / "data"),
            )
            return str(Path(document_root) / relative)

        # If already an absolute path, return as-is
        if Path(uri).is_absolute():
            return uri

        # Treat as relative to project data directory
        return str(_PROJECT_ROOT / "data" / uri)

    def _extract_gold_text(self, annotation: dict[str, Any]) -> str:
        """Extracts the human-verified text from a Label Studio annotation dict.

        Searches all result items for ``textarea`` type values and concatenates
        the text from the primary transcript field.

        Args:
            annotation: Label Studio annotation dictionary containing ``result`` list.

        Returns:
            Combined gold-standard text string; empty string if not found.
        """
        results: list[dict[str, Any]] = annotation.get("result", [])
        texts: list[str] = []

        for item in results:
            if item.get("type") != "textarea":
                continue
            # Prioritize items from the main transcript field
            if item.get("from_name") in {_AUDIO_FROM_NAME, _IMAGE_FROM_NAME}:
                value_texts = item.get("value", {}).get("text", [])
                texts.extend(str(t).strip() for t in value_texts if t)

        return " ".join(texts).strip()

    def _empty_prediction(self) -> dict[str, Any]:
        """Returns an empty prediction dictionary used as a safe fallback.

        Returns:
            Empty prediction with zero confidence.
        """
        return {
            "result": [],
            "score": 0.0,
            "model_version": "edumind-fallback",
        }

    def _placeholder_audio_prediction(self, audio_ref: str) -> dict[str, Any]:
        """Returns a placeholder audio transcript for unresolvable audio paths.

        Args:
            audio_ref: Original audio URI that could not be resolved.

        Returns:
            Prediction with a helpful placeholder text.
        """
        return {
            "result": [
                {
                    "from_name": _AUDIO_FROM_NAME,
                    "to_name": _AUDIO_TO_NAME,
                    "type": "textarea",
                    "value": {
                        "text": [f"[ASR ERROR — file not found: {audio_ref}]"],
                    },
                }
            ],
            "score": 0.0,
            "model_version": "edumind-fallback",
        }

    def _placeholder_image_prediction(self, image_ref: str) -> dict[str, Any]:
        """Returns a placeholder OCR result for unprocessable images.

        Args:
            image_ref: Original image URI that could not be processed.

        Returns:
            Prediction with a helpful placeholder text.
        """
        return {
            "result": [
                {
                    "from_name": _IMAGE_FROM_NAME,
                    "to_name": _IMAGE_TO_NAME,
                    "type": "textarea",
                    "value": {
                        "text": [f"[OCR ERROR — file not processed: {image_ref}]"],
                    },
                }
            ],
            "score": 0.0,
            "model_version": "edumind-fallback",
        }

    def _extract_prediction_text(self, task: dict[str, Any]) -> str:
        """Extracts the original ASR/OCR text prediction generated by this ML Backend.

        Searches within the prediction payload to find the initial generated text value.
        """
        predictions = task.get("predictions", [])
        if not predictions:
            return ""

        pred = predictions[0]
        results = pred.get("result", [])
        texts = []
        for item in results:
            if item.get("type") == "textarea" and item.get("from_name") in {_AUDIO_FROM_NAME, _IMAGE_FROM_NAME}:
                val = item.get("value", {}).get("text", [])
                texts.extend(str(t).strip() for t in val if t)
        return " ".join(texts).strip()
