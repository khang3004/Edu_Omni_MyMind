"""HuggingFace Seq2Seq Machine Translation Provider.

Translates code-mixed bilingual text utilizing neural seq2seq models from HuggingFace
with auto-fallback to rule-based translation upon failures.
"""

from __future__ import annotations

from edumind.core.logging import get_logger
from edumind.services.translation.base import TranslationProvider
from edumind.services.translation.rule_based import RuleBasedTranslationProvider

logger = get_logger(__name__)


class HuggingFaceTranslationProvider(TranslationProvider):
    """Neural machine translation provider using HuggingFace Transformers."""

    def __init__(self, model_name: str = "none"):
        """Initializes the provider.

        Args:
            model_name: HuggingFace model hub path (e.g. 'vinai/vinallm'). If 'none',
                runs exclusively in rule-based fallback mode.
        """
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._fallback = RuleBasedTranslationProvider()
        self._load_attempted = False
        self._usable = False
        self._supported_direction = "both"
        self._use_prefix = False

    def _load_model(self) -> None:
        """Lazily loads the transformer model and tokenizer."""
        if self._load_attempted:
            return

        self._load_attempted = True
        if not self._model_name or self._model_name.lower() == "none":
            logger.info("neural_translation_disabled_by_config")
            self._usable = False
            return

        # Analyze direction support based on model naming patterns
        name_lower = self._model_name.lower()
        if "vi-en" in name_lower or "vi2en" in name_lower:
            self._supported_direction = "vi-en"
        elif "en-vi" in name_lower or "en2vi" in name_lower:
            self._supported_direction = "en-vi"
        else:
            self._supported_direction = "both"

        # Check if model requires a prefix (typically T5/mT5 models)
        if "t5" in name_lower:
            self._use_prefix = True
        else:
            self._use_prefix = False

        try:
            logger.info(
                "loading_huggingface_translation_model",
                model_name=self._model_name,
                direction=self._supported_direction,
                use_prefix=self._use_prefix,
            )
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            self._usable = True
            logger.info("loaded_huggingface_translation_model", model_name=self._model_name)
        except Exception as e:
            logger.warning(
                "huggingface_translation_load_failed_falling_back",
                model_name=self._model_name,
                error=str(e),
            )
            self._usable = False
            self._model = None
            self._tokenizer = None

    @property
    def is_model_loaded(self) -> bool:
        """True if the neural model was successfully imported and loaded."""
        self._load_model()
        return self._usable and self._model is not None

    def translate_to_english(self, text: str) -> str:
        """Translates text to standard English using transformer model or rule-based fallback."""
        if not text or not text.strip():
            return ""

        self._load_model()
        # Fallback if model loading failed or if model only supports English -> Vietnamese
        if not self._usable or self._supported_direction == "en-vi":
            return self._fallback.translate_to_english(text)

        return self._model_translate(text, target="en")

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates text to standard Vietnamese using transformer model or rule-based fallback."""
        if not text or not text.strip():
            return ""

        self._load_model()
        # Fallback if model loading failed or if model only supports Vietnamese -> English
        if not self._usable or self._supported_direction == "vi-en":
            return self._fallback.translate_to_vietnamese(text)

        return self._model_translate(text, target="vi")

    def _model_translate(self, text: str, target: str = "en") -> str:
        """Invokes HuggingFace generation pipeline."""
        try:
            assert self._tokenizer is not None
            assert self._model is not None

            input_text = text
            if self._use_prefix:
                prefix = f"translate to {target}: "
                input_text = prefix + text

            inputs = self._tokenizer(
                input_text,
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
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return decoded.strip()
        except Exception as e:
            logger.warning(
                "neural_translation_generation_failed_using_fallback",
                target=target,
                error=str(e),
            )
            if target == "en":
                return self._fallback.translate_to_english(text)
            return self._fallback.translate_to_vietnamese(text)
