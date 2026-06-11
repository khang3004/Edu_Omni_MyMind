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
        self._is_causal = False

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
        if "vi-en" in name_lower or "vi2en" in name_lower or "vietmix-qwen2.5" in name_lower:
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
            from transformers import AutoConfig, AutoTokenizer

            try:
                config = AutoConfig.from_pretrained(self._model_name)
            except Exception as config_err:
                if "qwen2.5" in name_lower:
                    logger.info(
                        "config_not_found_using_qwen2.5_fallback_config",
                        model_name=self._model_name,
                    )
                    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
                else:
                    raise config_err

            is_causal = False
            if hasattr(config, "architectures") and config.architectures:
                is_causal = any("CausalLM" in arch for arch in config.architectures)
            self._is_causal = is_causal

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            if is_causal:
                import torch
                from transformers import AutoModelForCausalLM
                from edumind.core.container import get_settings

                try:
                    device = get_settings().DEVICE
                except Exception:
                    device = "cpu"
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"

                dtype = torch.float32
                if device != "cpu":
                    dtype = torch.float16
                    if device == "cuda" and torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16

                logger.info(
                    "loading_causal_lm_translation_model",
                    model_name=self._model_name,
                    device=device,
                    dtype=str(dtype),
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    config=config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                if device != "cpu":
                    self._model = self._model.to(device)
            else:
                from transformers import AutoModelForSeq2SeqLM

                self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, config=config)

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

            # Seq2Seq generation flow
            if not getattr(self, "_is_causal", False):
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
                # Ensure input tensors are on the same device as the model
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                outputs = self._model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                )
                decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                return decoded.strip()

            # CausalLM generation flow (e.g. Qwen2.5)
            else:
                # Format using ChatML template structure
                instruction = (
                    "Dịch câu tiếng Việt code-mixed AI/Data Science sau sang tiếng Anh học thuật chuẩn, "
                    "giữ nguyên các thuật ngữ kỹ thuật và ý nghĩa chuyên môn."
                )
                prompt = (
                    f"<|im_start|>system\n{instruction}<|im_end|>\n"
                    f"<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                )
                # Ensure input tensors are on the same device as the model
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                import torch

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,  # Greedy decoding for translation accuracy
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                # Slice output to only decode newly generated tokens
                decoded = self._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
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
