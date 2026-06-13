"""HuggingFace Machine Translation Provider.

Architecture-agnostic translation provider supporting both Seq2Seq
(MarianMT, mBART, T5) and CausalLM (Qwen, Llama, Mistral) models.
All model-specific behaviour is driven by configuration — zero hardcoding.
"""

from __future__ import annotations

from edumind.core.logging import get_logger
from edumind.services.translation.base import TranslationProvider
from edumind.services.translation.rule_based import RuleBasedTranslationProvider

logger = get_logger(__name__)


class HuggingFaceTranslationProvider(TranslationProvider):
    """Neural machine translation provider using HuggingFace Transformers.

    Supports any Seq2Seq or CausalLM model from the HuggingFace Hub.
    Architecture and direction are auto-detected or overridden via config.
    Falls back to rule-based translation on load failures or OOM risk.
    """

    def __init__(
        self,
        model_name: str = "none",
        *,
        arch: str = "auto",
        direction: str = "auto",
        base_model: str = "",
        max_model_gb: float = 4.0,
    ):
        """Initializes the provider.

        Args:
            model_name: HuggingFace model hub path. If 'none', uses rule-based only.
            arch: Architecture override — "auto", "seq2seq", or "causal".
            direction: Direction override — "auto", "vi-en", "en-vi", or "both".
            base_model: Fallback model ID for loading config (for LoRA adapters).
            max_model_gb: Maximum model size in GB to allow loading. 0 = no limit.
        """
        self._model_name = model_name
        self._arch_override = arch.lower()
        self._direction_override = direction.lower()
        self._base_model = base_model
        self._max_model_gb = max_model_gb

        self._model = None
        self._tokenizer = None
        self._fallback = RuleBasedTranslationProvider()
        self._load_attempted = False
        self._usable = False
        self._is_causal = False
        self._supported_direction = "both"
        self._use_prefix = False

    # ------------------------------------------------------------------
    # Direction detection
    # ------------------------------------------------------------------

    def _detect_direction(self) -> str:
        """Infers translation direction from model name patterns.

        Returns:
            "vi-en", "en-vi", or "both".
        """
        if self._direction_override != "auto":
            return self._direction_override

        name_lower = self._model_name.lower()
        if "vi-en" in name_lower or "vi2en" in name_lower:
            return "vi-en"
        elif "en-vi" in name_lower or "en2vi" in name_lower:
            return "en-vi"
        return "both"

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    def _detect_arch(self, config) -> bool:
        """Detects whether the model is CausalLM.

        Args:
            config: HuggingFace AutoConfig object.

        Returns:
            True if CausalLM, False if Seq2Seq.
        """
        if self._arch_override == "causal":
            return True
        if self._arch_override == "seq2seq":
            return False

        # Auto-detect from model config architectures
        if hasattr(config, "architectures") and config.architectures:
            return any("CausalLM" in arch for arch in config.architectures)
        return False

    # ------------------------------------------------------------------
    # Model size estimation
    # ------------------------------------------------------------------

    def _estimate_model_size_gb(self) -> float:
        """Estimates model size from HuggingFace Hub metadata.

        Returns:
            Estimated size in GB, or 0.0 if unable to determine.
        """
        # Bytes per parameter for each dtype
        _DTYPE_BYTES = {
            "F64": 8, "F32": 4, "BF16": 2, "F16": 2,
            "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4,
            "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
        }

        try:
            from huggingface_hub import model_info as hf_model_info

            info = hf_model_info(self._model_name)

            # Preferred: use SafeTensorsInfo metadata
            if info.safetensors and hasattr(info.safetensors, "parameters"):
                total_bytes = 0
                for dtype_name, param_count in info.safetensors.parameters.items():
                    bpp = _DTYPE_BYTES.get(dtype_name.upper(), 2)  # default 2 bytes
                    total_bytes += param_count * bpp
                if total_bytes > 0:
                    return total_bytes / (1024**3)

            # Fallback: sum sibling file sizes
            if info.siblings:
                total = sum(
                    s.size for s in info.siblings
                    if s.size and s.rfilename.endswith((".safetensors", ".bin"))
                )
                if total > 0:
                    return total / (1024**3)
        except Exception as e:
            logger.debug("model_size_estimation_failed", error=str(e))

        return 0.0

    # ------------------------------------------------------------------
    # Device and dtype selection (cross-platform, no hacks)
    # ------------------------------------------------------------------

    def _get_device_and_dtype(self):
        """Cross-platform device and dtype selection.

        Returns:
            Tuple of (device_str, torch_dtype).
        """
        import torch

        try:
            from edumind.core.container import get_settings
            device = get_settings().DEVICE
        except Exception:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

        # dtype selection: float16 for accelerators, float32 for CPU
        dtype = torch.float32
        if device != "cpu":
            dtype = torch.float16
            if device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16

        return device, dtype

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily loads the transformer model and tokenizer."""
        if self._load_attempted:
            return

        self._load_attempted = True
        if not self._model_name or self._model_name.lower() == "none":
            logger.info("neural_translation_disabled_by_config")
            self._usable = False
            return

        # Detect direction
        self._supported_direction = self._detect_direction()

        # Check if model requires a prefix (typically T5/mT5 models)
        name_lower = self._model_name.lower()
        self._use_prefix = "t5" in name_lower

        try:
            logger.info(
                "loading_huggingface_translation_model",
                model_name=self._model_name,
                direction=self._supported_direction,
                use_prefix=self._use_prefix,
            )

            # RAM guard: check model size before loading
            if self._max_model_gb > 0:
                estimated_gb = self._estimate_model_size_gb()
                if estimated_gb > 0 and estimated_gb > self._max_model_gb:
                    logger.warning(
                        "model_exceeds_ram_limit_using_fallback",
                        model_name=self._model_name,
                        estimated_gb=round(estimated_gb, 1),
                        max_gb=self._max_model_gb,
                    )
                    self._usable = False
                    return

            from transformers import AutoConfig, AutoTokenizer

            # Load config — try model first, then base_model fallback
            try:
                config = AutoConfig.from_pretrained(self._model_name)
            except Exception:
                if self._base_model:
                    logger.info(
                        "config_fallback_to_base_model",
                        model_name=self._model_name,
                        base_model=self._base_model,
                    )
                    config = AutoConfig.from_pretrained(self._base_model)
                else:
                    raise

            # Detect architecture
            self._is_causal = self._detect_arch(config)

            # Load tokenizer — try model first, then base_model fallback
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            except Exception:
                if self._base_model:
                    self._tokenizer = AutoTokenizer.from_pretrained(self._base_model)
                else:
                    raise

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load the model based on detected architecture
            if self._is_causal:
                self._load_causal_model(config)
            else:
                self._load_seq2seq_model(config)

            self._usable = True
            logger.info(
                "loaded_huggingface_translation_model",
                model_name=self._model_name,
                arch="causal" if self._is_causal else "seq2seq",
                direction=self._supported_direction,
            )
        except Exception as e:
            logger.warning(
                "huggingface_translation_load_failed_falling_back",
                model_name=self._model_name,
                error=str(e),
            )
            self._usable = False
            self._model = None
            self._tokenizer = None

    def _load_causal_model(self, config) -> None:
        """Loads a CausalLM model with proper device/dtype settings."""
        import torch
        from transformers import AutoModelForCausalLM

        device, dtype = self._get_device_and_dtype()

        logger.info(
            "loading_causal_lm_model",
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

    def _load_seq2seq_model(self, config) -> None:
        """Loads a Seq2Seq model with proper device/dtype settings."""
        import torch
        from transformers import AutoModelForSeq2SeqLM

        device, dtype = self._get_device_and_dtype()

        logger.info(
            "loading_seq2seq_model",
            model_name=self._model_name,
            device=device,
            dtype=str(dtype),
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_name,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if device != "cpu":
            self._model = self._model.to(device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        # Fallback if model loading failed or if model only supports English → Vietnamese
        if not self._usable or self._supported_direction == "en-vi":
            return self._fallback.translate_to_english(text)

        return self._model_translate(text, target="en")

    def translate_to_vietnamese(self, text: str) -> str:
        """Translates text to standard Vietnamese using transformer model or rule-based fallback."""
        if not text or not text.strip():
            return ""

        self._load_model()
        # Fallback if model loading failed or if model only supports Vietnamese → English
        if not self._usable or self._supported_direction == "vi-en":
            return self._fallback.translate_to_vietnamese(text)

        return self._model_translate(text, target="vi")

    # ------------------------------------------------------------------
    # Translation logic
    # ------------------------------------------------------------------

    def _model_translate(self, text: str, target: str = "en") -> str:
        """Invokes HuggingFace generation pipeline."""
        try:
            assert self._tokenizer is not None
            assert self._model is not None

            if self._is_causal:
                return self._translate_causal(text, target)
            else:
                return self._translate_seq2seq(text, target)

        except Exception as e:
            logger.warning(
                "neural_translation_generation_failed_using_fallback",
                target=target,
                error=str(e),
            )
            if target == "en":
                return self._fallback.translate_to_english(text)
            return self._fallback.translate_to_vietnamese(text)

    def _translate_seq2seq(self, text: str, target: str) -> str:
        """Seq2Seq generation flow (MarianMT, mBART, T5, etc.)."""
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

    def _translate_causal(self, text: str, target: str) -> str:
        """CausalLM generation flow — uses tokenizer.apply_chat_template() for portability."""
        import torch

        if target == "en":
            instruction = (
                "Dịch câu tiếng Việt code-mixed AI/Data Science sau sang tiếng Anh học thuật chuẩn, "
                "giữ nguyên các thuật ngữ kỹ thuật và ý nghĩa chuyên môn."
            )
        else:
            instruction = (
                "Translate the following English AI/Data Science text into standard Vietnamese, "
                "preserving technical terms and specialized meaning."
            )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ]

        # Use tokenizer.apply_chat_template() — handles any model's chat format
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            # Minimal fallback for tokenizers without chat_template
            prompt = f"{instruction}\n\n{text}\n\n"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        # Ensure input tensors are on the same device as the model
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = self._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return decoded.strip()
