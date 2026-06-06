#!/usr/bin/env python3
"""EduMIND — Multi-Task Instruction Tuning Corpus Builder.

Technique: Instruction Tuning with Multi-Task Objectives
=========================================================
Each VietMix style sample + AI/DS topic seed → 1 LLM call → 2 training records:

  Task 1 · Semantic Alignment (Cross-Lingual Bridge)
  ───────────────────────────────────────────────────
  instruction : "Dịch câu tiếng Việt code-mixed AI/DS sau sang tiếng Anh học thuật chuẩn."
  input       : Câu tiếng Việt kỹ thuật code-mixed về topic (e.g. "Em đang debug lỗi OOM khi train LLM bằng LoRA…")
  output      : Bản dịch tiếng Anh học thuật sạch (e.g. "I encountered an OOM error while fine-tuning the LLM via LoRA…")
  → Mô hình học: ánh xạ ngữ nghĩa Việt ↔ Anh trong miền AI/DS

  Task 2 · Domain Knowledge Execution (Pedagogical Response)
  ───────────────────────────────────────────────────────────
  instruction : "Bạn là trợ giảng AI/Data Science thông thái. Giải đáp ngắn gọn, chính xác."
  input       : Câu hỏi mở rộng của học viên về cùng topic (e.g. "…có cách nào optimize không anh?")
  output      : Câu trả lời sư phạm bằng tiếng Việt kỹ thuật
  → Mô hình học: trả lời đúng phong cách trợ giảng, dùng kiến thức kỹ thuật

Quota efficiency: 1 API call × 1 topic = 2 training records.
The two tasks are semantically coupled — same concept, same vocabulary.

Usage:
    uv run python scripts/build_multitask_corpus.py [--max-pairs N] [--sleep S]
    uv run python scripts/build_multitask_corpus.py --max-pairs 200
    uv run python scripts/build_multitask_corpus.py --provider groq --model qwen/qwen3-32b

Environment (from .env):
    HF_TOKEN              Hugging Face read token
    CORPUS_LLM_PROVIDER   groq | google | openai  (default: groq)
    CORPUS_LLM_MODEL      Model name               (default: llama-3.1-8b-instant)
    CORPUS_JSONL_PATH     Output JSONL path
"""


from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Task instructions (fixed per-task, become "instruction" in each record)
# ---------------------------------------------------------------------------

TASK1_INSTRUCTION = (
    "Dịch câu tiếng Việt code-mixed AI/Data Science sau sang tiếng Anh học thuật chuẩn, "
    "giữ nguyên các thuật ngữ kỹ thuật và ý nghĩa chuyên môn."
)

TASK2_INSTRUCTION = (
    "Bạn là một trợ giảng AI/Data Science thông thái. "
    "Hãy giải đáp thắc mắc của học viên một cách ngắn gọn, "
    "chính xác bằng văn phong kỹ thuật tự nhiên."
)

# ---------------------------------------------------------------------------
# AI/DS Topic seed bank (48 topics, cycled for diversity)
# ---------------------------------------------------------------------------

TOPIC_SEEDS: list[dict] = [
    # ── Fine-tuning & PEFT ───────────────────────────────────────────────
    {"topic": "LoRA fine-tuning",
     "keywords": ["LoRA", "rank", "adapter", "PEFT", "target_modules", "lora_alpha"]},
    {"topic": "QLoRA với 4-bit quantization",
     "keywords": ["QLoRA", "BitsAndBytes", "NF4", "double_quant", "VRAM", "Unsloth"]},
    {"topic": "Gradient checkpointing để tiết kiệm VRAM",
     "keywords": ["gradient_checkpointing", "OOM", "memory", "backward pass", "activation recomputation"]},
    {"topic": "Lỗi OOM khi fine-tune LLM",
     "keywords": ["OOM", "batch_size", "gradient_accumulation", "bfloat16", "float16"]},
    {"topic": "Lựa chọn learning rate và scheduler",
     "keywords": ["learning_rate", "cosine schedule", "warmup_steps", "lr_scheduler_type", "overfitting"]},
    {"topic": "SFT vs RLHF vs DPO alignment",
     "keywords": ["SFT", "RLHF", "DPO", "reward model", "preference data", "alignment"]},
    {"topic": "Instruction tuning với Alpaca format",
     "keywords": ["instruction", "input", "output", "chat template", "EOS token", "tokenizer"]},
    {"topic": "Merge LoRA weights vào base model",
     "keywords": ["merge_and_unload", "push_to_hub", "GGUF", "llama.cpp", "inference"]},
    # ── Transformer & Architecture ────────────────────────────────────────
    {"topic": "Cơ chế Self-Attention và Multi-Head Attention",
     "keywords": ["Q, K, V", "softmax", "scaled dot-product", "multi-head", "attention mask"]},
    {"topic": "Flash Attention 2 tăng tốc training",
     "keywords": ["FlashAttention", "IO-aware", "tiling", "SRAM", "throughput"]},
    {"topic": "KV Cache và tối ưu inference",
     "keywords": ["KV cache", "past_key_values", "speculative decoding", "prefill", "decode phase"]},
    {"topic": "Mixture of Experts (MoE)",
     "keywords": ["MoE", "router", "expert", "sparse activation", "Mixtral", "load balancing"]},
    {"topic": "Grouped Query Attention (GQA) và MQA",
     "keywords": ["GQA", "MQA", "KV cache", "inference speed", "memory bandwidth"]},
    {"topic": "Positional Encoding vs RoPE vs ALiBi",
     "keywords": ["positional encoding", "RoPE", "ALiBi", "extrapolation", "context length"]},
    # ── RAG & Vector Search ───────────────────────────────────────────────
    {"topic": "RAG pipeline cơ bản",
     "keywords": ["retrieval", "embedding", "vector search", "context injection", "hallucination"]},
    {"topic": "Chunking strategy cho RAG",
     "keywords": ["chunk_size", "overlap", "semantic chunking", "parent-child", "retrieval quality"]},
    {"topic": "Hybrid search BM25 + dense vector",
     "keywords": ["BM25", "dense retrieval", "sparse vector", "RRF", "reciprocal rank fusion"]},
    {"topic": "Re-ranking với Cross-Encoder",
     "keywords": ["cross-encoder", "bi-encoder", "ColBERT", "re-ranking", "latency tradeoff"]},
    {"topic": "Qdrant collection và payload filtering",
     "keywords": ["Qdrant", "collection", "payload", "filter", "cosine similarity", "upsert"]},
    {"topic": "HNSW index và approximate nearest neighbor",
     "keywords": ["HNSW", "ANN", "ef_construction", "m", "recall", "latency"]},
    # ── Embeddings ────────────────────────────────────────────────────────
    {"topic": "So sánh embedding models: BGE, E5, GTE",
     "keywords": ["BGE", "E5", "GTE", "MTEB benchmark", "sentence-transformers", "dimensionality"]},
    {"topic": "Fine-tune embedding model với contrastive loss",
     "keywords": ["contrastive learning", "triplet loss", "in-batch negatives", "hard negatives", "MNRL"]},
    {"topic": "Matryoshka Representation Learning (MRL)",
     "keywords": ["MRL", "adaptive embedding", "truncation", "efficiency", "MTEB"]},
    # ── Data Engineering ─────────────────────────────────────────────────
    {"topic": "Tokenization và vocabulary size",
     "keywords": ["BPE", "WordPiece", "SentencePiece", "vocab_size", "UNK token", "subword"]},
    {"topic": "Data deduplication cho pretraining",
     "keywords": ["MinHash", "LSH", "near-duplicate", "quality filter", "C4", "The Pile"]},
    {"topic": "Class imbalance trong classification",
     "keywords": ["oversampling", "SMOTE", "class_weight", "focal loss", "macro F1"]},
    {"topic": "Data augmentation cho NLP",
     "keywords": ["back-translation", "synonym replacement", "EDA", "mixup", "paraphrase"]},
    # ── Training Tricks ───────────────────────────────────────────────────
    {"topic": "Mixed precision training (fp16/bf16)",
     "keywords": ["fp16", "bf16", "AMP", "GradScaler", "loss scaling", "NaN gradients"]},
    {"topic": "Gradient clipping và exploding gradients",
     "keywords": ["gradient clipping", "max_grad_norm", "gradient explosion", "clip_grad_norm"]},
    {"topic": "DeepSpeed ZeRO và distributed training",
     "keywords": ["DeepSpeed", "ZeRO-3", "FSDP", "model parallelism", "data parallelism", "DDP"]},
    {"topic": "Batch size ảnh hưởng đến convergence",
     "keywords": ["batch_size", "gradient_accumulation", "effective_batch", "generalization"]},
    # ── Evaluation & Metrics ─────────────────────────────────────────────
    {"topic": "Đánh giá LLM bằng BLEU, ROUGE, BERTScore",
     "keywords": ["BLEU", "ROUGE", "BERTScore", "n-gram", "semantic similarity"]},
    {"topic": "Perplexity và cross-entropy loss",
     "keywords": ["perplexity", "cross-entropy", "bits-per-byte", "eval_loss"]},
    {"topic": "Benchmark LLM: MMLU, MT-Bench, HellaSwag",
     "keywords": ["MMLU", "HellaSwag", "TruthfulQA", "MT-Bench", "human eval", "contamination"]},
    # ── MLOps & Deployment ────────────────────────────────────────────────
    {"topic": "GGUF quantization và llama.cpp / Ollama",
     "keywords": ["GGUF", "Q4_K_M", "Q8_0", "llama.cpp", "ollama", "CPU inference"]},
    {"topic": "Serving LLM với vLLM",
     "keywords": ["vLLM", "PagedAttention", "continuous batching", "throughput", "OpenAI API compat"]},
    {"topic": "ONNX export và TensorRT optimization",
     "keywords": ["ONNX", "TensorRT", "fp16", "INT8", "latency", "throughput"]},
    {"topic": "Model versioning với MLflow / W&B",
     "keywords": ["MLflow", "experiment tracking", "run", "artifact", "model registry", "wandb"]},
    {"topic": "FastAPI endpoint cho ML model",
     "keywords": ["FastAPI", "pydantic", "async", "lifespan", "Depends", "health check"]},
    {"topic": "Docker multi-stage build cho ML app",
     "keywords": ["Dockerfile", "multi-stage", "uv", "ENTRYPOINT", "health check", "layer caching"]},
    # ── HuggingFace Ecosystem ─────────────────────────────────────────────
    {"topic": "HuggingFace Trainer vs SFTTrainer (TRL)",
     "keywords": ["Trainer", "SFTTrainer", "TrainingArguments", "dataset_text_field", "packing"]},
    {"topic": "Datasets library: map, filter, streaming",
     "keywords": ["datasets", "map", "filter", "streaming", "num_proc", "batched", "arrow"]},
    {"topic": "torch.compile và triton kernels",
     "keywords": ["torch.compile", "inductor", "triton", "graph mode", "dynamo", "speedup"]},
    {"topic": "Memory profiling PyTorch model",
     "keywords": ["torch.cuda.memory_summary", "memory_reserved", "profiler", "VRAM leak"]},
    {"topic": "Kaggle T4 / TPU training strategies",
     "keywords": ["T4", "TPU", "XLA", "JAX", "torch_xla", "Kaggle quota", "accelerator"]},
    # ── Statistics & Theory ──────────────────────────────────────────────
    {"topic": "Vanishing/Exploding gradient problem",
     "keywords": ["vanishing gradient", "ReLU", "residual connection", "layer norm", "clip_grad"]},
    {"topic": "Regularization: Dropout, Weight Decay, Label Smoothing",
     "keywords": ["dropout", "weight_decay", "label_smoothing", "L2", "overfitting"]},
    {"topic": "Cross-lingual transfer learning",
     "keywords": ["multilingual", "mBERT", "XLM-R", "code-switching", "zero-shot transfer"]},
]


# ---------------------------------------------------------------------------
# Multi-task generation prompt (1 call → 4 fields → 2 records)
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = """\
Bạn là chuyên gia xây dựng dữ liệu huấn luyện đa nhiệm (multi-task) cho AI/DS chatbot.

NHIỆM VỤ: Từ một topic kỹ thuật AI/DS, tạo ra 4 trường để sinh 2 bản ghi huấn luyện:

Trường 1 — vi_sentence:
  Một câu tiếng Việt ngắn (1-2 câu) code-mixed tự nhiên về topic này.
  Phong cách: học viên đang mô tả vấn đề/quan sát của mình.
  Ví dụ: "Em đang debug lỗi OOM khi fine-tune LLM bằng LoRA trên T4, VRAM bị đầy dù batch_size chỉ có 4."

Trường 2 — en_translation:
  Bản dịch tiếng Anh học thuật chuẩn của vi_sentence.
  Giữ nguyên tất cả thuật ngữ kỹ thuật (OOM, LoRA, batch_size...).
  Ví dụ: "I am debugging an Out-Of-Memory (OOM) error while fine-tuning the LLM via LoRA on T4; VRAM is exhausted even with a batch size of 4."

Trường 3 — student_question:
  Câu hỏi mở rộng tự nhiên của học viên về cùng topic (có thể khác vi_sentence một chút).
  Câu hỏi phải cụ thể, có context thực tế.
  Ví dụ: "Anh ơi, em bị OOM khi train LLM bằng LoRA, có cách nào optimize batch_size mà không giảm effective batch size không?"

Trường 4 — ta_answer:
  Câu trả lời sư phạm của trợ giảng: ngắn gọn, kỹ thuật chính xác, tiếng Việt tự nhiên.
  Dùng số thứ tự (1. 2. 3.) hoặc code snippet ngắn nếu cần.
  Ví dụ: "OOM xảy ra khi VRAM không chịu được tensor trong forward/backward. Em thử: 1. Giảm batch_size + tăng gradient_accumulation_steps tương ứng. 2. Bật gradient_checkpointing=True. 3. Cast model sang bfloat16 trước khi train."

QUY TẮC OUTPUT:
- Chỉ trả về JSON object hợp lệ, KHÔNG markdown fence, KHÔNG giải thích
- Đúng 4 keys: "vi_sentence", "en_translation", "student_question", "ta_answer"
- Tất cả values là strings không rỗng
"""


def _make_generation_prompt(vi_style_sample: str, topic: dict) -> str:
    kw_str = ", ".join(topic["keywords"])
    return (
        f'Phong cách viết tự nhiên (học style, KHÔNG copy nội dung):\n"{vi_style_sample[:180]}"\n\n'
        f'Topic kỹ thuật: {topic["topic"]}\n'
        f'Keywords phải xuất hiện trong câu trả lời (ít nhất 2-3): {kw_str}\n\n'
        "Trả về ONLY JSON object với 4 keys."
    )


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env(dotenv_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not dotenv_path.exists():
        raise FileNotFoundError(f".env not found: {dotenv_path}")
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _get(env: dict, key: str, default: str = "") -> str:
    return os.environ.get(key) or env.get(key, default)


def _require(env: dict, key: str) -> str:
    v = _get(env, key)
    if not v:
        raise ValueError(f"Required env var '{key}' missing in .env")
    return v


# ---------------------------------------------------------------------------
# Provider-agnostic LLM caller
# ---------------------------------------------------------------------------

GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {
    "groq":   "llama-3.1-8b-instant",
    "google": "gemini-1.5-pro",
    "openai": "gpt-4o-mini",
}


def _collect_keys(env: dict, prefix: str) -> list[str]:
    keys = [_get(env, f"{prefix}_{i}") for i in range(1, 10)]
    keys = [k for k in keys if k]
    plain = _get(env, prefix)
    if plain and plain not in keys:
        keys.insert(0, plain)
    return keys


def build_llm_caller(env: dict):
    """Returns (callable, model_name). callable(user_msg) → str."""
    provider = _get(env, "CORPUS_LLM_PROVIDER", "groq").lower()
    model    = _get(env, "CORPUS_LLM_MODEL", DEFAULT_MODELS.get(provider, ""))

    if provider == "groq":
        base_url, key_prefix = GROQ_BASE_URL, "GROQ_API_KEY"
    elif provider == "google":
        base_url, key_prefix = GOOGLE_BASE_URL, "GEMINI_API_KEY"
    elif provider == "openai":
        base_url, key_prefix = None, "OPENAI_API_KEY"
    else:
        raise ValueError(f"Unknown provider '{provider}'. Use: groq | google | openai")

    raw_keys = _collect_keys(env, key_prefix)
    if not raw_keys:
        raise ValueError(f"No API keys for provider '{provider}' in .env")

    rotator = itertools.cycle(raw_keys)
    print(f"Provider : {provider.upper()}", flush=True)
    print(f"Model    : {model}", flush=True)
    print(f"API keys : {len(raw_keys)} key(s) in rotation", flush=True)

    def _call(user_msg: str, max_attempts: int = 4) -> str:
        from openai import OpenAI  # type: ignore[import]
        last_error: Exception = RuntimeError("unknown")
        for attempt in range(1, max_attempts + 1):
            key = next(rotator)
            try:
                kwargs: dict = dict(api_key=key)
                if base_url:
                    kwargs["base_url"] = base_url
                client = OpenAI(**kwargs)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": GENERATION_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.75,
                    max_tokens=900,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"  [attempt {attempt}/{max_attempts}] ...{key[-6:]} "
                      f"{type(exc).__name__}: {str(exc)[:100]}", flush=True)
                if attempt < max_attempts:
                    wait = min(2 ** attempt, 30)
                    print(f"  Retrying in {wait}s …", flush=True)
                    time.sleep(wait)
        raise RuntimeError(f"LLM failed after {max_attempts} attempts") from last_error

    return _call, model


# ---------------------------------------------------------------------------
# JSON parser — robust to LLM escape quirks
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("vi_sentence", "en_translation", "student_question", "ta_answer")


def parse_multitask_response(raw: str) -> dict[str, str]:
    """Parse and validate the 4-field LLM response."""
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Fix invalid JSON escape sequences (e.g. \' → \\\')
    raw = re.sub(r"\\(?![\"\\\/bfnrtu])", r"\\\\", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Regex fallback extraction
        data = {}
        for key in _REQUIRED_KEYS:
            m = re.search(rf'"{key}"\s*:\s*"(.*?)"(?=\s*[,}}])', raw, re.DOTALL)
            if m:
                data[key] = m.group(1)

    for key in _REQUIRED_KEYS:
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"Missing or empty key '{key}' in LLM response")

    return {k: str(data[k]) for k in _REQUIRED_KEYS}


# ---------------------------------------------------------------------------
# Convert to 2 training records
# ---------------------------------------------------------------------------

def to_training_records(parsed: dict[str, str]) -> list[dict]:
    """Expand one 4-field response into 2 instruction-tuning records."""
    return [
        # ── Task 1: Semantic Alignment ──────────────────────────────────
        {
            "task":        "semantic_alignment",
            "instruction": TASK1_INSTRUCTION,
            "input":       parsed["vi_sentence"],
            "output":      parsed["en_translation"],
        },
        # ── Task 2: Domain Knowledge Execution ─────────────────────────
        {
            "task":        "domain_qa",
            "instruction": TASK2_INSTRUCTION,
            "input":       parsed["student_question"],
            "output":      parsed["ta_answer"],
        },
    ]


# ---------------------------------------------------------------------------
# HF streaming / JSONL I/O
# ---------------------------------------------------------------------------

def iter_vietmix_rows(hf_token: str) -> Iterable[str]:
    from datasets import load_dataset  # type: ignore[import]
    print("Connecting to HF: razent/vietmix/test (streaming) …", flush=True)
    ds = load_dataset("razent/vietmix", split="test", streaming=True, token=hf_token)
    for row in ds:
        vi = str(row.get("vi") or row.get("text_vi") or "").strip()
        if vi and len(vi) > 20:
            yield vi


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def preview_jsonl(path: Path, count: int = 4) -> None:
    sep = "=" * 72
    print(f"\n{sep}", flush=True)
    print(f"Preview — first {count} records (2 pairs = 4 records if count=4)", flush=True)
    print(sep, flush=True)
    if not path.exists():
        print("<output file not found>", flush=True)
        return
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if idx > count:
                break
            try:
                parsed = json.loads(line)
                task = parsed.get("task", "?")
                label = "🔤 Task 1 — Semantic Alignment" if task == "semantic_alignment" \
                        else "🎓 Task 2 — Domain Q&A"
                print(f"\n{label}", flush=True)
                print(json.dumps(parsed, ensure_ascii=False, indent=2), flush=True)
            except json.JSONDecodeError:
                print(line.rstrip(), flush=True)
    print(sep, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Task Corpus Builder: Semantic Alignment + Domain Q&A",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-pairs",     type=int,   default=None, metavar="N",
                   help="Max LLM calls (each call → 2 records). Omit to stream all VietMix rows.")
    p.add_argument("--sleep-seconds", type=float, default=0.5,  metavar="S",
                   help="Sleep between successful API calls.")
    p.add_argument("--preview",       type=int,   default=4,    metavar="K",
                   help="Records to pretty-print at the end (4 = 2 pairs).")
    p.add_argument("--provider",      type=str,   default=None, metavar="PROVIDER",
                   help="Override CORPUS_LLM_PROVIDER.")
    p.add_argument("--model",         type=str,   default=None, metavar="MODEL",
                   help="Override CORPUS_LLM_MODEL.")
    p.add_argument("--seed",          type=int,   default=42,   metavar="SEED",
                   help="Random seed for topic shuffling.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    root = Path.cwd()
    env  = load_env(root / ".env")

    if args.provider:
        env["CORPUS_LLM_PROVIDER"] = args.provider
    if args.model:
        env["CORPUS_LLM_MODEL"] = args.model

    hf_token    = _require(env, "HF_TOKEN")
    corpus_path = Path(_get(env, "CORPUS_JSONL_PATH", "data/processed/corpus.jsonl")).expanduser()
    if not corpus_path.is_absolute():
        corpus_path = root / corpus_path

    llm_call, model_name = build_llm_caller(env)

    topic_cycle = itertools.cycle(random.sample(TOPIC_SEEDS, len(TOPIC_SEEDS)))

    api_calls  = 0   # LLM calls made
    committed  = 0   # records written (= api_calls × 2)
    skipped    = 0   # failed calls

    print("=" * 72, flush=True)
    print("Technique : Multi-Task Instruction Tuning", flush=True)
    print("Tasks     : [Task 1] Semantic Alignment  +  [Task 2] Domain Q&A", flush=True)
    print(f"Ratio     : 1 API call → 2 training records", flush=True)
    print(f"Topics    : {len(TOPIC_SEEDS)} seed topics (cycled)", flush=True)
    print(f"Target    : {corpus_path}", flush=True)
    print(f"Max pairs : {args.max_pairs or 'all VietMix rows'}", flush=True)
    print(f"Sleep     : {args.sleep_seconds}s / call", flush=True)
    print("=" * 72, flush=True)

    for vi_style in iter_vietmix_rows(hf_token):
        if args.max_pairs and api_calls >= args.max_pairs:
            break

        topic       = next(topic_cycle)
        user_prompt = _make_generation_prompt(vi_style, topic)

        try:
            raw     = llm_call(user_prompt)
            parsed  = parse_multitask_response(raw)
            records = to_training_records(parsed)
            append_jsonl(corpus_path, records)

            api_calls += 1
            committed += len(records)

            topic_short = topic["topic"][:35]
            print(
                f"pair={api_calls:>4d}  records={committed:>5d}  "
                f"[{topic_short}]  "
                f"vi={parsed['vi_sentence'][:45]!r}",
                flush=True,
            )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(
                f"  [SKIP #{skipped}] {topic['topic']!r} "
                f"{type(exc).__name__}: {str(exc)[:100]}",
                flush=True,
            )

    print(
        f"\nDone. api_calls={api_calls}, records_written={committed}, skipped={skipped}",
        flush=True,
    )
    preview_jsonl(corpus_path, count=args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
