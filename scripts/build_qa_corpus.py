#!/usr/bin/env python3
"""EduMIND — AI/DS Q&A Corpus Builder for SLM Fine-tuning.

Architecture: End-to-End Chatbot for AI/Data Science.

Strategy:
  1. Stream VietMix to extract natural Vietnamese code-mixed STYLE patterns.
  2. Pair each style sample with a rotating seed topic from a curated
     AI/DS/ML/NLP knowledge bank.
  3. Prompt the LLM to generate a realistic student-question + expert-answer
     pair in authentic Vietnamese technical register.
  4. Append instruction-tuning records to CORPUS_JSONL_PATH.

Output format (Unsloth / Axolotl / TRL compatible):
    {
      "instruction": "<system prompt — TA persona>",
      "input":  "<student question in Vietnamese, possibly code-mixed>",
      "output": "<expert answer: concise, technically accurate Vietnamese>"
    }

Usage:
    uv run python scripts/build_qa_corpus.py [--max-samples N] [--sleep S]
    uv run python scripts/build_qa_corpus.py --max-samples 200
    uv run python scripts/build_qa_corpus.py                       # stream all

Provider/model override (no code changes needed):
    uv run python scripts/build_qa_corpus.py --provider groq --model qwen/qwen3-32b

Environment (loaded from .env):
    HF_TOKEN              Hugging Face read token
    CORPUS_LLM_PROVIDER   groq | google | openai  (default: groq)
    CORPUS_LLM_MODEL      Model name               (default: llama-3.3-70b-versatile)
    CORPUS_JSONL_PATH     Output path              (default: data/processed/corpus.jsonl)
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
# System persona (fixed — becomes "instruction" field in every record)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "Bạn là một trợ giảng AI/Data Science thông thái. "
    "Hãy giải đáp thắc mắc của học viên một cách ngắn gọn, "
    "chính xác bằng văn phong kỹ thuật tự nhiên."
)

# ---------------------------------------------------------------------------
# AI/DS topic seed bank — ensures diversity across the corpus
# Topics are randomly sampled; each call gets a different one.
# ---------------------------------------------------------------------------

TOPIC_SEEDS: list[dict] = [
    # ── Fine-tuning & PEFT ───────────────────────────────────────────────
    {"topic": "LoRA fine-tuning",
     "keywords": ["LoRA", "rank", "adapter", "PEFT", "merge_weights", "target_modules"]},
    {"topic": "QLoRA với 4-bit quantization",
     "keywords": ["QLoRA", "BitsAndBytes", "NF4", "double_quant", "VRAM", "Unsloth"]},
    {"topic": "Gradient checkpointing để tiết kiệm VRAM",
     "keywords": ["gradient_checkpointing", "OOM", "memory", "backward pass", "activation recomputation"]},
    {"topic": "Lỗi OOM khi fine-tune LLM",
     "keywords": ["OOM", "batch_size", "gradient_accumulation", "bfloat16", "float16"]},
    {"topic": "Lựa chọn learning rate và scheduler",
     "keywords": ["learning_rate", "cosine schedule", "warmup_steps", "lr_scheduler_type", "overfitting"]},
    {"topic": "Early stopping và overfitting",
     "keywords": ["eval_loss", "val_loss", "patience", "load_best_model_at_end", "overfitting"]},
    {"topic": "SFT vs RLHF vs DPO",
     "keywords": ["SFT", "RLHF", "DPO", "reward model", "preference data", "alignment"]},
    {"topic": "Instruction tuning với Alpaca format",
     "keywords": ["instruction", "input", "output", "chat template", "tokenizer", "EOS token"]},
    {"topic": "Merge LoRA weights vào base model",
     "keywords": ["merge_and_unload", "push_to_hub", "GGUF", "llama.cpp", "inference"]},

    # ── Transformer & Architecture ────────────────────────────────────────
    {"topic": "Cơ chế Attention và Self-Attention",
     "keywords": ["Q, K, V", "softmax", "scaled dot-product", "multi-head", "attention mask"]},
    {"topic": "Positional Encoding vs RoPE vs ALiBi",
     "keywords": ["positional encoding", "RoPE", "ALiBi", "extrapolation", "context length"]},
    {"topic": "Grouped Query Attention (GQA) và MQA",
     "keywords": ["GQA", "MQA", "KV cache", "inference speed", "memory bandwidth"]},
    {"topic": "Flash Attention 2 tăng tốc training",
     "keywords": ["FlashAttention", "IO-aware", "tiling", "SRAM", "throughput"]},
    {"topic": "KV Cache và tối ưu inference",
     "keywords": ["KV cache", "past_key_values", "speculative decoding", "prefill", "decode phase"]},
    {"topic": "Mixture of Experts (MoE)",
     "keywords": ["MoE", "router", "expert", "sparse activation", "Mixtral", "load balancing"]},

    # ── RAG & Vector Search ───────────────────────────────────────────────
    {"topic": "RAG pipeline cơ bản",
     "keywords": ["retrieval", "embedding", "vector search", "context injection", "hallucination"]},
    {"topic": "Chunking strategy cho RAG",
     "keywords": ["chunk_size", "overlap", "semantic chunking", "parent-child", "retrieval quality"]},
    {"topic": "Hybrid search BM25 + vector",
     "keywords": ["BM25", "dense retrieval", "sparse vector", "reciprocal rank fusion", "RRF"]},
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

    # ── Data & Preprocessing ─────────────────────────────────────────────
    {"topic": "Tokenization và vocabulary size",
     "keywords": ["BPE", "WordPiece", "SentencePiece", "vocab_size", "UNK token", "subword"]},
    {"topic": "Data deduplication cho pretraining",
     "keywords": ["MinHash", "LSH", "near-duplicate", "quality filter", "C4", "The Pile"]},
    {"topic": "Class imbalance trong classification",
     "keywords": ["oversampling", "SMOTE", "class_weight", "focal loss", "macro F1"]},
    {"topic": "Data augmentation cho NLP",
     "keywords": ["back-translation", "synonym replacement", "EDA", "mixup", "paraphrase"]},
    {"topic": "Train/val/test split đúng cách",
     "keywords": ["stratified split", "data leakage", "time-series split", "cross-validation"]},

    # ── Training Tricks ───────────────────────────────────────────────────
    {"topic": "Mixed precision training (fp16/bf16)",
     "keywords": ["fp16", "bf16", "AMP", "GradScaler", "loss scaling", "NaN gradients"]},
    {"topic": "Gradient clipping và exploding gradients",
     "keywords": ["gradient clipping", "max_grad_norm", "gradient explosion", "NaN", "clip_grad_norm"]},
    {"topic": "DeepSpeed ZeRO và distributed training",
     "keywords": ["DeepSpeed", "ZeRO-3", "FSDP", "model parallelism", "data parallelism", "DDP"]},
    {"topic": "Batch size ảnh hưởng đến convergence",
     "keywords": ["batch_size", "gradient_accumulation", "effective_batch", "generalization", "sharp minima"]},
    {"topic": "Weight initialization strategies",
     "keywords": ["Xavier", "Kaiming", "He initialization", "vanishing gradient", "dying ReLU"]},

    # ── Evaluation & Metrics ─────────────────────────────────────────────
    {"topic": "Đánh giá LLM bằng BLEU, ROUGE, BERTScore",
     "keywords": ["BLEU", "ROUGE", "BERTScore", "n-gram", "semantic similarity", "hallucination"]},
    {"topic": "Perplexity và loss trong language modeling",
     "keywords": ["perplexity", "cross-entropy", "bits-per-byte", "eval_loss", "language modeling"]},
    {"topic": "Vibe-check vs benchmark cho LLM",
     "keywords": ["MMLU", "HellaSwag", "TruthfulQA", "MT-Bench", "human eval", "contamination"]},

    # ── MLOps & Deployment ────────────────────────────────────────────────
    {"topic": "GGUF quantization và llama.cpp",
     "keywords": ["GGUF", "Q4_K_M", "Q8_0", "llama.cpp", "ollama", "CPU inference"]},
    {"topic": "ONNX export và TensorRT optimization",
     "keywords": ["ONNX", "TensorRT", "fp16", "INT8", "latency", "throughput", "triton"]},
    {"topic": "Serving LLM với vLLM",
     "keywords": ["vLLM", "PagedAttention", "continuous batching", "throughput", "OpenAI API"]},
    {"topic": "Model versioning với MLflow",
     "keywords": ["MLflow", "experiment tracking", "run", "artifact", "model registry", "wandb"]},
    {"topic": "FastAPI endpoint cho ML model",
     "keywords": ["FastAPI", "pydantic", "async", "lifespan", "Depends", "health check"]},
    {"topic": "Docker containerize ML app",
     "keywords": ["Dockerfile", "multi-stage", "uv", "ENTRYPOINT", "health check", "layer caching"]},

    # ── Python & Libraries ────────────────────────────────────────────────
    {"topic": "HuggingFace Trainer vs SFTTrainer",
     "keywords": ["Trainer", "SFTTrainer", "TrainingArguments", "dataset_text_field", "packing"]},
    {"topic": "Datasets library: map, filter, shuffle",
     "keywords": ["datasets", "map", "filter", "streaming", "num_proc", "batched", "arrow"]},
    {"topic": "torch.compile và triton kernels",
     "keywords": ["torch.compile", "inductor", "triton", "graph mode", "dynamo", "speedup"]},
    {"topic": "Memory profiling PyTorch model",
     "keywords": ["torch.cuda.memory_summary", "memory_reserved", "profiler", "bottleneck", "VRAM leak"]},
    {"topic": "Kaggle TPU vs GPU training",
     "keywords": ["TPU", "T4", "P100", "XLA", "JAX", "torch_xla", "accelerator"]},
]


# ---------------------------------------------------------------------------
# LLM prompt for Q&A generation
# ---------------------------------------------------------------------------

def _build_generation_prompt(
    vi_style_sample: str,
    topic: dict,
) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for the Q&A generation call."""

    keyword_str = ", ".join(topic["keywords"])

    system = f"""Bạn là chuyên gia xây dựng dữ liệu huấn luyện cho AI/DS chatbot.

Nhiệm vụ: Tạo 1 cặp hỏi-đáp THỰC TẾ giữa học viên và trợ giảng AI/DS.

PHONG CÁCH NGÔN NGỮ (học từ ví dụ bên dưới):
- Câu hỏi phải dùng văn phong tự nhiên như ví dụ — có thể mix tiếng Việt và tiếng Anh kỹ thuật
- Câu hỏi nên mang tính cụ thể, có context thực tế (đang train model, debug, implement...)
- Câu trả lời phải ngắn gọn, súc tích, kỹ thuật chính xác — KHÔNG dài dòng
- Câu trả lời có thể dùng số thứ tự (1. 2. 3.) hoặc code snippet ngắn nếu cần

YÊU CẦU OUTPUT:
- Chỉ trả về JSON object hợp lệ, KHÔNG có markdown fence, KHÔNG có giải thích
- Keys bắt buộc: "question" và "answer"
- "question": câu hỏi của học viên (tiếng Việt, có thể có English tech terms)
- "answer": câu trả lời của trợ giảng (tiếng Việt kỹ thuật, ngắn gọn, có thể kèm code ngắn)

Ví dụ output:
{{"question": "Em bị dính lỗi OOM khi tuning con LLM bằng LoRA, có cách nào optimize batch size không anh?", "answer": "Lỗi OOM xảy ra do VRAM không gánh nổi kích thước tensor trong forward/backward pass. Em thử: 1. Giảm batch_size (8→4→2) + tăng gradient_accumulation_steps tương ứng để giữ effective batch. 2. Bật gradient_checkpointing=True. 3. Cast model sang bfloat16 trước khi train nhé!"}}"""

    user = f"""Ví dụ phong cách viết tự nhiên của người Việt (dùng để học style, KHÔNG copy nội dung):
"{vi_style_sample[:200]}"

Chủ đề kỹ thuật cần tạo Q&A: {topic['topic']}
Keywords phải xuất hiện trong câu trả lời (ít nhất 2-3): {keyword_str}

Hãy tạo 1 cặp hỏi-đáp thực tế. Trả về ONLY JSON object."""

    return system, user


# ---------------------------------------------------------------------------
# .env helpers
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def _get(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or env.get(key, default)


def _require(env: dict[str, str], key: str) -> str:
    value = _get(env, key)
    if not value:
        raise ValueError(f"Required env var '{key}' is missing in .env")
    return value


# ---------------------------------------------------------------------------
# Provider-agnostic LLM caller (OpenAI-compatible for all providers)
# ---------------------------------------------------------------------------

GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {
    "groq":   "llama-3.3-70b-versatile",
    "google": "gemini-1.5-pro",
    "openai": "gpt-4o-mini",
}


def _collect_keys(env: dict[str, str], prefix: str) -> list[str]:
    keys = []
    for i in range(1, 10):
        k = _get(env, f"{prefix}_{i}")
        if k:
            keys.append(k)
    plain = _get(env, prefix)
    if plain and plain not in keys:
        keys.insert(0, plain)
    return keys


def build_llm_caller(env: dict[str, str]):
    """Returns callable: call(system, user) → str (raw LLM response)."""
    provider = _get(env, "CORPUS_LLM_PROVIDER", "groq").lower().strip()
    model    = _get(env, "CORPUS_LLM_MODEL", DEFAULT_MODELS.get(provider, ""))

    if not model:
        raise ValueError(f"CORPUS_LLM_MODEL not set and no default for provider '{provider}'")

    if provider == "groq":
        base_url = GROQ_BASE_URL
        raw_keys = _collect_keys(env, "GROQ_API_KEY")
    elif provider == "google":
        base_url = GOOGLE_BASE_URL
        raw_keys = _collect_keys(env, "GEMINI_API_KEY")
    elif provider == "openai":
        base_url = None
        raw_keys = _collect_keys(env, "OPENAI_API_KEY")
    else:
        raise ValueError(f"Unknown CORPUS_LLM_PROVIDER='{provider}'. Use: groq | google | openai")

    if not raw_keys:
        raise ValueError(f"No API keys found for provider '{provider}' in .env")

    key_rotator = itertools.cycle(raw_keys)
    print(f"Provider : {provider.upper()}", flush=True)
    print(f"Model    : {model}", flush=True)
    print(f"API keys : {len(raw_keys)} key(s) in rotation", flush=True)

    def _call(system_prompt: str, user_message: str, max_attempts: int = 4) -> str:
        from openai import OpenAI  # type: ignore[import]

        last_error: Exception = RuntimeError("unknown")
        for attempt in range(1, max_attempts + 1):
            api_key = next(key_rotator)
            try:
                kwargs: dict = dict(api_key=api_key)
                if base_url:
                    kwargs["base_url"] = base_url
                client = OpenAI(**kwargs)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.8,   # slightly higher for diversity
                    max_tokens=768,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(
                    f"  [attempt {attempt}/{max_attempts}] key=...{api_key[-6:]} "
                    f"{type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
                if attempt < max_attempts:
                    wait = min(2 ** attempt, 30)
                    print(f"  Retrying in {wait}s …", flush=True)
                    time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {max_attempts} attempts") from last_error

    return _call, model


# ---------------------------------------------------------------------------
# Parse & validate LLM JSON response
# ---------------------------------------------------------------------------

def parse_qa_response(raw: str) -> dict[str, str]:
    """Strips fences, parses JSON, validates keys. Handles LLM quirks."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Some models emit `\'` inside JSON strings — replace invalid escapes
    raw = re.sub(r"\\(?![\"\\\/bfnrtu])", r"\\\\", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last-resort: extract with regex if JSON is malformed
        q_match = re.search(r'"question"\s*:\s*"(.*?)"(?=\s*,\s*"answer")', raw, re.DOTALL)
        a_match = re.search(r'"answer"\s*:\s*"(.*?)"(?=\s*[}\n])', raw, re.DOTALL)
        if q_match and a_match:
            data = {"question": q_match.group(1), "answer": a_match.group(1)}
        else:
            raise

    if "question" not in data or "answer" not in data:
        raise ValueError(f"Missing required JSON keys. Got: {list(data.keys())}")
    if not str(data["question"]).strip() or not str(data["answer"]).strip():
        raise ValueError("Empty question or answer in LLM response")
    return {"question": str(data["question"]), "answer": str(data["answer"])}


# ---------------------------------------------------------------------------
# HF streaming
# ---------------------------------------------------------------------------

def iter_vietmix_rows(hf_token: str) -> Iterable[str]:
    """Streams VietMix Vietnamese sentences as style seeds."""
    from datasets import load_dataset  # type: ignore[import]

    print("Connecting to HF: razent/vietmix/test (streaming) …", flush=True)
    ds = load_dataset(
        "razent/vietmix",
        split="test",
        streaming=True,
        token=hf_token,
    )
    for row in ds:
        vi = str(row.get("vi") or row.get("text_vi") or "").strip()
        if vi and len(vi) > 20:
            yield vi


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def preview_jsonl(path: Path, count: int = 3) -> None:
    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print(f"Preview — first {count} record(s) from {path}", flush=True)
    print(sep, flush=True)
    if not path.exists():
        print("<output file not found>", flush=True)
        return
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if idx > count:
                break
            try:
                print(json.dumps(json.loads(line), ensure_ascii=False, indent=2), flush=True)
            except json.JSONDecodeError:
                print(line.rstrip(), flush=True)
            print("-" * 50, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VietMix → AI/DS Q&A instruction-tuning corpus builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-samples",   type=int,   default=None, metavar="N",
                   help="Stop after N committed samples (omit to process all VietMix rows)")
    p.add_argument("--sleep-seconds", type=float, default=0.3,  metavar="S",
                   help="Sleep between successful API calls")
    p.add_argument("--preview",       type=int,   default=3,    metavar="K",
                   help="Records to pretty-print at the end")
    p.add_argument("--provider",      type=str,   default=None, metavar="PROVIDER",
                   help="Override CORPUS_LLM_PROVIDER (groq|google|openai)")
    p.add_argument("--model",         type=str,   default=None, metavar="MODEL",
                   help="Override CORPUS_LLM_MODEL")
    p.add_argument("--seed",          type=int,   default=42,   metavar="SEED",
                   help="Random seed for topic sampling")
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

    processed = skipped = 0

    print("=" * 70, flush=True)
    print("Mode     : AI/DS Q&A Chatbot Corpus (SLM fine-tuning)", flush=True)
    print(f"Topics   : {len(TOPIC_SEEDS)} seed topics (cycled)", flush=True)
    print(f"Target   : {corpus_path}", flush=True)
    print(f"Max      : {args.max_samples or 'all VietMix rows'}", flush=True)
    print(f"Sleep    : {args.sleep_seconds}s between calls", flush=True)
    print("=" * 70, flush=True)

    for vi_style in iter_vietmix_rows(hf_token):
        if args.max_samples and processed >= args.max_samples:
            break

        topic = next(topic_cycle)
        system_prompt, user_message = _build_generation_prompt(vi_style, topic)

        try:
            raw = llm_call(system_prompt, user_message)
            qa  = parse_qa_response(raw)

            append_jsonl(
                corpus_path,
                {
                    "instruction": SYSTEM_INSTRUCTION,
                    "input":       qa["question"],
                    "output":      qa["answer"],
                },
            )
            processed += 1
            topic_short = topic["topic"][:40]
            print(
                f"committed={processed:>4d}  [{topic_short}]  "
                f"Q: {qa['question'][:55]!r}",
                flush=True,
            )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(
                f"  [SKIP #{skipped}] topic={topic['topic']!r} "
                f"{type(exc).__name__}: {str(exc)[:100]}",
                flush=True,
            )

    print(f"\nDone. committed={processed}, skipped={skipped}", flush=True)
    preview_jsonl(corpus_path, count=args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
