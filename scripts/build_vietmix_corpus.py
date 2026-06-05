#!/usr/bin/env python3
"""EduMIND — VietMix Domain-Adaptation Corpus Builder.

Streams VietMix test split from Hugging Face, rewrites each code-mixed
sentence into a pedagogical Data-Science/ML/AI utterance via a configurable
LLM provider (Groq, Google Gemini, OpenAI), then appends instruction-tuning
records to CORPUS_JSONL_PATH.

Provider selection is fully driven by .env — no code changes needed:

    CORPUS_LLM_PROVIDER = groq         # groq | google | openai
    CORPUS_LLM_MODEL    = llama-3.3-70b-versatile

Usage:
    uv run python scripts/build_vietmix_corpus.py [--max-samples N] [--sleep S]
    uv run python scripts/build_vietmix_corpus.py --max-samples 50
    uv run python scripts/build_vietmix_corpus.py --max-samples 0   # stream all

Environment variables (loaded from .env):
    HF_TOKEN                     Hugging Face read token
    CORPUS_LLM_PROVIDER          LLM provider: groq | google | openai  (default: groq)
    CORPUS_LLM_MODEL             Model name for chosen provider         (default: llama-3.3-70b-versatile)
    GROQ_API_KEY_1..4            Groq API keys (rotated round-robin)
    GEMINI_API_KEY_1..4          Gemini API keys (used if provider=google)
    OPENAI_API_KEY               OpenAI API key (used if provider=openai)
    CORPUS_JSONL_PATH            Output JSONL path (default: data/processed/corpus.jsonl)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME  = "razent/vietmix"
DATASET_SPLIT = "test"

INSTRUCTION = (
    "Translate the following Data Science code-mixed Vietnamese text into "
    "standard academic English."
)

# Domain-mapping vocabulary injected into the system prompt
DOMAIN_MAPPING = """
shopping/mua sắm             → data crawling/scraping
cooking/nấu ăn/meal prep     → data preprocessing / feature engineering
travel/du lịch               → model deployment / serving
fashion/thời trang           → model architecture / design
gym/workout/tập luyện        → model training / fine-tuning
dating/hẹn hò                → hyperparameter tuning / optimization
social media/đăng ảnh/post   → experiment tracking / logging (MLflow, W&B)
party/tiệc                   → evaluation / benchmarking
coffee/cà phê                → compute resources / GPU time
friends/bạn bè               → colleagues / team members / collaborators
restaurant/nhà hàng          → research lab / office
music/nhạc                   → loss function / training metrics
movie/phim                   → dataset / benchmark
"""

SYSTEM_PROMPT = f"""You are a domain-adaptation expert specializing in Vietnamese-English code-mixed text.

Your task: Convert a general lifestyle/social-media Vietnamese-English code-mixed sentence into a realistic, natural utterance that a Vietnamese Data Science / ML / AI engineering student would say.

Rules (MUST follow):
1. Keep the EXACT sentence structure and length of the original.
2. Preserve the original code-mixing ratio (same proportion of Vietnamese vs English words).
3. Map general concepts to DS/ML/AI domain using this vocabulary:
{DOMAIN_MAPPING}
4. Output ONLY a valid JSON object — no markdown, no explanation, no code fences.
5. JSON keys MUST be exactly: "aligned_vi" and "aligned_en"
   - "aligned_vi" : The domain-adapted code-mixed Vietnamese sentence.
   - "aligned_en" : The standard academic English translation of aligned_vi.

Example input:
  vi: "Hôm nay mình đi shopping với bạn bè, mệt quá luôn"
  en: "Today I went shopping with friends, so tired"

Example output:
{{"aligned_vi": "Hôm nay mình run data crawling pipeline với team members, mệt quá luôn", "aligned_en": "Today I ran the data crawling pipeline with my team members, so exhausted"}}
"""


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env(dotenv_path: Path) -> dict[str, str]:
    """Parses a .env file and returns key→value pairs (does not override os.environ)."""
    env: dict[str, str] = {}
    if not dotenv_path.exists():
        raise FileNotFoundError(f".env not found: {dotenv_path}")
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def _get(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or env.get(key, default)


def _require(env: dict[str, str], key: str) -> str:
    value = _get(env, key)
    if not value:
        raise ValueError(f"Required env var '{key}' is missing or empty in .env")
    return value


# ---------------------------------------------------------------------------
# LLM provider factory — provider-agnostic, OpenAI-compatible where possible
# ---------------------------------------------------------------------------

GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini OpenAI compat

DEFAULT_MODELS = {
    "groq":   "llama-3.3-70b-versatile",
    "google": "gemini-1.5-pro",
    "openai": "gpt-4o-mini",
}


def _collect_keys(env: dict[str, str], prefix: str) -> list[str]:
    """Collects PROVIDER_KEY_1..9 from env and returns non-empty values."""
    keys = []
    for i in range(1, 10):
        k = _get(env, f"{prefix}_{i}")
        if k:
            keys.append(k)
    # Also try plain key (e.g. OPENAI_API_KEY)
    plain = _get(env, prefix)
    if plain and plain not in keys:
        keys.insert(0, plain)
    return keys


def build_llm_caller(env: dict[str, str]):
    """Returns a callable:  call(vi_text, en_text) → dict[str, str].

    The callable rotates API keys automatically and uses the OpenAI-compatible
    endpoint for all supported providers.
    """
    provider = _get(env, "CORPUS_LLM_PROVIDER", "groq").lower().strip()
    model    = _get(env, "CORPUS_LLM_MODEL", DEFAULT_MODELS.get(provider, ""))

    if not model:
        raise ValueError(f"CORPUS_LLM_MODEL is not set and no default exists for provider '{provider}'")

    # Resolve base_url and key pool
    if provider == "groq":
        base_url  = GROQ_BASE_URL
        raw_keys  = _collect_keys(env, "GROQ_API_KEY")
        key_label = "GROQ_API_KEY"
    elif provider == "google":
        base_url  = GOOGLE_BASE_URL
        raw_keys  = _collect_keys(env, "GEMINI_API_KEY")
        key_label = "GEMINI_API_KEY"
    elif provider == "openai":
        base_url  = None   # use default openai base
        raw_keys  = _collect_keys(env, "OPENAI_API_KEY")
        key_label = "OPENAI_API_KEY"
    else:
        raise ValueError(
            f"Unknown CORPUS_LLM_PROVIDER='{provider}'. "
            "Supported: groq | google | openai"
        )

    if not raw_keys:
        raise ValueError(
            f"No API keys found for provider '{provider}'. "
            f"Set {key_label}_1 (or {key_label}) in .env"
        )

    key_rotator = itertools.cycle(raw_keys)
    print(f"Provider : {provider.upper()}", flush=True)
    print(f"Model    : {model}", flush=True)
    print(f"API keys : {len(raw_keys)} key(s) in rotation", flush=True)

    def _call(vi_text: str, en_text: str, max_attempts: int = 4) -> dict[str, str]:
        from openai import OpenAI  # type: ignore[import]

        user_content = (
            f'vi: "{vi_text}"\n'
            f'en: "{en_text}"\n\n'
            "Output ONLY the JSON object."
        )

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
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    temperature=0.7,
                    max_tokens=512,
                )
                raw = (resp.choices[0].message.content or "").strip()

                # Strip accidental markdown fences
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
                raw = raw.strip()

                data = json.loads(raw)
                if "aligned_vi" not in data or "aligned_en" not in data:
                    raise ValueError(f"Missing required JSON keys in response: {list(data.keys())}")
                if not data["aligned_vi"].strip() or not data["aligned_en"].strip():
                    raise ValueError("Empty aligned_vi or aligned_en in LLM response")
                return data

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                short_err = str(exc)[:150]
                print(
                    f"  [attempt {attempt}/{max_attempts}] key=...{api_key[-6:]} "
                    f"{type(exc).__name__}: {short_err}",
                    flush=True,
                )
                if attempt < max_attempts:
                    wait = min(2 ** attempt, 30)
                    print(f"  Retrying in {wait}s …", flush=True)
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM alignment failed after {max_attempts} attempts"
        ) from last_error

    return _call


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------

def iter_vietmix_rows(hf_token: str) -> Iterable[dict[str, str]]:
    """Streams VietMix rows lazily from Hugging Face (zero RAM overhead)."""
    from datasets import load_dataset  # type: ignore[import]

    print(f"Connecting to HF: {DATASET_NAME}/{DATASET_SPLIT} (streaming) …", flush=True)
    ds = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
        token=hf_token,
    )
    for row in ds:
        vi = str(row.get("vi") or row.get("text_vi") or "").strip()
        en = str(row.get("en") or row.get("text_en") or "").strip()
        if vi and en:
            yield {"vi": vi, "en": en}


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def preview_jsonl(path: Path, count: int = 3) -> None:
    sep = "=" * 60
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
                parsed = json.loads(line)
                print(json.dumps(parsed, ensure_ascii=False, indent=2), flush=True)
            except json.JSONDecodeError:
                print(line.rstrip(), flush=True)
            print("-" * 40, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VietMix → DS/ML/AI domain-adapted instruction-tuning corpus builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Stop after N committed samples. Omit to stream everything.",
    )
    p.add_argument(
        "--sleep-seconds", type=float, default=0.5, metavar="S",
        help="Seconds to sleep between successful API calls (rate-limit buffer).",
    )
    p.add_argument(
        "--preview", type=int, default=3, metavar="K",
        help="Number of output records to pretty-print at the end.",
    )
    p.add_argument(
        "--provider", type=str, default=None, metavar="PROVIDER",
        help="Override CORPUS_LLM_PROVIDER from .env (groq | google | openai).",
    )
    p.add_argument(
        "--model", type=str, default=None, metavar="MODEL",
        help="Override CORPUS_LLM_MODEL from .env.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    root = Path.cwd()
    env  = load_env(root / ".env")

    # CLI overrides (highest priority)
    if args.provider:
        env["CORPUS_LLM_PROVIDER"] = args.provider
    if args.model:
        env["CORPUS_LLM_MODEL"] = args.model

    hf_token    = _require(env, "HF_TOKEN")
    corpus_path = Path(_get(env, "CORPUS_JSONL_PATH", "data/processed/corpus.jsonl")).expanduser()
    if not corpus_path.is_absolute():
        corpus_path = root / corpus_path

    llm_call = build_llm_caller(env)

    processed = skipped = 0

    print("=" * 60, flush=True)
    print(f"Dataset  : {DATASET_NAME}/{DATASET_SPLIT} (HF streaming)", flush=True)
    print(f"Target   : {corpus_path}", flush=True)
    print(f"Max      : {args.max_samples or 'all'}", flush=True)
    print(f"Sleep    : {args.sleep_seconds}s between calls", flush=True)
    print("Secrets  : loaded from .env — not logged", flush=True)
    print("=" * 60, flush=True)

    for row in iter_vietmix_rows(hf_token):
        if args.max_samples and processed >= args.max_samples:
            break
        try:
            aligned = llm_call(row["vi"], row["en"])
            append_jsonl(
                corpus_path,
                {
                    "instruction": INSTRUCTION,
                    "input":       aligned["aligned_vi"],
                    "output":      aligned["aligned_en"],
                },
            )
            processed += 1
            print(
                f"committed={processed:>4d}  {row['vi'][:65]!r}",
                flush=True,
            )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
        except Exception as exc:  # noqa: BLE001 — keep streaming on bad rows / transient errors
            skipped += 1
            print(f"  [SKIP #{skipped}] {type(exc).__name__}: {str(exc)[:100]}", flush=True)

    print(f"\nDone. committed={processed}, skipped={skipped}", flush=True)
    preview_jsonl(corpus_path, count=args.preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
