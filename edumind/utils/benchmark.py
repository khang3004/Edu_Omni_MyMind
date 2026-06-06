"""EduMIND Performance Benchmark Utility.

Measures processing latency, throughput, and efficiency across all core modules:
  1. Teencode Correction
  2. Language Identification (Syllable LID) & CMI Calculation
  3. Machine Translation (Rule-based vs HuggingFace fallback)
  4. RAG Query Retrieval & Cross-Encoder Re-ranking
"""

from __future__ import annotations

import time

import numpy as np

from edumind.config import get_settings
from edumind.models.chunks import DocumentChunk
from edumind.modules.rag_engine import MultimodalRAG
from edumind.modules.speech_processor import CodeSwitchedASR
from edumind.modules.vietmix_translator import VietMixTranslator

# ANSI color codes for pretty printing
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_benchmark() -> None:
    """Runs latency benchmarks across all subsystems and reports performance statistics."""
    print(f"\n{BOLD}{CYAN}===================================================={RESET}")
    print(f"{BOLD}{CYAN}         EduMIND PERFORMANCE BENCHMARK SUITE       {RESET}")
    print(f"{BOLD}{CYAN}===================================================={RESET}\n")

    settings = get_settings()
    print(f"🖥️  {BOLD}System Details:{RESET}")
    print(f"   - Computation Device: {YELLOW}{settings.DEVICE}{RESET}")
    print(f"   - Whisper Model:      {settings.WHISPER_MODEL}")
    print(f"   - Embedding Model:    {settings.EMBEDDING_MODEL}")
    print(f"   - Translation Model:  {settings.TRANSLATION_MODEL}")
    print(f"   - Qdrant Mode:        {settings.QDRANT_MODE}\n")

    # Initialize modules
    print(f"🔄 {BOLD}Initializing EduMIND Core Modules...{RESET}")
    t_start = time.perf_counter()
    asr = CodeSwitchedASR()
    translator = VietMixTranslator()
    rag = MultimodalRAG()
    t_init = time.perf_counter() - t_start
    print(f"   ✅ Initialized in {GREEN}{t_init:.4f}s{RESET}\n")

    # --- Benchmark 1: Teencode Correction ---
    print(f"📊 {BOLD}[1/4] Benchmarking ASR Teencode Correction...{RESET}")
    test_phrases = [
        "Hôm nay mình ko đi học dc vì bận làm dl và review bài trc nhé",
        "Explain giúp t về loss fn và lr trong deep learning model này với mn",
        "Set bs và epoch thế nào cho BERT và transformer model nói chung?",
        "Mn nhớ submit bài trc dl nhé, ko là ăn con ngỗng",
        "attn và backprop là hai phần rất quan trọng trong NLP và DL",
    ]

    asr_times = []
    # Dry run
    asr.post_process(test_phrases[0])

    for phrase in test_phrases:
        t0 = time.perf_counter()
        asr.post_process(phrase)
        asr_times.append(time.perf_counter() - t0)

    avg_asr = np.mean(asr_times) * 1000
    p95_asr = np.percentile(asr_times, 95) * 1000
    print(f"   - Input sentences:       {len(test_phrases)}")
    print(f"   - Average Latency:      {GREEN}{avg_asr:.2f} ms{RESET}")
    print(f"   - 95th Percentile (p95): {YELLOW}{p95_asr:.2f} ms{RESET}\n")

    # --- Benchmark 2: VietMix Token LID & CMI ---
    print(f"📊 {BOLD}[2/4] Benchmarking Syllable LID & CMI Calculation...{RESET}")
    cmi_phrases = [
        "Hôm nay discuss về attention mechanism trong transformer model nhé",
        "Các bạn cần submit bài trước deadline nhé không là bị trừ điểm",
        "Cái learning rate nên set khoảng 2e-5 cho fine-tuning BERT",
        "Mọi người nhớ review lại backpropagation và gradient descent trước buổi sau",
        "Đầu tiên mình giải thích về activation function trong neural network",
    ]

    cmi_times = []
    # Dry run
    translator.calculate_cmi(cmi_phrases[0])

    for phrase in cmi_phrases:
        t0 = time.perf_counter()
        translator.calculate_cmi(phrase)
        cmi_times.append(time.perf_counter() - t0)

    avg_cmi = np.mean(cmi_times) * 1000
    p95_cmi = np.percentile(cmi_times, 95) * 1000
    print(f"   - Input sentences:       {len(cmi_phrases)}")
    print(f"   - Average Latency:      {GREEN}{avg_cmi:.2f} ms{RESET}")
    print(f"   - p95 Latency:          {YELLOW}{p95_cmi:.2f} ms{RESET}\n")

    # --- Benchmark 3: Translation Speed (Rule-based vs HuggingFace) ---
    print(f"📊 {BOLD}[3/4] Benchmarking Translation Providers...{RESET}")
    trans_text = "Hôm nay mình sẽ discuss về loss function trong deep learning model"

    # Rule-based
    t0 = time.perf_counter()
    translator.translate_to_english(trans_text)
    t_rule = (time.perf_counter() - t0) * 1000
    print(f"   - Rule-based Translation Latency:   {GREEN}{t_rule:.2f} ms{RESET}")

    # HuggingFace (if loaded)
    if translator.is_model_loaded:
        t0 = time.perf_counter()
        translator.translate_to_english(trans_text)
        t_hf = (time.perf_counter() - t0) * 1000
        print(f"   - HuggingFace Neural Model Latency: {YELLOW}{t_hf:.2f} ms{RESET}")
    else:
        print(f"   - HuggingFace Neural Model Latency: {RED}Disabled/Not Loaded (Bypassed){RESET}")
    print()

    # --- Benchmark 4: RAG Retrieval & Re-ranking Latency ---
    print(f"📊 {BOLD}[4/4] Benchmarking Qdrant Search & Cross-Encoder Re-ranking...{RESET}")

    # Create mock chunks to benchmark search
    mock_chunks = [
        DocumentChunk(
            text=f"This is document chunk number {i} talking about deep learning and attention mechanism in transformers.",
            metadata={"source": "test"},
        )
        for i in range(20)
    ]

    # Index chunks
    print("   - Indexing 20 mock chunks to vector store...")
    rag.clear_index()
    rag.embed_and_store(mock_chunks)

    query = "attention mechanism in transformers"

    # Measure Qdrant Search Latency (no reranking)
    t0 = time.perf_counter()
    # bypass rerank for raw comparison
    emb_provider = rag._get_embedding_provider()
    vectorstore = rag._get_vectorstore()
    expanded = rag._expand_query(query)
    q_vec = emb_provider.encode([expanded])[0].tolist()
    _ = vectorstore.search(q_vec, limit=5)
    t_search = (time.perf_counter() - t0) * 1000
    print(f"   - Raw Vector Search (Qdrant) Latency:      {GREEN}{t_search:.2f} ms{RESET}")

    # Measure Cross-Encoder Re-ranking Latency (Cache miss)
    # Clear cache to measure fresh prediction
    rag._reranker._cache.clear()
    t0 = time.perf_counter()
    rag.query(query, top_k=5)
    t_query_fresh = (time.perf_counter() - t0) * 1000
    print(f"   - RAG Query + Re-ranking (Cache Miss):     {YELLOW}{t_query_fresh:.2f} ms{RESET}")

    # Measure Cross-Encoder Re-ranking Latency (Cache hit)
    t0 = time.perf_counter()
    rag.query(query, top_k=5)
    t_query_cached = (time.perf_counter() - t0) * 1000
    print(f"   - RAG Query + Re-ranking (Cache Hit):      {GREEN}{t_query_cached:.2f} ms{RESET}")

    print(f"\n{BOLD}{CYAN}===================================================={RESET}")
    print(f"{BOLD}{GREEN}            BENCHMARK COMPLETED SUCCESSFULLY        {RESET}")
    print(f"{BOLD}{CYAN}===================================================={RESET}\n")


if __name__ == "__main__":
    run_benchmark()
