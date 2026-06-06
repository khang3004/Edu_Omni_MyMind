"""EduMIND — Interactive Streamlit Dashboard.

Interactive web interface for the EduMIND Multimodal Lecture Assistant.
Contains 4 distinct, non-overlapping functional tabs:

    🎙️ Tab 1: Speech-to-Text    — Audio transcription + teencode correction
    🔄 Tab 2: VietMix Translator — CMI analysis + bidirectional translation
    📚 Tab 3: Knowledge Base     — PDF ingestion (§A) + Semantic Q&A (§B)
    🕸️ Tab 4: Knowledge Graph   — Concept relationship explorer

Run:
    streamlit run edumind/app.py
    # or
    make app

The FastAPI REST layer runs separately on port 8000:
    make api          → launches uvicorn on :8000
    make dev          → launches both API + UI concurrently
"""

from __future__ import annotations

# Suppress HuggingFace transformers verbose warnings (Streamlit file-watcher)
try:
    import transformers

    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass

import gc
from pathlib import Path
import tempfile

import streamlit as st

from edumind.utils.data_manager import ensure_data_dirs, get_raw_dir, get_storage_stats

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduMIND — Multimodal Lecture Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared Premium CSS (Dark Theme / Glassmorphism)
# ──────────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-card: rgba(17, 25, 40, 0.75);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-secondary: #8B949E;
        --accent-purple: #A855F7;
        --accent-cyan: #06B6D4;
        --accent-green: #10B981;
        --accent-orange: #F59E0B;
        --gradient-primary: linear-gradient(135deg, #A855F7 0%, #06B6D4 100%);
    }

    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.3);
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.1);
    }

    .gradient-title {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
    }

    .subtitle { color: var(--text-secondary); font-size: 0.95rem; margin-top: 4px; }

    .gradient-divider {
        height: 2px;
        background: var(--gradient-primary);
        border: none;
        border-radius: 1px;
        margin: 16px 0;
        opacity: 0.6;
    }

    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .status-ready  { background: rgba(16,185,129,.15); color:#10B981; border:1px solid rgba(16,185,129,.3); }
    .status-mock   { background: rgba(245,158,11,.15);  color:#F59E0B; border:1px solid rgba(245,158,11,.3); }
    .status-error  { background: rgba(239,68,68,.15);   color:#EF4444; border:1px solid rgba(239,68,68,.3); }

    .result-card {
        background: rgba(17,25,40,.6);
        border: 1px solid var(--border-glass);
        border-left: 3px solid var(--accent-purple);
        border-radius: 8px; padding: 16px; margin: 8px 0;
    }
    .result-card .score  { color: var(--accent-cyan); font-weight:700; font-size:.85rem; }
    .result-card .citation { color: var(--text-secondary); font-style:italic; font-size:.82rem; }

    .token-chip {
        display:inline-block; padding:3px 10px; border-radius:6px;
        font-size:.82rem; font-weight:500; margin:2px 3px; transition:transform .2s;
    }
    .token-chip:hover { transform: translateY(-2px); }
    .token-vi    { background:rgba(59,130,246,.2);  color:#60A5FA; border:1px solid rgba(59,130,246,.3); }
    .token-en    { background:rgba(249,115,22,.2);  color:#FB923C; border:1px solid rgba(249,115,22,.3); }
    .token-other { background:rgba(156,163,175,.15);color:#9CA3AF; border:1px solid rgba(156,163,175,.2);}

    .cmi-bar-container {
        width:100%; height:24px; background:rgba(255,255,255,.05);
        border-radius:12px; overflow:hidden; margin:8px 0; border:1px solid var(--border-glass);
    }
    .cmi-bar-fill {
        height:100%; border-radius:12px; transition:width .5s ease;
        display:flex; align-items:center; justify-content:center;
        font-size:.72rem; font-weight:700; color:white;
    }

    section[data-testid="stSidebar"] {
        background: rgba(14,17,23,.95);
        border-right: 1px solid var(--border-glass);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cached Resource Loaders
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_asr():
    from edumind.config import get_settings
    from edumind.modules.speech_processor import CodeSwitchedASR

    return CodeSwitchedASR(model_name=get_settings().WHISPER_MODEL)


@st.cache_resource(show_spinner=False)
def _load_translator():
    from edumind.modules.vietmix_translator import VietMixTranslator

    return VietMixTranslator()


@st.cache_resource(show_spinner=False)
def _load_rag():
    from edumind.modules.rag_engine import MultimodalRAG

    return MultimodalRAG()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<p class="gradient-title">🧠 EduMIND</p>'
            '<p class="subtitle">All-in-One Multimodal Lecture Assistant</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("### ⚡ System Status")
        from edumind.config import get_settings

        settings = get_settings()

        # Device
        device = str(settings.DEVICE)
        if "cuda" in device:
            _badge("🟢 GPU CUDA", "status-ready")
        elif "mps" in device:
            _badge("🟢 Apple MPS", "status-ready")
        else:
            _badge("🟡 CPU Mode", "status-mock")

        # ASR
        try:
            asr = _load_asr()
            _badge(
                "🟢 ASR Ready" if asr.is_ready else "🟡 ASR Mock",
                "status-ready" if asr.is_ready else "status-mock",
            )
        except Exception:
            _badge("🔴 ASR Error", "status-error")

        # Translator
        try:
            t = _load_translator()
            label = "Model" if t.is_model_loaded else "Rule-Based"
            cls = "status-ready" if t.is_model_loaded else "status-mock"
            emoji = "🟢" if t.is_model_loaded else "🟡"
            _badge(f"{emoji} Translator: {label}", cls)
        except Exception:
            _badge("🔴 Translator Error", "status-error")

        # RAG
        try:
            rag = _load_rag()
            info = rag.get_collection_info()
            pts = info.get("points_count", 0) or 0
            if rag.is_ready:
                _badge(f"🟢 RAG: {pts} chunks", "status-ready")
            else:
                _badge("🔴 RAG Not Ready", "status-error")
        except Exception:
            _badge("🔴 RAG Error", "status-error")

        # Graph
        try:
            from edumind.core.container import get_graph_store

            gs = get_graph_store()
            g_info = gs.graph_info()
            if gs.is_ready:
                n = g_info.get("nodes_count", 0)
                e = g_info.get("edges_count", 0)
                _badge(f"🟢 Graph: {n}n {e}e", "status-ready")
            else:
                _badge("🟡 Graph Mock (RAM)", "status-mock")
        except Exception:
            _badge("🔴 Graph Error", "status-error")

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔧 Configuration")
        for k, v in settings.summary().items():
            st.text(f"{k.replace('_', ' ').title()}: {v}")

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔌 API")
        st.markdown(
            "FastAPI server (when running):\n\n"
            "- Swagger: [localhost:8000/docs](http://localhost:8000/docs)\n"
            "- Health: [localhost:8000/health](http://localhost:8000/health)\n\n"
            "Start with `make api`"
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("**EduMIND v2.0.0**\n\n🏫 *HCMUS Underdogs Team*")


def _badge(label: str, css_class: str) -> None:
    st.markdown(f'<span class="status-badge {css_class}">{label}</span>', unsafe_allow_html=True)


def _divider() -> None:
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Shared UI Components (extracted to avoid duplication)
# ──────────────────────────────────────────────────────────────────────────────
def _render_result_cards(results) -> None:
    """Renders a list of RetrievedChunk objects as styled result cards."""
    for i, res in enumerate(results, start=1):
        page = res.metadata.get("page_number", "?")
        source = res.metadata.get("source_file", "Unknown")
        section = res.metadata.get("section_header", "")
        source_type = res.metadata.get("source_type", "")

        type_label = f"{source_type} — " if source_type else ""
        section_label = f" | §{section}" if section and section != "Untitled Section" else ""

        st.markdown(
            f'<div class="result-card">'
            f'<span class="score">#{i} {type_label}Relevance: {res.score:.4f}</span><br>'
            f"{res.text[:400]}{'...' if len(res.text) > 400 else ''}<br>"
            f'<span class="citation">📄 Page {page} | {source}{section_label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_cmi_gauge(score: float) -> None:
    """Renders a colour-coded CMI gauge bar."""
    if score < 0.2:
        colour = "linear-gradient(90deg, #10B981, #34D399)"
    elif score < 0.5:
        colour = "linear-gradient(90deg, #F59E0B, #FBBF24)"
    else:
        colour = "linear-gradient(90deg, #EF4444, #F87171)"

    fill = max(score * 100, 5)
    st.markdown(
        f'<div class="cmi-bar-container">'
        f'<div class="cmi-bar-fill" style="width:{fill}%;background:{colour};">'
        f"{score:.2f}</div></div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: 🎙️ Speech-to-Text
# ──────────────────────────────────────────────────────────────────────────────
def _tab_asr() -> None:
    st.markdown("## 🎙️ Speech-to-Text")
    st.markdown(
        "**Input:** Upload a lecture audio file (WAV / MP3 / FLAC / M4A / OGG).  \n"
        "**Output:** Full transcript, teencode-corrected text, timestamped segments."
    )
    _divider()

    asr = _load_asr()

    audio_file = st.file_uploader(
        "📁 Lecture audio file",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        key="asr_upload",
        help="Supported: WAV, MP3, FLAC, M4A, OGG",
    )

    col_real, col_mock = st.columns(2)
    with col_real:
        run_btn = st.button(
            "🎤 Transcribe Audio",
            key="btn_transcribe",
            use_container_width=True,
            type="primary",
            disabled=audio_file is None,
        )
    with col_mock:
        mock_btn = st.button(
            "🎭 Demo (Mock Data)",
            key="btn_mock_asr",
            use_container_width=True,
        )

    if run_btn and audio_file:
        with st.spinner("Transcribing with Whisper..."):
            with tempfile.NamedTemporaryFile(
                suffix=Path(audio_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            result = asr.transcribe(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
        _display_transcript(asr, result)

    elif mock_btn:
        with st.spinner("Generating mock transcript..."):
            result = asr._mock_transcribe()
        _display_transcript(asr, result)

    elif run_btn and not audio_file:
        st.warning("⚠️ Please upload an audio file first.")


def _display_transcript(asr, result) -> None:
    if result.is_mock:
        st.info("🎭 Showing simulated mock data — Whisper model bypassed.")

    raw = result.text
    corrected = asr.post_process(raw)

    col_raw, col_fixed = st.columns(2)
    with col_raw:
        st.markdown("#### 📝 Raw Transcript")
        st.markdown(f'<div class="glass-card">{raw}</div>', unsafe_allow_html=True)
    with col_fixed:
        st.markdown("#### ✅ Corrected Transcript")
        st.markdown(
            f'<div class="glass-card" style="border-left:3px solid var(--accent-green);">'
            f"{corrected}</div>",
            unsafe_allow_html=True,
        )

    changes = asr.get_corrections(raw, corrected)
    if changes:
        with st.expander(f"🔍 {len(changes)} corrections applied", expanded=False):
            for ch in changes:
                st.markdown(f"  `{ch['original']}` → **{ch['corrected']}**")

    if result.segments:
        st.markdown("#### ⏱️ Timestamps")
        seg_data = []
        for seg in result.segments:
            s = f"{int(seg.start // 60):02d}:{seg.start % 60:05.2f}"
            e = f"{int(seg.end // 60):02d}:{seg.end % 60:05.2f}"
            seg_data.append({"Start": s, "End": e, "Text": seg.text})
        st.dataframe(seg_data, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: 🔄 VietMix Translator
# ──────────────────────────────────────────────────────────────────────────────
def _tab_translation() -> None:
    st.markdown("## 🔄 VietMix Translator")
    st.markdown(
        "**Input:** Code-mixed Vietnamese-English sentence.  \n"
        "**Output:** CMI score, per-token language labels, English translation, Vietnamese translation."
    )
    _divider()

    translator = _load_translator()

    EXAMPLES = [
        "Hôm nay mình sẽ discuss về loss function trong deep learning model",
        "Các bạn cần submit bài trước deadline nhé, không là bị trừ điểm",
        "Bây giờ mình bắt đầu explain về attention mechanism trong transformer",
        "Cái learning rate nên set khoảng 2e-5 cho fine-tuning BERT",
        "Mọi người nhớ review lại backpropagation và gradient descent",
    ]

    selected = st.selectbox(
        "💡 Quick examples:",
        ["(Custom input)"] + EXAMPLES,
        key="trans_example",
    )
    default = "" if selected == "(Custom input)" else selected
    input_text = st.text_area(
        "✏️ Input sentence",
        value=default,
        height=100,
        key="trans_input",
        placeholder="Type here, e.g., Hôm nay mình sẽ discuss về loss function...",
    )

    if not input_text.strip():
        st.caption("👆 Enter a sentence above to see analysis and translation.")
        return

    # ── CMI Analysis ─────────────────────────────────────────────────────────
    cmi = translator.calculate_cmi(input_text)

    st.markdown("### 📊 Code-Mixing Index (CMI)")
    _render_cmi_gauge(cmi.score)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CMI Score", f"{cmi.score:.4f}")
    c2.metric("🇻🇳 Vietnamese", cmi.vi_count)
    c3.metric("🇬🇧 English", cmi.en_count)
    c4.metric("Dominant", cmi.dominant_language.upper())

    # ── Token Labels ─────────────────────────────────────────────────────────
    st.markdown("### 🏷️ Token Classification")
    chips = ""
    for tl in cmi.token_labels:
        emoji = "🇻🇳" if tl.language == "vi" else ("🇬🇧" if tl.language == "en" else "⚪")
        chips += f'<span class="token-chip token-{tl.language}">{emoji} {tl.token}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    _divider()

    # ── Translation Outputs ───────────────────────────────────────────────────
    st.markdown("### 🌐 Translation")
    col_en, col_vi = st.columns(2)

    with col_en:
        st.markdown("#### 🇬🇧 → English")
        en_result = translator.translate_to_english(input_text)
        st.markdown(
            f'<div class="glass-card" style="border-left:3px solid var(--accent-orange);">'
            f"{en_result}</div>",
            unsafe_allow_html=True,
        )

    with col_vi:
        st.markdown("#### 🇻🇳 → Vietnamese")
        vi_result = translator.translate_to_vietnamese(input_text)
        st.markdown(
            f'<div class="glass-card" style="border-left:3px solid #3B82F6;">{vi_result}</div>',
            unsafe_allow_html=True,
        )

    st.caption(f"🔧 Translation mode: **{translator.mode}**")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: 📚 Knowledge Base  (§A Document Mgmt  +  §B Semantic Q&A)
# ──────────────────────────────────────────────────────────────────────────────
def _tab_knowledge_base() -> None:
    st.markdown("## 📚 Knowledge Base")
    st.markdown(
        "Manage lecture documents and search your indexed knowledge base.\n\n"
        "- **§A Document Management** — Upload PDFs, ingest transcripts, monitor index\n"
        "- **§B Semantic Q&A** — Ask questions; get cited answers from your slides"
    )
    _divider()

    rag = _load_rag()

    # ── §A: Document Management ───────────────────────────────────────────────
    with st.expander("§A — Document Management", expanded=True):
        st.markdown(
            "**Input:** One or more PDF files.  \n"
            "**Output:** Number of chunks indexed into Qdrant vector store."
        )

        uploaded_files = st.file_uploader(
            "📄 Upload PDF lecture slides / materials",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_upload",
        )

        # Collection status strip
        info = rag.get_collection_info()
        if info.get("status") == "ready":
            pts = info.get("points_count", 0) or 0
            st.info(f"📦 **Qdrant:** `{info.get('collection_name')}` — **{pts}** chunks indexed")
        else:
            st.warning("⚠️ Vector store not ready.")

        col_ingest, col_clear = st.columns([3, 1])
        with col_ingest:
            ingest_btn = st.button(
                "📥 Index Documents",
                key="btn_ingest",
                use_container_width=True,
                type="primary",
                disabled=not uploaded_files,
            )
        with col_clear:
            clear_btn = st.button(
                "🗑️ Clear Index",
                key="btn_clear_rag",
                use_container_width=True,
            )

        if clear_btn:
            if rag.clear_index():
                st.success("✅ Qdrant index cleared!")
            else:
                st.error("❌ Failed to clear index.")
            st.rerun()

        if ingest_btn and uploaded_files:
            total = 0
            bar = st.progress(0, text="Indexing documents...")
            for i, f in enumerate(uploaded_files):
                bar.progress(i / len(uploaded_files), text=f"📄 {f.name}...")
                dest = get_raw_dir() / f.name
                with open(dest, "wb") as fh:
                    fh.write(f.getbuffer())
                chunks = rag.ingest_pdf(dest)
                for c in chunks:
                    c.metadata["source_file"] = f.name
                total += rag.embed_and_store(chunks)
                gc.collect()
                try:
                    import torch

                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
            bar.progress(1.0, text="✅ Done!")
            st.success(f"✅ Indexed **{total}** chunks from **{len(uploaded_files)}** file(s).")
            st.rerun()

    _divider()

    # ── §B: Semantic Q&A ──────────────────────────────────────────────────────
    with st.expander("§B — Semantic Q&A", expanded=True):
        st.markdown(
            "**Input:** A natural-language question.  \n"
            "**Output:** LLM-synthesized answer + ranked source chunks with citations."
        )

        query = st.text_input(
            "🔍 Your question",
            key="rag_query",
            placeholder="e.g., How does the attention mechanism focus on relevant words?",
        )
        top_k = st.slider("Result count (top-k)", 1, 20, 5, key="rag_topk")

        search_btn = st.button(
            "🔍 Search Knowledge Base",
            key="btn_search",
            use_container_width=True,
            type="primary",
        )

        if search_btn and not query.strip():
            st.warning("⚠️ Please enter a question.")
        elif search_btn:
            with st.spinner("Searching..."):
                results = rag.query(query, top_k=top_k)

            if results:
                answer = rag.generate_answer(query, results)
                st.markdown("#### 📋 Synthesized Answer")
                st.markdown(
                    f'<div class="glass-card" style="border-left:3px solid var(--accent-green);">'
                    f"{answer}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("#### 📄 Source Chunks")
                _render_result_cards(results)
            else:
                st.warning("⚠️ No results. Upload and index documents first (§A above).")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: 🕸️ Knowledge Graph
# ──────────────────────────────────────────────────────────────────────────────
def _tab_graph() -> None:
    st.markdown("## 🕸️ Knowledge Graph")
    st.markdown(
        "**Input:** A concept name to search for.  \n"
        "**Output:** Related concepts and their semantic relationships extracted from ingested documents."
    )
    _divider()

    from edumind.core.container import get_graph_store

    gs = get_graph_store()

    stats = get_storage_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Storage Mode", stats.get("qdrant_mode", "local").upper())
    c2.metric("Raw Files", stats.get("raw_files_count", 0))
    c3.metric(
        "DB Size", f"{stats.get('qdrant_size_mb', 0.0) + stats.get('raw_size_mb', 0.0):.2f} MB"
    )
    info = gs.graph_info()
    c4.metric("Graph Mode", info.get("storage_mode", "Mock").replace("_", " ").title())

    _divider()
    st.markdown("### 🔍 Concept Neighborhood Search")
    concept = st.text_input(
        "Concept name (case-sensitive)",
        key="graph_concept",
        placeholder="e.g., Attention, Transformer, Machine Learning",
    )

    if concept.strip():
        neighbors = gs.query_neighborhood(concept)
        if neighbors:
            st.success(f"Found **{len(neighbors)}** relationships for '{concept}'.")
            for r in neighbors:
                st.markdown(
                    f'<div class="result-card" style="border-left:3px solid var(--accent-purple);">'
                    f"<b>{r['source']}</b> —[{r['relationship']}]→ <b>{r['target']}</b> "
                    f"({r.get('target_type', 'Concept')})"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"No connections found for '{concept}'.")

    _divider()
    st.markdown("### 🗺️ Full Graph (Top 50 edges)")

    connections: list[dict] = []
    try:
        if hasattr(gs, "_driver") and gs._driver is not None:
            cypher = (
                "MATCH (a:Concept)-[r]->(b:Concept) "
                "RETURN a.name AS source, type(r) AS relationship, b.name AS target LIMIT 50"
            )
            with gs._driver.session() as session:
                connections = [
                    {
                        "Source": r["source"],
                        "Relationship": r["relationship"],
                        "Target": r["target"],
                    }
                    for r in session.run(cypher)
                ]
        elif hasattr(gs, "_edges"):
            connections = [
                {"Source": e["source"], "Relationship": e["type"], "Target": e["target"]}
                for e in gs._edges[:50]
            ]
    except Exception as exc:
        st.error(f"Error loading graph: {exc}")

    if connections:
        st.dataframe(connections, use_container_width=True)
    else:
        st.info("💡 Graph is empty. Ingest PDFs in the Knowledge Base tab to populate it.")

    if st.button("🗑️ Wipe Graph Store", key="btn_clear_graph", use_container_width=True):
        if gs.clear_graph():
            st.success("✅ Graph wiped!")
            st.rerun()
        else:
            st.error("❌ Failed to clear graph.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main execution point for the Streamlit dashboard."""
    ensure_data_dirs()
    _inject_css()
    _render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎙️ Speech-to-Text",
            "🔄 VietMix Translator",
            "📚 Knowledge Base",
            "🕸️ Knowledge Graph",
        ]
    )

    with tab1:
        _tab_asr()
    with tab2:
        _tab_translation()
    with tab3:
        _tab_knowledge_base()
    with tab4:
        _tab_graph()


if __name__ == "__main__":
    main()
