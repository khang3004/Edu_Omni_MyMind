"""EduMIND — Interactive Streamlit Dashboard.

Interactive web interface for the EduMIND Multimodal Lecture Assistant system.
Contains 4 main functional tabs:
    🎙️ Tab 1: Bilingual Note-Taker — Real-time speech transcription & correction.
    🔄 Tab 2: VietMix Translation — Code-mixed machine translation & CMI analytics.
    📚 Tab 3: Anti-Forget RAG — Intelligent layout-aware PDF indexing & vector QA.
    🧠 Tab 4: ALL-IN-ONE Pipeline — Unified end-to-end integration pipeline.

Run:
    streamlit run edumind/app.py
    # or
    make app
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EduMIND — Multimodal Lecture Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Inject Custom Premium Stylesheets (Dark Theme / Glassmorphism)
# ----------------------------------------------------------------------
def inject_custom_css():
    """Injects custom high-end premium CSS styles into the Streamlit session."""
    st.markdown("""
    <style>
    /* ===== Google Font ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ===== Root Variables ===== */
    :root {
        --bg-primary: #0E1117;
        --bg-card: rgba(17, 25, 40, 0.75);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #E6EDF3;
        --text-secondary: #8B949E;
        --accent-purple: #A855F7;
        --accent-cyan: #06B6D4;
        --accent-green: #10B981;
        --accent-orange: #F59E0B;
        --accent-red: #EF4444;
        --gradient-primary: linear-gradient(135deg, #A855F7 0%, #06B6D4 100%);
        --gradient-warm: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
    }

    /* ===== Global ===== */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== Glassmorphic Cards ===== */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.3);
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.1);
    }

    /* ===== Gradient Title ===== */
    .gradient-title {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }

    .subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 4px;
    }

    /* ===== Status Badges ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .status-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-mock {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ===== CMI Gauge Bar ===== */
    .cmi-bar-container {
        width: 100%;
        height: 24px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
        margin: 8px 0;
        border: 1px solid var(--border-glass);
    }

    .cmi-bar-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }

    /* ===== Token Language Chips ===== */
    .token-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 2px 3px;
        transition: transform 0.2s;
    }

    .token-chip:hover {
        transform: translateY(-2px);
    }

    .token-vi {
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .token-en {
        background: rgba(249, 115, 22, 0.2);
        color: #FB923C;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }

    .token-other {
        background: rgba(156, 163, 175, 0.15);
        color: #9CA3AF;
        border: 1px solid rgba(156, 163, 175, 0.2);
    }

    /* ===== Result Cards ===== */
    .result-card {
        background: rgba(17, 25, 40, 0.6);
        border: 1px solid var(--border-glass);
        border-left: 3px solid var(--accent-purple);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    .result-card .score {
        color: var(--accent-cyan);
        font-weight: 700;
        font-size: 0.85rem;
    }

    .result-card .citation {
        color: var(--text-secondary);
        font-style: italic;
        font-size: 0.82rem;
    }

    /* ===== Animated Divider ===== */
    .gradient-divider {
        height: 2px;
        background: var(--gradient-primary);
        border: none;
        border-radius: 1px;
        margin: 16px 0;
        opacity: 0.6;
    }

    /* ===== Sidebar Styling ===== */
    section[data-testid="stSidebar"] {
        background: rgba(14, 17, 23, 0.95);
        border-right: 1px solid var(--border-glass);
    }

    /* ===== Tab Styling ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* ===== Transcript Segment Table ===== */
    .segment-time {
        color: var(--accent-cyan);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* ===== Pipeline Step Indicator ===== */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        border-radius: 8px;
        margin: 4px 0;
    }

    .pipeline-step.active {
        background: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .pipeline-step.done {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .pipeline-step.waiting {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-glass);
        opacity: 0.5;
    }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Global Resource Loaders (Cached globally)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_asr_module():
    """Instantiates and caches the speech recognition module."""
    from edumind.config import settings
    from edumind.modules.speech_processor import CodeSwitchedASR
    return CodeSwitchedASR(model_name=settings.WHISPER_MODEL)


@st.cache_resource(show_spinner=False)
def load_translator_module():
    """Instantiates and caches the bilingual translation module."""
    from edumind.config import settings
    from edumind.modules.vietmix_translator import VietMixTranslator
    return VietMixTranslator(model_name=settings.TRANSLATION_MODEL)


@st.cache_resource(show_spinner=False)
def load_rag_module():
    """Instantiates and caches the layout-aware RAG search module."""
    from edumind.modules.rag_engine import MultimodalRAG
    return MultimodalRAG()


# ----------------------------------------------------------------------
# Sidebar Layout Component
# ----------------------------------------------------------------------
def render_sidebar():
    """Renders Sidebar containing Logo, System status, and Configuration info."""
    with st.sidebar:
        st.markdown(
            '<p class="gradient-title">🧠 EduMIND</p>'
            '<p class="subtitle">All-in-One Multimodal Lecture Assistant</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("### ⚡ System Status")

        from edumind.config import settings

        # Device status
        device_str = str(settings.DEVICE)
        if "cuda" in device_str:
            st.markdown(
                '<span class="status-badge status-ready">🟢 GPU CUDA</span>',
                unsafe_allow_html=True,
            )
        elif "mps" in device_str:
            st.markdown(
                '<span class="status-badge status-ready">🟢 Apple MPS</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-mock">🟡 CPU Mode</span>',
                unsafe_allow_html=True,
            )

        # ASR status
        try:
            asr = load_asr_module()
            if asr.is_ready:
                st.markdown(
                    '<span class="status-badge status-ready">🟢 ASR Ready</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge status-mock">🟡 ASR Mock</span>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                '<span class="status-badge status-error">🔴 ASR Error</span>',
                unsafe_allow_html=True,
            )

        # Translation status
        try:
            translator = load_translator_module()
            mode_label = "Model" if translator.is_model_loaded else "Rule-Based"
            badge_class = "status-ready" if translator.is_model_loaded else "status-mock"
            st.markdown(
                f'<span class="status-badge {badge_class}">🟡 Translator: {mode_label}</span>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<span class="status-badge status-error">🔴 Translator Error</span>',
                unsafe_allow_html=True,
            )

        # RAG status
        try:
            rag = load_rag_module()
            if rag.is_ready:
                info = rag.get_collection_info()
                count = info.get("points_count", 0) or 0
                st.markdown(
                    f'<span class="status-badge status-ready">🟢 RAG: {count} chunks</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-badge status-error">🔴 RAG Not Ready</span>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                '<span class="status-badge status-error">🔴 RAG Error</span>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("### 🔧 Configuration")
        config_summary = settings.summary()
        for key, val in config_summary.items():
            display_key = key.replace("_", " ").title()
            st.text(f"{display_key}: {val}")

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("### ℹ️ About")
        st.markdown(
            "**EduMIND v1.0.0** — MVP\n\n"
            "An integrated multimodal lecture note-taking, translation, "
            "and question-answering assistant.\n\n"
            "🏫 *HCMUS Underdogs Team*"
        )


# ----------------------------------------------------------------------
# Tab 1: 🎙️ Bilingual Note-Taker (ASR Interface)
# ----------------------------------------------------------------------
def render_tab_asr():
    """Renders the Bilingual Note-Taker tab interface."""
    st.markdown("## 🎙️ Bilingual Note-Taker")
    st.markdown(
        "Upload a lecture audio file to auto-transcribe speech and correct "
        " Vietnamese teencode, tech slang, and abbreviations."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    asr = load_asr_module()

    audio_file = st.file_uploader(
        "📁 Upload lecture audio",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        key="asr_upload",
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG",
    )

    col_btn, col_mock = st.columns([1, 1])

    with col_btn:
        transcribe_btn = st.button(
            "🎤 Transcribe Audio",
            key="btn_transcribe",
            use_container_width=True,
            type="primary",
        )

    with col_mock:
        mock_btn = st.button(
            "🎭 Load Demo (Mock Data)",
            key="btn_mock_asr",
            use_container_width=True,
        )

    if transcribe_btn and audio_file is not None:
        with st.spinner("Transcribing audio file..."):
            with tempfile.NamedTemporaryFile(
                suffix=Path(audio_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            result = asr.transcribe(tmp_path)
            _display_transcript_result(asr, result)

    elif mock_btn:
        with st.spinner("Generating simulated lecture transcription..."):
            result = asr._mock_transcribe()
            _display_transcript_result(asr, result)

    elif transcribe_btn and audio_file is None:
        st.warning("⚠️ Please upload an audio file first!")


def _display_transcript_result(asr, result):
    """Displays transcription results side-by-side with teencode corrections."""
    if result.is_mock:
        st.info("🎭 Showing simulated mock data — Whisper model bypassed.")

    raw_text = result.text
    corrected_text = asr.post_process(raw_text)

    col_raw, col_corrected = st.columns(2)

    with col_raw:
        st.markdown("#### 📝 Raw Transcription")
        st.markdown(f'<div class="glass-card">{raw_text}</div>', unsafe_allow_html=True)

    with col_corrected:
        st.markdown("#### ✅ Corrected Transcription")
        st.markdown(
            f'<div class="glass-card" style="border-left: 3px solid var(--accent-green);">'
            f'{corrected_text}</div>',
            unsafe_allow_html=True,
        )

    changes = asr.get_corrections(raw_text, corrected_text)
    if changes:
        with st.expander(f"🔍 Correction Details ({len(changes)} corrections made)", expanded=False):
            for ch in changes:
                st.markdown(
                    f"  `{ch['original']}` → **{ch['corrected']}**"
                )

    if result.segments:
        st.markdown("#### ⏱️ Timestamps")
        seg_data = []
        for seg in result.segments:
            start_fmt = f"{int(seg.start // 60):02d}:{seg.start % 60:05.2f}"
            end_fmt = f"{int(seg.end // 60):02d}:{seg.end % 60:05.2f}"
            seg_data.append({
                "Start": start_fmt,
                "End": end_fmt,
                "Segment Text": seg.text,
            })
        st.dataframe(seg_data, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# Tab 2: 🔄 VietMix Translation (Machine Translation Interface)
# ----------------------------------------------------------------------
def render_tab_translation():
    """Renders the VietMix Translation tab interface."""
    st.markdown("## 🔄 VietMix Translation")
    st.markdown(
        "Enter a code-mixed Vietnamese-English sentence to calculate the Code-Mixing Index (CMI) "
        "and translate it into clean English or clean Vietnamese."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    translator = load_translator_module()

    example_sentences = [
        "Hôm nay mình sẽ discuss về loss function trong deep learning model",
        "Các bạn cần submit bài trước deadline nhé, không là bị trừ điểm",
        "Bây giờ mình bắt đầu explain về attention mechanism trong transformer",
        "Cái learning rate nên set khoảng 2e-5 cho fine-tuning BERT",
        "Mọi người nhớ review lại backpropagation và gradient descent",
    ]

    st.markdown("**💡 Select Example Sentence:**")
    selected_example = st.selectbox(
        "Select an example sentence or type below",
        options=["(Custom Input)"] + example_sentences,
        key="translation_example",
        label_visibility="collapsed",
    )

    default_text = "" if selected_example == "(Custom Input)" else selected_example
    input_text = st.text_area(
        "✏️ Input Code-Mixed Sentence",
        value=default_text,
        height=100,
        key="translation_input",
        placeholder="Type here, e.g., Hôm nay mình sẽ discuss về loss function...",
    )

    if input_text.strip():
        cmi_result = translator.calculate_cmi(input_text)

        st.markdown("### 📊 Code-Mixing Index (CMI)")

        # Determine color gradient for CMI Gauge
        if cmi_result.score < 0.2:
            bar_color = "linear-gradient(90deg, #10B981, #34D399)"
        elif cmi_result.score < 0.5:
            bar_color = "linear-gradient(90deg, #F59E0B, #FBBF24)"
        else:
            bar_color = "linear-gradient(90deg, #EF4444, #F87171)"

        fill_width = max(cmi_result.score * 100, 5)
        st.markdown(
            f'<div class="cmi-bar-container">'
            f'<div class="cmi-bar-fill" style="width: {fill_width}%; background: {bar_color};">'
            f'{cmi_result.score:.2f}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        col_cmi1, col_cmi2, col_cmi3, col_cmi4 = st.columns(4)
        col_cmi1.metric("CMI Score", f"{cmi_result.score:.4f}")
        col_cmi2.metric("🇻🇳 Vietnamese Tokens", cmi_result.vi_count)
        col_cmi3.metric("🇬🇧 English Tokens", cmi_result.en_count)
        col_cmi4.metric("Dominant Language", cmi_result.dominant_language.upper())

        st.markdown("### 🏷️ Token Language Classification")
        chips_html = ""
        for tl in cmi_result.token_labels:
            css_class = f"token-{tl.language}"
            lang_emoji = "🇻🇳" if tl.language == "vi" else ("🇬🇧" if tl.language == "en" else "⚪")
            chips_html += (
                f'<span class="token-chip {css_class}">'
                f'{lang_emoji} {tl.token}</span>'
            )
        st.markdown(chips_html, unsafe_allow_html=True)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("### 🌐 Translation Outputs")
        col_en, col_vi = st.columns(2)

        with col_en:
            st.markdown("#### 🇬🇧 → Standard English")
            en_result = translator.translate_to_english(input_text)
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid var(--accent-orange);">'
                f'{en_result}</div>',
                unsafe_allow_html=True,
            )

        with col_vi:
            st.markdown("#### 🇻🇳 → Standard Vietnamese")
            vi_result = translator.translate_to_vietnamese(input_text)
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid #3B82F6;">'
                f'{vi_result}</div>',
                unsafe_allow_html=True,
            )

        st.caption(f"🔧 Translation execution mode: **{translator.mode}**")


# ----------------------------------------------------------------------
# Tab 3: 📚 Anti-Forget RAG (Document Search & QA Interface)
# ----------------------------------------------------------------------
def render_tab_rag():
    """Renders the Anti-Forget RAG tab interface."""
    st.markdown("## 📚 Anti-Forget RAG")
    st.markdown(
        "Upload PDF documents or lecture slides, store them into the Qdrant vector database, "
        "and retrieve knowledge segments with detailed source citations."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    rag = load_rag_module()

    st.markdown("### 📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Select PDF Slides / Lecture Materials",
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_upload",
    )

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
            "🗑️ Clear Vector Index",
            key="btn_clear_rag",
            use_container_width=True,
        )

    if clear_btn:
        if rag.clear_index():
            st.success("✅ Successfully wiped Qdrant collections!")
        else:
            st.error("❌ Failed to clear database collections.")

    if ingest_btn and uploaded_files:
        total_chunks = 0
        progress_bar = st.progress(0, text="Preprocessing and indexing documents...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(
                (i) / len(uploaded_files),
                text=f"📄 Ingesting: {uploaded_file.name}...",
            )

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = rag.ingest_pdf(tmp_path)

            for chunk in chunks:
                chunk.metadata["source_file"] = uploaded_file.name

            stored = rag.embed_and_store(chunks)
            total_chunks += stored

        progress_bar.progress(1.0, text="✅ Completed successfully!")
        st.success(f"✅ Indexed {total_chunks} text chunks from {len(uploaded_files)} files!")

    info = rag.get_collection_info()
    if info.get("status") == "ready":
        points = info.get("points_count", 0) or 0
        st.info(f"📦 **Qdrant Index:** `{info.get('collection_name')}` — {points} chunks indexed")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Intelligent Semantic QA")

    query_text = st.text_input(
        "Enter question",
        key="rag_query",
        placeholder="Type here, e.g., How does the attention mechanism focus on relevant words?",
    )

    top_k = st.slider("Result Count (top-k)", min_value=1, max_value=20, value=5, key="rag_topk")

    search_btn = st.button(
        "🔍 Search Knowledge Base",
        key="btn_search",
        use_container_width=True,
        type="primary",
    )

    if search_btn and query_text.strip():
        with st.spinner("Searching semantic vectors..."):
            results = rag.query(query_text, top_k=top_k)

        if results:
            answer = rag.generate_answer(query_text, results)
            st.markdown("#### 📋 Synthesized Answer")
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid var(--accent-green);">'
                f'{answer}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### 📄 Matched Source Chunks")
            for i, res in enumerate(results, start=1):
                page = res.metadata.get("page_number", "?")
                source = res.metadata.get("source_file", "Unknown")
                section = res.metadata.get("section_header", "")

                st.markdown(
                    f'<div class="result-card">'
                    f'<span class="score">#{i} — Relevance Score: {res.score:.4f}</span><br>'
                    f'{res.text[:400]}{"..." if len(res.text) > 400 else ""}<br>'
                    f'<span class="citation">📄 Page {page} | {source}'
                    f'{" | §" + section if section and section != "Untitled Section" else ""}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ No relevant matches found. Upload slides and index them first.")

    elif search_btn:
        st.warning("⚠️ Please enter a question first!")


# ----------------------------------------------------------------------
# Tab 4: 🧠 EduMIND ALL-IN-ONE Pipeline (Unified Workflow)
# ----------------------------------------------------------------------
def render_tab_pipeline():
    """Renders the end-to-end integrated Pipeline tab interface."""
    st.markdown("## 🧠 EduMIND ALL-IN-ONE Pipeline")
    st.markdown(
        "Perform end-to-end processing: Upload a lecture PDF slide and an audio recording simultaneously. "
        "Run ASR transcription, clean the transcripts, ingest slide pages, and index both into a single database."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    asr = load_asr_module()
    rag = load_rag_module()

    col_pdf, col_audio = st.columns(2)

    with col_pdf:
        st.markdown("#### 📄 Slide PDF File")
        pdf_file = st.file_uploader(
            "Upload lecture slides",
            type=["pdf"],
            key="pipeline_pdf",
        )

    with col_audio:
        st.markdown("#### 🎙️ Lecture Audio File")
        audio_file = st.file_uploader(
            "Upload lecture audio recording",
            type=["wav", "mp3", "flac", "m4a"],
            key="pipeline_audio",
        )

    col_run, col_mock_run = st.columns(2)

    with col_run:
        run_btn = st.button(
            "🚀 Run Full Pipeline",
            key="btn_run_pipeline",
            use_container_width=True,
            type="primary",
            disabled=not (pdf_file or audio_file),
        )

    with col_mock_run:
        mock_pipeline_btn = st.button(
            "🎭 Run Demo Pipeline (Mock Data)",
            key="btn_mock_pipeline",
            use_container_width=True,
        )

    if run_btn or mock_pipeline_btn:
        progress_bar = st.progress(0, text="Launching pipeline...")

        # Step 1: Speech-to-Text Transcription
        progress_bar.progress(0.1, text="🎤 Step 1/5: Transcribing lecture recording...")
        if mock_pipeline_btn or audio_file is None:
            transcript_result = asr._mock_transcribe()
        else:
            with tempfile.NamedTemporaryFile(
                suffix=Path(audio_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            transcript_result = asr.transcribe(tmp_path)

        # Step 2: Audio Post-processing & Teencode correction
        progress_bar.progress(0.3, text="✏️ Step 2/5: Normalizing text and correcting abbreviations...")
        corrected_transcript = asr.post_process(transcript_result.text)

        # Step 3: Parse PDF slides
        progress_bar.progress(0.5, text="📄 Step 3/5: Parsing PDF slide pages layout...")
        all_chunks = []

        if pdf_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            pdf_chunks = rag.ingest_pdf(tmp_path)
            for chunk in pdf_chunks:
                chunk.metadata["source_type"] = "📄 Slide"
                chunk.metadata["source_file"] = pdf_file.name
            all_chunks.extend(pdf_chunks)

        # Segment and ingest transcripts
        transcript_chunks = rag.ingest_text(
            corrected_transcript,
            source_name="🎙️ Transcript",
            metadata_extra={"source_type": "🎙️ Transcript"},
        )
        all_chunks.extend(transcript_chunks)

        # Step 4: Text Vectorization & Qdrant upsertion
        progress_bar.progress(0.7, text="📐 Step 4/5: Generating embeddings and indexing...")
        stored_count = rag.embed_and_store(all_chunks)

        # Step 5: Completed
        progress_bar.progress(1.0, text="✅ Pipeline complete!")

        st.success(
            f"✅ E2E Pipeline complete! "
            f"Indexed {stored_count} chunks "
            f"({len(transcript_chunks)} from corrected transcripts + "
            f"{len(all_chunks) - len(transcript_chunks)} from PDF pages)."
        )

        with st.expander("📝 Transcription Outputs", expanded=True):
            col_raw, col_fixed = st.columns(2)
            with col_raw:
                st.markdown("**Raw Whisper Output:**")
                st.text(transcript_result.text[:500])
            with col_fixed:
                st.markdown("**Corrected Post-ASR Text:**")
                st.text(corrected_transcript[:500])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 Semantic Integrated Search (Slides + Transcripts)")

    # Unified query input
    pipeline_query = st.text_input(
        "Search indexed slides and lecture transcripts",
        key="pipeline_query",
        placeholder="Type here, e.g., Explain the loss function used in deep learning.",
    )

    pipeline_search_btn = st.button(
        "🔍 Search Integrated Database",
        key="btn_pipeline_search",
        use_container_width=True,
        type="primary",
    )

    if pipeline_search_btn and pipeline_query.strip():
        with st.spinner("Retrieving integrated answers..."):
            results = rag.query(pipeline_query, top_k=5)

        if results:
            answer = rag.generate_answer(pipeline_query, results)
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid var(--accent-green);">'
                f'{answer}</div>',
                unsafe_allow_html=True,
            )

            for i, res in enumerate(results, start=1):
                source_type = res.metadata.get("source_type", "📄")
                page = res.metadata.get("page_number", "?")
                source = res.metadata.get("source_file", "Unknown")

                st.markdown(
                    f'<div class="result-card">'
                    f'<span class="score">#{i} {source_type} — '
                    f'Relevance Score: {res.score:.4f}</span><br>'
                    f'{res.text[:300]}{"..." if len(res.text) > 300 else ""}<br>'
                    f'<span class="citation">Page {page} | {source}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("💡 No data found. Execute the pipeline first to load lecture slides and recordings.")


# ----------------------------------------------------------------------
# Main Application Entry Point
# ----------------------------------------------------------------------
def main():
    """Main execution point for the Streamlit dashboard."""
    inject_custom_css()
    render_sidebar()

    # Main dashboard divided into 4 tabs
    tab_asr, tab_trans, tab_rag, tab_pipeline = st.tabs([
        "🎙️ Bilingual Note-Taker",
        "🔄 VietMix Translation",
        "📚 Anti-Forget RAG",
        "🧠 ALL-IN-ONE Pipeline",
    ])

    with tab_asr:
        render_tab_asr()

    with tab_trans:
        render_tab_translation()

    with tab_rag:
        render_tab_rag()

    with tab_pipeline:
        render_tab_pipeline()


if __name__ == "__main__":
    main()
