"""
EduMIND — Interactive Streamlit Dashboard
==========================================
Giao diện web tương tác cho hệ thống hỗ trợ giảng dạy đa phương thức EduMIND.

4 Tab chính:
    🎙️ Tab 1: Bilingual Note-Taker — Phiên âm giọng nói song ngữ
    🔄 Tab 2: VietMix Translation — Dịch thuật + CMI
    📚 Tab 3: Anti-Forget RAG — Tìm kiếm tài liệu thông minh
    🧠 Tab 4: ALL-IN-ONE Pipeline — Pipeline tích hợp end-to-end

Chạy:
    streamlit run src/app.py
    # hoặc
    make app
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Cấu hình trang Streamlit
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduMIND — Multimodal Lecture Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Custom CSS — Dark theme với glassmorphism, gradient accents
# ──────────────────────────────────────────────────────────────────────
def inject_custom_css():
    """Inject CSS tùy chỉnh cho giao diện EduMIND premium."""
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


# ──────────────────────────────────────────────────────────────────────
# Cached resource loaders (khởi tạo 1 lần duy nhất)
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_asr_module():
    """Tải module ASR (cached across reruns)."""
    from src.config import settings
    from src.modules.speech_processor import CodeSwitchedASR
    return CodeSwitchedASR(model_name=settings.WHISPER_MODEL)


@st.cache_resource(show_spinner=False)
def load_translator_module():
    """Tải module VietMix Translator (cached across reruns)."""
    from src.config import settings
    from src.modules.vietmix_translator import VietMixTranslator
    return VietMixTranslator(model_name=settings.TRANSLATION_MODEL)


@st.cache_resource(show_spinner=False)
def load_rag_module():
    """Tải module RAG Engine (cached across reruns)."""
    from src.modules.rag_engine import MultimodalRAG
    return MultimodalRAG()


# ──────────────────────────────────────────────────────────────────────
# Sidebar: Logo, System Status, Configuration
# ──────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render sidebar với logo, trạng thái hệ thống, và cấu hình."""
    with st.sidebar:
        # Logo và tên ứng dụng
        st.markdown(
            '<p class="gradient-title">🧠 EduMIND</p>'
            '<p class="subtitle">All-in-One Multimodal Lecture Assistant</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # System Status
        st.markdown("### ⚡ System Status")

        from src.config import settings

        # Device
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

        # Module status
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

        # Configuration display
        st.markdown("### 🔧 Configuration")
        config_summary = settings.summary()
        for key, val in config_summary.items():
            display_key = key.replace("_", " ").title()
            st.text(f"{display_key}: {val}")

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # About section
        st.markdown("### ℹ️ About")
        st.markdown(
            "**EduMIND v1.0.0** — MVP\n\n"
            "Hệ thống hỗ trợ giảng dạy đa phương thức "
            "tích hợp ASR, Dịch thuật, và RAG.\n\n"
            "🏫 *HCMUS Underdogs Team*"
        )


# ──────────────────────────────────────────────────────────────────────
# Tab 1: 🎙️ Bilingual Note-Taker (ASR)
# ──────────────────────────────────────────────────────────────────────
def render_tab_asr():
    """Render tab phiên âm giọng nói song ngữ."""
    st.markdown("## 🎙️ Bilingual Note-Taker")
    st.markdown(
        "Tải lên file audio bài giảng → phiên âm tự động → sửa lỗi teencode/viết tắt."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    asr = load_asr_module()

    # File uploader
    audio_file = st.file_uploader(
        "📁 Tải lên file audio",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        key="asr_upload",
        help="Hỗ trợ: WAV, MP3, FLAC, M4A, OGG",
    )

    col_btn, col_mock = st.columns([1, 1])

    with col_btn:
        transcribe_btn = st.button(
            "🎤 Phiên Âm",
            key="btn_transcribe",
            use_container_width=True,
            type="primary",
        )

    with col_mock:
        mock_btn = st.button(
            "🎭 Demo (Mock Data)",
            key="btn_mock_asr",
            use_container_width=True,
        )

    # Xử lý phiên âm
    if transcribe_btn and audio_file is not None:
        with st.spinner("🔄 Đang phiên âm..."):
            # Lưu file tạm
            with tempfile.NamedTemporaryFile(
                suffix=Path(audio_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            result = asr.transcribe(tmp_path)
            _display_transcript_result(asr, result)

    elif mock_btn:
        with st.spinner("🎭 Đang tạo mock transcript..."):
            result = asr._mock_transcribe()
            _display_transcript_result(asr, result)

    elif transcribe_btn and audio_file is None:
        st.warning("⚠️ Vui lòng tải lên file audio trước khi phiên âm!")


def _display_transcript_result(asr, result):
    """Hiển thị kết quả phiên âm với so sánh trước/sau sửa lỗi."""
    from src.modules.speech_processor import TranscriptResult

    if result.is_mock:
        st.info("🎭 Đây là dữ liệu mock — mô hình Whisper chưa được tải.")

    # So sánh trước/sau sửa lỗi
    raw_text = result.text
    corrected_text = asr.post_process(raw_text)

    col_raw, col_corrected = st.columns(2)

    with col_raw:
        st.markdown("#### 📝 Văn bản gốc (Raw)")
        st.markdown(f'<div class="glass-card">{raw_text}</div>', unsafe_allow_html=True)

    with col_corrected:
        st.markdown("#### ✅ Sau sửa lỗi (Corrected)")
        st.markdown(
            f'<div class="glass-card" style="border-left: 3px solid var(--accent-green);">'
            f'{corrected_text}</div>',
            unsafe_allow_html=True,
        )

    # Hiển thị các thay đổi
    changes = asr.get_corrections(raw_text, corrected_text)
    if changes:
        with st.expander(f"🔍 Chi tiết sửa lỗi ({len(changes)} thay đổi)", expanded=False):
            for ch in changes:
                st.markdown(
                    f"  `{ch['original']}` → **{ch['corrected']}**"
                )

    # Bảng segments với timestamps
    if result.segments:
        st.markdown("#### ⏱️ Timestamps")
        seg_data = []
        for seg in result.segments:
            start_fmt = f"{int(seg.start // 60):02d}:{seg.start % 60:05.2f}"
            end_fmt = f"{int(seg.end // 60):02d}:{seg.end % 60:05.2f}"
            seg_data.append({
                "Bắt đầu": start_fmt,
                "Kết thúc": end_fmt,
                "Nội dung": seg.text,
            })
        st.dataframe(seg_data, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────
# Tab 2: 🔄 VietMix Translation
# ──────────────────────────────────────────────────────────────────────
def render_tab_translation():
    """Render tab dịch thuật Vi-En code-mixed."""
    st.markdown("## 🔄 VietMix Translation")
    st.markdown(
        "Nhập câu pha trộn Việt-Anh → đo mức độ code-mixing (CMI) → dịch sang tiếng Anh/Việt chuẩn."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    translator = load_translator_module()

    # Ví dụ mẫu
    example_sentences = [
        "Hôm nay mình sẽ discuss về loss function trong deep learning model",
        "Các bạn cần submit bài trước deadline nhé, không là bị trừ điểm",
        "Bây giờ mình bắt đầu explain về attention mechanism trong transformer",
        "Cái learning rate nên set khoảng 2e-5 cho fine-tuning BERT",
        "Mọi người nhớ review lại backpropagation và gradient descent",
    ]

    # Selector ví dụ mẫu
    st.markdown("**💡 Câu ví dụ mẫu:**")
    selected_example = st.selectbox(
        "Chọn câu ví dụ hoặc nhập tùy ý bên dưới",
        options=["(Nhập tùy ý)"] + example_sentences,
        key="translation_example",
        label_visibility="collapsed",
    )

    # Text input
    default_text = "" if selected_example == "(Nhập tùy ý)" else selected_example
    input_text = st.text_area(
        "✏️ Nhập câu code-mixed Vi-En",
        value=default_text,
        height=100,
        key="translation_input",
        placeholder="Ví dụ: Hôm nay mình sẽ discuss về loss function...",
    )

    if input_text.strip():
        # Tính CMI
        cmi_result = translator.calculate_cmi(input_text)

        # CMI Gauge
        st.markdown("### 📊 Code-Mixing Index (CMI)")

        # Xác định màu cho gauge bar
        if cmi_result.score < 0.2:
            bar_color = "linear-gradient(90deg, #10B981, #34D399)"
            cmi_label = "Gần đơn ngữ"
        elif cmi_result.score < 0.5:
            bar_color = "linear-gradient(90deg, #F59E0B, #FBBF24)"
            cmi_label = "Pha trộn vừa"
        else:
            bar_color = "linear-gradient(90deg, #EF4444, #F87171)"
            cmi_label = "Pha trộn mạnh"

        fill_width = max(cmi_result.score * 100, 5)  # Tối thiểu 5% để hiện text
        st.markdown(
            f'<div class="cmi-bar-container">'
            f'<div class="cmi-bar-fill" style="width: {fill_width}%; background: {bar_color};">'
            f'{cmi_result.score:.2f}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Thống kê CMI
        col_cmi1, col_cmi2, col_cmi3, col_cmi4 = st.columns(4)
        col_cmi1.metric("CMI Score", f"{cmi_result.score:.4f}")
        col_cmi2.metric("🇻🇳 Tokens Vi", cmi_result.vi_count)
        col_cmi3.metric("🇬🇧 Tokens En", cmi_result.en_count)
        col_cmi4.metric("Dominant", cmi_result.dominant_language.upper())

        # Token-level language labels
        st.markdown("### 🏷️ Token Language Labels")
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

        # Dịch thuật
        st.markdown("### 🌐 Translation Results")
        col_en, col_vi = st.columns(2)

        with col_en:
            st.markdown("#### 🇬🇧 → Clean English")
            en_result = translator.translate_to_english(input_text)
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid var(--accent-orange);">'
                f'{en_result}</div>',
                unsafe_allow_html=True,
            )

        with col_vi:
            st.markdown("#### 🇻🇳 → Clean Vietnamese")
            vi_result = translator.translate_to_vietnamese(input_text)
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid #3B82F6;">'
                f'{vi_result}</div>',
                unsafe_allow_html=True,
            )

        # Mode indicator
        st.caption(f"🔧 Translation mode: **{translator.mode}**")


# ──────────────────────────────────────────────────────────────────────
# Tab 3: 📚 Anti-Forget RAG
# ──────────────────────────────────────────────────────────────────────
def render_tab_rag():
    """Render tab RAG cho tìm kiếm tài liệu thông minh."""
    st.markdown("## 📚 Anti-Forget RAG")
    st.markdown(
        "Tải lên PDF bài giảng → index vào Qdrant → tìm kiếm thông minh với trích dẫn nguồn."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    rag = load_rag_module()

    # --- Upload Section ---
    st.markdown("### 📄 Tải lên tài liệu")

    uploaded_files = st.file_uploader(
        "Chọn file PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="rag_upload",
    )

    col_ingest, col_clear = st.columns([3, 1])

    with col_ingest:
        ingest_btn = st.button(
            "📥 Index tài liệu",
            key="btn_ingest",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        )

    with col_clear:
        clear_btn = st.button(
            "🗑️ Xóa index",
            key="btn_clear_rag",
            use_container_width=True,
        )

    # Xử lý clear
    if clear_btn:
        if rag.clear_index():
            st.success("✅ Đã xóa toàn bộ index!")
        else:
            st.error("❌ Không thể xóa index.")

    # Xử lý ingestion
    if ingest_btn and uploaded_files:
        total_chunks = 0
        progress_bar = st.progress(0, text="Đang xử lý tài liệu...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(
                (i) / len(uploaded_files),
                text=f"📄 Đang xử lý: {uploaded_file.name}...",
            )

            # Lưu file tạm
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Ingest PDF
            chunks = rag.ingest_pdf(tmp_path)

            # Cập nhật metadata nguồn
            for chunk in chunks:
                chunk.metadata["source_file"] = uploaded_file.name

            # Embed và lưu vào Qdrant
            stored = rag.embed_and_store(chunks)
            total_chunks += stored

        progress_bar.progress(1.0, text="✅ Hoàn tất!")
        st.success(f"✅ Đã index {total_chunks} chunks từ {len(uploaded_files)} file!")

    # --- Collection Info ---
    info = rag.get_collection_info()
    if info.get("status") == "ready":
        points = info.get("points_count", 0) or 0
        st.info(f"📦 **Qdrant Collection:** `{info.get('collection_name')}` — {points} chunks đã index")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Search Section ---
    st.markdown("### 🔍 Tìm kiếm thông minh")

    query_text = st.text_input(
        "Nhập câu hỏi",
        key="rag_query",
        placeholder="VD: Attention mechanism hoạt động thế nào?",
    )

    top_k = st.slider("Số kết quả (top-k)", min_value=1, max_value=20, value=5, key="rag_topk")

    search_btn = st.button(
        "🔍 Tìm kiếm",
        key="btn_search",
        use_container_width=True,
        type="primary",
    )

    if search_btn and query_text.strip():
        with st.spinner("🔄 Đang tìm kiếm..."):
            results = rag.query(query_text, top_k=top_k)

        if results:
            # Synthesized answer
            answer = rag.generate_answer(query_text, results)
            st.markdown("#### 📋 Câu trả lời tổng hợp")
            st.markdown(
                f'<div class="glass-card" style="border-left: 3px solid var(--accent-green);">'
                f'{answer}</div>',
                unsafe_allow_html=True,
            )

            # Individual results
            st.markdown("#### 📄 Chi tiết kết quả")
            for i, res in enumerate(results, start=1):
                page = res.metadata.get("page_number", "?")
                source = res.metadata.get("source_file", "Unknown")
                section = res.metadata.get("section_header", "")

                st.markdown(
                    f'<div class="result-card">'
                    f'<span class="score">#{i} — Relevance: {res.score:.4f}</span><br>'
                    f'{res.text[:400]}{"..." if len(res.text) > 400 else ""}<br>'
                    f'<span class="citation">📄 Trang {page} | {source}'
                    f'{" | §" + section if section and section != "Untitled Section" else ""}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ Không tìm thấy kết quả. Hãy thử tải lên tài liệu và index trước.")

    elif search_btn:
        st.warning("⚠️ Vui lòng nhập câu hỏi!")


# ──────────────────────────────────────────────────────────────────────
# Tab 4: 🧠 EduMIND ALL-IN-ONE Pipeline
# ──────────────────────────────────────────────────────────────────────
def render_tab_pipeline():
    """Render tab pipeline tích hợp end-to-end."""
    st.markdown("## 🧠 EduMIND ALL-IN-ONE Pipeline")
    st.markdown(
        "Upload slide PDF + audio bài giảng → Phiên âm → Sửa lỗi → Index → Tìm kiếm tích hợp."
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    asr = load_asr_module()
    rag = load_rag_module()

    # --- Upload Section ---
    col_pdf, col_audio = st.columns(2)

    with col_pdf:
        st.markdown("#### 📄 Slide PDF")
        pdf_file = st.file_uploader(
            "Tải lên slide bài giảng",
            type=["pdf"],
            key="pipeline_pdf",
        )

    with col_audio:
        st.markdown("#### 🎙️ Audio bài giảng")
        audio_file = st.file_uploader(
            "Tải lên audio bài giảng",
            type=["wav", "mp3", "flac", "m4a"],
            key="pipeline_audio",
        )

    # Pipeline control
    col_run, col_mock_run = st.columns(2)

    with col_run:
        run_btn = st.button(
            "🚀 Chạy Pipeline",
            key="btn_run_pipeline",
            use_container_width=True,
            type="primary",
            disabled=not (pdf_file or audio_file),
        )

    with col_mock_run:
        mock_pipeline_btn = st.button(
            "🎭 Demo Pipeline (Mock Data)",
            key="btn_mock_pipeline",
            use_container_width=True,
        )

    if run_btn or mock_pipeline_btn:
        # Step tracking
        steps = [
            ("🎤 Phiên âm audio", "transcribe"),
            ("✏️ Sửa lỗi teencode", "correct"),
            ("📄 Phân tích PDF", "parse_pdf"),
            ("📐 Embedding & Index", "index"),
            ("✅ Sẵn sàng tìm kiếm", "ready"),
        ]

        progress_bar = st.progress(0, text="Bắt đầu pipeline...")

        # Step 1: Transcribe
        progress_bar.progress(0.1, text="🎤 Bước 1/5: Đang phiên âm...")
        if mock_pipeline_btn or audio_file is None:
            transcript_result = asr._mock_transcribe()
        else:
            with tempfile.NamedTemporaryFile(
                suffix=Path(audio_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            transcript_result = asr.transcribe(tmp_path)

        # Step 2: Post-process
        progress_bar.progress(0.3, text="✏️ Bước 2/5: Đang sửa lỗi teencode...")
        corrected_transcript = asr.post_process(transcript_result.text)

        # Step 3: Parse PDF
        progress_bar.progress(0.5, text="📄 Bước 3/5: Đang phân tích PDF...")
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

        # Index transcript
        transcript_chunks = rag.ingest_text(
            corrected_transcript,
            source_name="🎙️ Transcript",
            metadata_extra={"source_type": "🎙️ Transcript"},
        )
        all_chunks.extend(transcript_chunks)

        # Step 4: Embed & Index
        progress_bar.progress(0.7, text="📐 Bước 4/5: Đang embedding & index...")
        stored_count = rag.embed_and_store(all_chunks)

        # Step 5: Complete
        progress_bar.progress(1.0, text="✅ Pipeline hoàn tất!")

        # Display pipeline results
        st.success(
            f"✅ Pipeline hoàn tất! "
            f"Đã index {stored_count} chunks "
            f"({len(transcript_chunks)} từ transcript + "
            f"{len(all_chunks) - len(transcript_chunks)} từ PDF)."
        )

        # Show transcript
        with st.expander("📝 Kết quả phiên âm", expanded=True):
            col_raw, col_fixed = st.columns(2)
            with col_raw:
                st.markdown("**Gốc:**")
                st.text(transcript_result.text[:500])
            with col_fixed:
                st.markdown("**Đã sửa:**")
                st.text(corrected_transcript[:500])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Query section
        st.markdown("### 🔍 Tìm kiếm tích hợp (Slide + Transcript)")

    # Query (luôn hiển thị)
    pipeline_query = st.text_input(
        "Nhập câu hỏi tìm kiếm",
        key="pipeline_query",
        placeholder="VD: Giải thích attention mechanism",
    )

    pipeline_search_btn = st.button(
        "🔍 Tìm kiếm",
        key="btn_pipeline_search",
        use_container_width=True,
        type="primary",
    )

    if pipeline_search_btn and pipeline_query.strip():
        with st.spinner("🔄 Đang tìm kiếm..."):
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
                    f'Relevance: {res.score:.4f}</span><br>'
                    f'{res.text[:300]}{"..." if len(res.text) > 300 else ""}<br>'
                    f'<span class="citation">Trang {page} | {source}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("💡 Chưa có dữ liệu. Hãy chạy pipeline trước khi tìm kiếm.")


# ──────────────────────────────────────────────────────────────────────
# Main Application Entry Point
# ──────────────────────────────────────────────────────────────────────
def main():
    """Entry point chính cho ứng dụng EduMIND Streamlit."""
    # Inject CSS
    inject_custom_css()

    # Render sidebar
    render_sidebar()

    # Main content — 4 Tabs
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
