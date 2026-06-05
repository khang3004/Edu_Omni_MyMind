# =============================================================================
# EduMIND — Main Application Dockerfile
# =============================================================================
# Multi-stage build for the EduMIND API (FastAPI) and UI (Streamlit) services.
# Base image: python:3.10-slim (matches pyproject.toml requires-python ~=3.10)
# Build context: project root (docker build -f Dockerfile .)
#
# Services using this image:
#   - api      → uvicorn edumind.api.main:app  (port 8000)
#   - streamlit → streamlit run edumind/app.py (port 8501)
# =============================================================================

FROM python:3.10-slim AS base

# --- System dependencies ------------------------------------------------------
# ffmpeg    : required by openai-whisper for audio decoding
# libgomp1  : required by sentence-transformers (OpenMP for CPU parallelism)
# poppler-utils : required by docling for PDF rendering
# git       : required by HuggingFace / other VCS dependencies
# curl      : required for health-checking
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        poppler-utils \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Python environment -------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Prevent HuggingFace from polluting /root inside container
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

WORKDIR /workspace

# --- Install uv ---------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- Copy project configurations first for layer caching ----------------------
COPY pyproject.toml uv.lock README.md LICENSE ./

# --- Create dummy package directory for flit & uv ----------------------------
RUN mkdir edumind && touch edumind/__init__.py

# --- Install Python dependencies (without optional extras) -------------------
RUN uv pip install --system .

# --- Copy full source ---------------------------------------------------------
COPY edumind/ ./edumind/

# --- Install EduMIND in editable mode (no dep rebuild) -----------------------
RUN uv pip install --system -e . --no-deps

# --- Data directories (populated via volume mounts in docker-compose) ---------
RUN mkdir -p data/raw/audio_chunks \
             data/raw/pdf_slides \
             data/processed \
             .cache/huggingface \
             .cache/torch

# --- Non-root user for security -----------------------------------------------
RUN useradd -m -u 1000 edumind && chown -R edumind:edumind /workspace
USER edumind

# =============================================================================
# FastAPI Stage
# =============================================================================
FROM base AS api

# Expose FastAPI port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "edumind.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Streamlit Stage
# =============================================================================
FROM base AS streamlit

# Expose Streamlit port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "edumind/app.py", \
     "--server.headless", "true", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
