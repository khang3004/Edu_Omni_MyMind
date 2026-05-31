#!/usr/bin/env bash
# =============================================================================
# EduMIND — Label Studio Environment Bootstrap
# =============================================================================
# PURPOSE  : Configures environment variables for Label Studio local-file-serving
#            and launches both the Label Studio UI (port 8080) and the EduMIND
#            ML Backend (port 9090) in the same terminal session.
#
# USAGE    : bash label_studio_backend/setup_env.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve project root (directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---------------------------------------------------------------------------
# 1. Label Studio — Local File Serving
#    Allows Label Studio tasks to reference files on the host filesystem
#    without uploading them to the Label Studio database.
# ---------------------------------------------------------------------------
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

# Root directories that Label Studio is allowed to serve files from.
# Both audio chunks and PDF slides live inside the project data tree.
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="${PROJECT_ROOT}/data"

# ---------------------------------------------------------------------------
# 2. Label Studio — Storage & Auth
#    Override these or put them in .env before running for production use.
# ---------------------------------------------------------------------------
export LABEL_STUDIO_PORT="${LABEL_STUDIO_PORT:-8080}"
export LABEL_STUDIO_USERNAME="${LABEL_STUDIO_USERNAME:-admin@edumind.local}"
export LABEL_STUDIO_PASSWORD="${LABEL_STUDIO_PASSWORD:-edumind_admin_2024}"

# ---------------------------------------------------------------------------
# 3. EduMIND ML Backend — Server Settings
# ---------------------------------------------------------------------------
export ML_BACKEND_HOST="${ML_BACKEND_HOST:-0.0.0.0}"
export ML_BACKEND_PORT="${ML_BACKEND_PORT:-9090}"

# Path to the growing gold-standard corpus JSONL file (one record per annotation)
export CORPUS_JSONL_PATH="${CORPUS_JSONL_PATH:-${PROJECT_ROOT}/data/processed/corpus.jsonl}"

# ---------------------------------------------------------------------------
# 4. Load existing .env if present (project-level secrets: API keys, etc.)
# ---------------------------------------------------------------------------
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    echo "[setup_env] Loading project .env → ${ENV_FILE}"
    # Export only lines of the form KEY=VALUE (skip comments and blank lines)
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
fi

# ---------------------------------------------------------------------------
# 5. Ensure required data directories exist
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_ROOT}/data/raw/audio_chunks"
mkdir -p "${PROJECT_ROOT}/data/raw/pdf_slides"
mkdir -p "${PROJECT_ROOT}/data/processed"
echo "[setup_env] Data directories verified."

# ---------------------------------------------------------------------------
# 6. Print configuration summary
# ---------------------------------------------------------------------------
echo ""
echo "==========================================================="
echo "  EduMIND × Label Studio — Environment Configuration"
echo "==========================================================="
echo "  Project Root     : ${PROJECT_ROOT}"
echo "  LS File Root     : ${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT}"
echo "  LS Port          : ${LABEL_STUDIO_PORT}"
echo "  ML Backend Port  : ${ML_BACKEND_PORT}"
echo "  Corpus JSONL     : ${CORPUS_JSONL_PATH}"
echo "==========================================================="
echo ""

# ---------------------------------------------------------------------------
# 7. Launch services
#    Run the ML Backend in background; Label Studio in foreground.
#    Both processes are killed when this shell exits via the EXIT trap.
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[setup_env] Shutting down services…"
    # Kill background ML backend if it was started by this script
    if [[ -n "${ML_BACKEND_PID:-}" ]]; then
        kill "${ML_BACKEND_PID}" 2>/dev/null || true
        echo "[setup_env] ML Backend (PID ${ML_BACKEND_PID}) stopped."
    fi
}
trap cleanup EXIT

# --- 7a. Start EduMIND ML Backend -----------------------------------------
echo "[setup_env] Starting EduMIND ML Backend on port ${ML_BACKEND_PORT}…"
label-studio-ml start \
    "${SCRIPT_DIR}" \
    --host "${ML_BACKEND_HOST}" \
    --port "${ML_BACKEND_PORT}" \
    --with \
    CORPUS_JSONL_PATH="${CORPUS_JSONL_PATH}" \
    &
ML_BACKEND_PID=$!
echo "[setup_env] ML Backend PID=${ML_BACKEND_PID}"

# Give the backend a moment to boot before Label Studio starts
sleep 3

# --- 7b. Start Label Studio UI --------------------------------------------
echo "[setup_env] Starting Label Studio on port ${LABEL_STUDIO_PORT}…"
label-studio start \
    --port "${LABEL_STUDIO_PORT}" \
    --username "${LABEL_STUDIO_USERNAME}" \
    --password "${LABEL_STUDIO_PASSWORD}"
