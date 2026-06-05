"""EduMIND — FastAPI Application Factory.

Run with:
    make api
    # or directly:
    uvicorn edumind.api.main:app --reload --host 0.0.0.0 --port 8000

OpenAPI docs available at:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edumind.api.routers import asr, health, pipeline, rag, translation
from edumind.core.logging import get_logger
from edumind.utils.data_manager import ensure_data_dirs

logger = get_logger(__name__)

_API_VERSION = "v1"
_API_PREFIX = f"/api/{_API_VERSION}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    On startup: ensures data directories exist and pre-warms service singletons
    so the first real request does not incur cold-start latency.
    On shutdown: logs graceful shutdown.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    ensure_data_dirs()
    logger.info("edumind_api_startup", prefix=_API_PREFIX)

    # Pre-warm lightweight singletons (translator is fast; ASR/RAG are lazy)
    try:
        from edumind.api.dependencies import get_translator
        _ = get_translator()
        logger.info("translator_pre_warmed")
    except Exception as exc:
        logger.warning("translator_pre_warm_failed", error=str(exc))

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("edumind_api_shutdown")


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI application."""
    app = FastAPI(
        title="EduMIND REST API",
        description=(
            "**EduMIND** — All-in-One Multimodal Lecture Assistant API.\n\n"
            "Provides REST endpoints for:\n"
            "- 🎙️ **ASR** — Speech-to-text transcription (Whisper)\n"
            "- 🔄 **Translation** — VietMix bilingual translation + CMI analysis\n"
            "- 📚 **RAG** — PDF ingestion, semantic search, LLM Q&A\n"
            "- 🧠 **Pipeline** — End-to-end lecture ingestion workflow\n\n"
            "All endpoints are fully documented below and testable via Swagger UI."
        ),
        version="2.0.0",
        contact={
            "name": "HCMUS Underdogs",
            "url": "https://github.com/khang3004/NLP_playground_hcmus_underdogs",
        },
        license_info={"name": "MIT"},
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow Streamlit (port 8501) and any localhost origin to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",  # Streamlit default
            "http://127.0.0.1:8501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Health routes are mounted at root level (no version prefix)
    app.include_router(health.router)

    # All domain routes are versioned under /api/v1/
    app.include_router(asr.router,         prefix=_API_PREFIX)
    app.include_router(translation.router, prefix=_API_PREFIX)
    app.include_router(rag.router,         prefix=_API_PREFIX)
    app.include_router(pipeline.router,    prefix=_API_PREFIX)

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
