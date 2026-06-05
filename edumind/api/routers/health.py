"""EduMIND API — Health Check Endpoints.

Routes:
    GET /health           → Lightweight liveness probe (no deps loaded)
    GET /health/detail    → Full readiness probe with per-service status
"""

from __future__ import annotations

from fastapi import APIRouter

from edumind.models.api import DetailedHealthResponse, HealthResponse, ServiceStatus

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Lightweight ping — always returns 200 OK if server is running.",
)
def health_check() -> HealthResponse:
    """Returns minimal liveness status."""
    return HealthResponse(status="ok")


@router.get(
    "/health/detail",
    response_model=DetailedHealthResponse,
    summary="Readiness probe",
    description="Full status check — loads each service and reports readiness.",
)
def health_detail() -> DetailedHealthResponse:
    """Returns detailed per-service health report."""
    from edumind.config import get_settings

    services: list[ServiceStatus] = []
    settings = get_settings()

    # ── ASR ──────────────────────────────────────────────────────────────────
    try:
        from edumind.api.dependencies import get_asr
        asr = get_asr()
        services.append(ServiceStatus(
            name="asr",
            ready=asr.is_ready,
            detail=f"model={settings.WHISPER_MODEL}",
        ))
    except Exception as exc:
        services.append(ServiceStatus(name="asr", ready=False, detail=str(exc)))

    # ── Translator ───────────────────────────────────────────────────────────
    try:
        from edumind.api.dependencies import get_translator
        translator = get_translator()
        services.append(ServiceStatus(
            name="translator",
            ready=True,
            detail=f"mode={translator.mode}",
        ))
    except Exception as exc:
        services.append(ServiceStatus(name="translator", ready=False, detail=str(exc)))

    # ── RAG / Vector Store ───────────────────────────────────────────────────
    try:
        from edumind.api.dependencies import get_rag
        rag = get_rag()
        info = rag.get_collection_info()
        pts = info.get("points_count", 0) or 0
        services.append(ServiceStatus(
            name="rag",
            ready=rag.is_ready,
            detail=f"collection={info.get('collection_name', '?')}, chunks={pts}",
        ))
    except Exception as exc:
        services.append(ServiceStatus(name="rag", ready=False, detail=str(exc)))

    # ── Graph Store ──────────────────────────────────────────────────────────
    try:
        from edumind.core.container import get_graph_store
        gs = get_graph_store()
        g_info = gs.graph_info()
        services.append(ServiceStatus(
            name="graph",
            ready=gs.is_ready,
            detail=f"mode={g_info.get('storage_mode', '?')}",
        ))
    except Exception as exc:
        services.append(ServiceStatus(name="graph", ready=False, detail=str(exc)))

    overall = "ok" if all(s.ready for s in services) else "degraded"

    return DetailedHealthResponse(
        status=overall,
        services=services,
        config=settings.summary(),
    )
