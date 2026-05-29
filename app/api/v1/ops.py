"""Operational endpoints exposing health checks and warmup hooks."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Request

from app.core.config import get_version_info
from app.observability import retrieval_health
from app.retriever.vector_store import get_vector_store

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/health/liveness")
def liveness() -> dict[str, object]:
    return {"status": "alive", "version": get_version_info()}


@router.get("/health/readiness")
def readiness(request: Request) -> dict[str, object]:
    degraded: list[str] = []
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        degraded.append("scheduler_unavailable")
    ingest_worker = getattr(request.app.state, "ingest_worker", None)
    if ingest_worker is None:
        degraded.append("ingest_worker_disabled")
    return {
        "status": "degraded" if degraded else "ready",
        "degraded": degraded,
        "version": get_version_info(),
    }


@router.get("/health/dependencies")
def dependencies() -> dict[str, object]:
    """Dependency checks with degradation semantics."""
    checks: dict[str, str] = {"vector_store": "ok"}
    status = "ok"
    try:
        get_vector_store().ensure_ready()
    except Exception:
        checks["vector_store"] = "degraded"
        status = "degraded"
    snap = retrieval_health.snapshot()
    if snap["degraded"]:
        checks["retrieval"] = snap["severity"]
        status = "degraded"
    else:
        checks["retrieval"] = "ok"
    return {"status": status, "checks": checks, "version": get_version_info()}


@router.post("/warmup")
def warmup() -> dict[str, object]:
    """Ensure the vector store is ready before serving traffic."""

    started = perf_counter()
    vector_store = get_vector_store()
    vector_store.ensure_ready()
    elapsed_ms = int((perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "vectorstore_ready_ms": elapsed_ms,
        "version": get_version_info(),
    }


__all__ = ["router"]
