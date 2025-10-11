"""Operational endpoints exposing health checks and warmup hooks."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter

from app.core.config import get_version_info
from app.retriever.vector_store import get_vector_store

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/health")
def health() -> dict[str, object]:
    """Lightweight health indicator returning service metadata."""

    return {"status": "ok", "version": get_version_info()}


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
