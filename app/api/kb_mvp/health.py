"""Public (no-auth) MVP endpoints: /health and /providers."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
from fastapi import Request
from app.api.kb_auth import auth_status
from app.observability import retrieval_health
from app.services import kb_embeddings, kb_llm, kb_rerank
from .common import public, _store_for


@public.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness probe with LLM, embedder, reranker, auth, KB stats and compliance."""

    import shutil as _shutil
    import sqlite3 as _sqlite3

    store = _store_for(request)
    db_path = Path(store.db_path)
    documents_count = 0
    chunks_count = 0
    distinct_dims = 0
    db_size_bytes = 0
    last_indexed_at: Optional[str] = None
    if db_path.is_file():
        db_size_bytes = db_path.stat().st_size
        try:
            conn = _sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()
                if row:
                    documents_count = int(row[0])
                row = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()
                if row:
                    chunks_count = int(row[0])
                row = conn.execute("SELECT MAX(created_at) FROM kb_documents").fetchone()
                if row and row[0]:
                    last_indexed_at = str(row[0])
                row = conn.execute("SELECT COUNT(DISTINCT dim) FROM kb_chunks").fetchone()
                distinct_dims = int(row[0]) if row else 0
            finally:
                conn.close()
        except _sqlite3.Error:
            pass

    try:
        disk_target = db_path.parent if db_path.parent.is_dir() else Path.cwd()
        disk_free_bytes = _shutil.disk_usage(str(disk_target)).free
    except OSError:
        disk_free_bytes = 0

    compliance_mode = os.environ.get("KB_COMPLIANCE_MODE") or None

    extra: list[tuple] = []
    try:
        # Prefer the store's own embedder to detect degradation; fall back to global.
        _store_embedder = getattr(store, "embedder", None)
        _embedder_name = getattr(_store_embedder, "name", None)
        if _embedder_name is None:
            _embedder_name = kb_embeddings.embedder_status().get("name")
        if _embedder_name == "hash":
            extra.append((retrieval_health.RetrievalReason.HASHING_EMBEDDER, "embedder=hash"))
    except Exception:  # pragma: no cover - never let a probe break health
        pass
    if distinct_dims > 1:
        extra.append(
            (
                retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH,
                f"{distinct_dims} distinct embedding dims present",
            )
        )
    retrieval = retrieval_health.snapshot(extra=tuple(extra))

    return {
        "status": "ok",
        "degraded": retrieval["degraded"],
        "retrieval": retrieval,
        "llm": kb_llm.provider_status(),
        "embedder": kb_embeddings.embedder_status(),
        "reranker": kb_rerank.reranker_status(),
        "auth": auth_status(),
        "kb_stats": {
            "documents_count": documents_count,
            "chunks_count": chunks_count,
            "db_size_bytes": db_size_bytes,
            "disk_free_bytes": disk_free_bytes,
            "last_indexed_at": last_indexed_at,
        },
        "compliance_mode": compliance_mode,
        "compliance_implemented": False,
    }


@public.get("/providers")
def providers() -> dict[str, Any]:
    """Detailed snapshot of LLM providers seen by the service."""

    return kb_llm.provider_status()
