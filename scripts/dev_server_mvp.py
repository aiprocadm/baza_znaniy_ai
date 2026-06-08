"""Minimal FastAPI app exposing only the MVP /api/kb/* router.

Useful for local dev preview without pulling in the full multi-tenant
v1 stack (which requires sqlmodel, qdrant-client, sentence-transformers,
llama-cpp, etc.). The MVP router is intentionally lightweight and only
needs FastAPI + httpx + python-multipart from the runtime requirements.

Also mounts ``data/www/`` as static files so the MVP frontend
(``index.html``) is served on the same origin as the API — no CORS
gymnastics during preview.

Run:

    python -m uvicorn scripts.dev_server_mvp:app --reload --port 8001

Then open:

    http://127.0.0.1:8001/        — MVP frontend (data/www/index.html)
    http://127.0.0.1:8001/docs    — Swagger UI
    http://127.0.0.1:8001/api/kb/health — health probe
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.kb_mvp import router as kb_router

ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/../)
STATIC_DIR = ROOT / "data" / "www"

app = FastAPI(title="kb-mvp-dev", description="MVP-only dev server for /api/kb/*")


@app.on_event("startup")
async def _startup_preflight() -> None:
    """Log one status line (LLM / embedder / mode) at server boot.

    Entirely best-effort: any failure is swallowed to a debug log so it
    can never prevent the server from starting.
    """
    try:
        import os

        from app.services.hardware_probe import probe_system
        from app.services.startup_preflight import log_preflight
        from app.services import kb_llm
        from app.services.kb_embeddings import get_embedder

        probe_system()  # logs a low-RAM warning if applicable
        _prov = kb_llm.select_provider()
        _emb = get_embedder()
        _mode = (
            "api"
            if os.environ.get("KB_LLM_LOCAL_FALLBACK", "").lower() in {"0", "false", "no", "off"}
            else "bundled"
        )
        log_preflight(
            llm_name=getattr(_prov, "name", None),
            llm_model=getattr(_prov, "model", None),
            embedder_name=getattr(_emb, "name", "unknown"),
            mode=_mode,
        )
    except Exception:  # preflight is best-effort; never break startup
        logging.getLogger(__name__).debug("preflight logging skipped", exc_info=True)


# Permissive CORS for local dev — production locks this down via Settings.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# MVP router under /api/kb (same prefix as in production via app.api.router).
app.include_router(kb_router, prefix="/api/kb")

# Serve the MVP frontend at root. ``html=True`` makes index.html the default.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="mvp-ui")


__all__ = ["app"]
