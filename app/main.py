"""Application entrypoint configuring FastAPI and shared state."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.main import api_router
from app.chat.store import ChatStore, ChatStoreProtocol
from app.chat.summarizer import ConversationSummarizer
from app.core.deps import get_data_dir
from app.memory.store import MemoryStore
from app.ollama_client import generate
from app.services.files import FileStore, IngestQueue

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
_DEFAULT_CHAT_DB_PATH = FILES_ROOT / "db" / "chat_history.sqlite"
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(_DEFAULT_CHAT_DB_PATH)))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "12"))
CHAT_SUMMARY_TRIGGER = int(os.getenv("CHAT_SUMMARY_TRIGGER", "10"))
MIN_CITATIONS = max(1, int(os.getenv("CHAT_MIN_CITATIONS", "3")))
MAX_CITATIONS = max(MIN_CITATIONS, int(os.getenv("CHAT_MAX_CITATIONS", "5")))
RETRIEVE_TOPK = max(1, int(os.getenv("RETRIEVE_TOPK", "10")))
_configured_rerank = int(os.getenv("RERANK_TOPK", str(RETRIEVE_TOPK)))
RERANK_TOPK = max(1, min(RETRIEVE_TOPK, _configured_rerank))


def _get_env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _init_chat_store() -> ChatStoreProtocol:
    backend = (os.getenv("CHAT_DB_BACKEND", "sqlite")).strip().lower()
    if backend in {"postgres", "postgresql"}:
        dsn = _get_env_value("CHAT_DB_DSN", "CHAT_DB_URL", "DATABASE_URL")
        if not dsn:
            raise RuntimeError("CHAT_DB_BACKEND=postgres requires CHAT_DB_DSN/CHAT_DB_URL/DATABASE_URL")
        schema = os.getenv("CHAT_DB_SCHEMA")
        try:
            from app.chat.postgres_store import PostgresChatStore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.exception("Postgres backend requested but unavailable")
            raise RuntimeError("Postgres backend requested but psycopg is missing") from exc
        try:
            return PostgresChatStore(dsn, schema=schema)
        except Exception:
            logger.exception("Failed to initialise Postgres chat store; falling back to SQLite")
    elif backend not in {"sqlite", ""}:
        logger.warning("Unknown CHAT_DB_BACKEND %s; using SQLite", backend)

    db_path = Path(os.getenv("CHAT_DB_PATH", str(CHAT_DB_PATH)))
    return ChatStore(str(db_path))


chat_store = _init_chat_store()
summarizer = ConversationSummarizer(chat_store, generate)


def _init_memory_store() -> MemoryStore | None:
    enabled_value = _get_env_value("CHAT_MEMORY_ENABLED", "MEMORY_ENABLED", default="")
    enabled = str(enabled_value or "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    memory_db_path = Path(
        os.getenv("MEMORY_DB_PATH", str(FILES_ROOT / "db" / "memory.sqlite"))
    )
    ttl_days = int(_get_env_value("CHAT_MEMORY_TTL_DAYS", "MEMORY_TTL_DAYS", default="90") or "90")
    summary_trigger = int(
        _get_env_value(
            "CHAT_SUMMARY_TRIGGER",
            "MEMORY_SUMMARY_TRIGGER",
            default=str(CHAT_SUMMARY_TRIGGER),
        )
        or str(CHAT_SUMMARY_TRIGGER)
    )
    max_tokens = int(
        _get_env_value("CHAT_MEMORY_MAXTOK", "MEMORY_MAX_TOKENS", default="2000")
        or "2000"
    )

    try:
        return MemoryStore(
            db_path=str(memory_db_path),
            ttl_days=ttl_days,
            summary_trigger=summary_trigger,
            max_tokens=max_tokens,
        )
    except Exception:  # pragma: no cover - defensive initialisation
        logger.exception("Failed to initialise memory store")
        return None


app.state.chat_store = chat_store
app.state.summarizer = summarizer
app.state.memory_store = _init_memory_store()
app.state.file_store = FileStore()
app.state.ingest_queue = IngestQueue()
app.state.retrieve_topk = RETRIEVE_TOPK
app.state.rerank_topk = RERANK_TOPK
app.state.min_citations = MIN_CITATIONS
app.state.max_citations = MAX_CITATIONS
app.state.chat_history_limit = CHAT_HISTORY_LIMIT
app.state.chat_summary_trigger = CHAT_SUMMARY_TRIGGER


@app.on_event("startup")
def _ensure_data_dir() -> None:  # pragma: no cover - trivial filesystem side effect
    get_data_dir()


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


@app.head("/health")
def health_head() -> JSONResponse:
    return health()


app.include_router(api_router)


__all__ = ["app"]
