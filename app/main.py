        codex/replace-compose.yml-with-docker-compose.yml
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.chat.store import ChatStore, ConversationAccessError, ChatStoreProtocol
from app.chat.summarizer import ConversationSummarizer
from app.ingest import parse_and_chunk
from app.memory.store import MemoryStore
from app.ollama_client import ensure_model, generate
from app.qdrant_client import ensure_collection, search_chunks
from app.rag.context import build_context

load_dotenv()


def _resolve_path(value: str | None, default: Path) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return default


DATA_DIR = _resolve_path(os.getenv("DATA_DIR"), Path("/app/var/data"))
FILES_ROOT = _resolve_path(os.getenv("FILES_ROOT"), DATA_DIR / "files")
LOG_DIR = _resolve_path(os.getenv("LOG_DIR"), DATA_DIR / "logs")
LOG_FILE = _resolve_path(os.getenv("APP_LOG_FILE"), LOG_DIR / "app.log")
_DEFAULT_CHAT_DB_PATH = FILES_ROOT / "db" / "chat_history.sqlite"
_DEFAULT_MEMORY_DB_PATH = FILES_ROOT / "db" / "memory.sqlite"


def _ensure_runtime_directories() -> None:
    for directory in (DATA_DIR, FILES_ROOT, FILES_ROOT / "db", LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)


def _configure_logging() -> None:
    _ensure_runtime_directories()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == LOG_FILE
        for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    for handler in root_logger.handlers:
        handler.setLevel(log_level)


_configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")

CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(_DEFAULT_CHAT_DB_PATH)))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "12"))
CHAT_SUMMARY_TRIGGER = int(os.getenv("CHAT_SUMMARY_TRIGGER", "10"))
MIN_CITATIONS = max(1, int(os.getenv("CHAT_MIN_CITATIONS", "3")))
MAX_CITATIONS = max(MIN_CITATIONS, int(os.getenv("CHAT_MAX_CITATIONS", "5")))
RETRIEVE_TOPK = max(1, int(os.getenv("RETRIEVE_TOPK", "10")))
_configured_rerank = int(os.getenv("RERANK_TOPK", str(RETRIEVE_TOPK)))
RERANK_TOPK = max(1, min(RETRIEVE_TOPK, _configured_rerank))

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


def _get_env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _init_memory_store() -> MemoryStore | None:
    enabled_value = _get_env_value("CHAT_MEMORY_ENABLED", "MEMORY_ENABLED", default="")
    enabled = str(enabled_value or "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    memory_db_path = Path(os.getenv("MEMORY_DB_PATH", str(_DEFAULT_MEMORY_DB_PATH)))
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


app.mem = _init_memory_store()

class ChatIn(BaseModel):
    user_id: str
    message: str
    conversation_id: str | None = None


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


@app.head("/health")
def health_head() -> JSONResponse:
    return health()


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _index_chunks(chunks: Iterable[dict[str, Any]]) -> int:
    items = list(chunks)
    if not items:
        return 0

    try:  # pragma: no cover - optional dependency initialisation
        ensure_collection()
    except Exception:
        logger.exception("Failed to ensure vector store; using fallback index")
        _FALLBACK_INDEX.extend(items)
        return len(items)

"""Entrypoint for the FastAPI application."""
        main

from __future__ import annotations

from app.core.app import create_app

app = create_app()

__all__ = ["app"]
