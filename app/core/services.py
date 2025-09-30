"""Service initialisation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from app.chat.store import ChatStore, ChatStoreProtocol
from app.memory.store import MemoryStore

from .config import Settings

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - defensive IO handling
        logger.exception("Failed to create directory for %s", path)


def init_chat_store(settings: Settings) -> ChatStoreProtocol:
    backend = settings.chat_db_backend

    if backend in {"postgres", "postgresql"}:
        dsn = settings.chat_db_dsn
        if not dsn:
            raise RuntimeError(
                "CHAT_DB_BACKEND=postgres requires CHAT_DB_DSN/CHAT_DB_URL/DATABASE_URL"
            )
        schema = settings.chat_db_schema
        try:
            from app.chat.postgres_store import PostgresChatStore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.exception("Postgres backend requested but unavailable")
            raise RuntimeError("Postgres backend requested but psycopg is missing") from exc
        try:
            return PostgresChatStore(dsn, schema=schema)
        except Exception:  # pragma: no cover - fallback to SQLite on failure
            logger.exception("Failed to initialise Postgres chat store; falling back to SQLite")
    elif backend not in {"sqlite", ""}:
        logger.warning("Unknown CHAT_DB_BACKEND %s; using SQLite", backend)

    db_path = settings.chat_db_path_resolved
    _ensure_parent_dir(db_path)
    return ChatStore(str(db_path))


def init_memory_store(settings: Settings) -> MemoryStore | None:
    if not settings.chat_memory_enabled:
        return None

    memory_db_path = settings.memory_db_path_resolved
    _ensure_parent_dir(memory_db_path)

    try:
        return MemoryStore(
            db_path=str(memory_db_path),
            ttl_days=settings.chat_memory_ttl_days,
            summary_trigger=settings.chat_summary_trigger,
            max_tokens=settings.chat_memory_max_tokens,
        )
    except Exception:  # pragma: no cover - defensive initialisation
        logger.exception("Failed to initialise memory store")
        return None


__all__ = ["init_chat_store", "init_memory_store"]
