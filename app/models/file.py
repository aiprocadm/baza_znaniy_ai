"""SQLModel definitions for document ingestion artifacts."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional

import sqlite3
from urllib.parse import unquote, urlparse

from sqlalchemy import Column, JSON, MetaData, Text, UniqueConstraint, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import create_async_engine
try:  # pragma: no cover - optional dependency during lightweight test runs
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - fallback when SQLAlchemy ORM is absent
    sessionmaker = None  # type: ignore[assignment]
from sqlmodel import Field, SQLModel, Session, create_engine

from app.observability.metrics import (
    record_sqlmodel_metadata_alert,
    record_sqlmodel_metadata_state,
)
from app.models.engine_guard import SyncEngineGuard
from app.models.entities import JobRecord, SettingRecord


logger = logging.getLogger(__name__)


if getattr(SQLModel, "metadata", None) is None:
    try:
        SQLModel.metadata = MetaData()  # type: ignore[assignment]
    except Exception:  # pragma: no cover - defensive fallback when assignment fails
        logger.warning(
            "SQLModel.metadata is unavailable; schema creation may be skipped"
        )


def _record_sqlmodel_metadata_health(metadata: Any | None, *, origin: str) -> None:
    """Update Prometheus metrics describing the SQLModel metadata state."""

    try:
        healthy, reason = record_sqlmodel_metadata_state(metadata, origin=origin)
    except Exception:  # pragma: no cover - instrumentation must not break flows
        logger.debug("Failed to record SQLModel metadata metrics", exc_info=True)
        return

    if healthy:
        return

    try:
        record_sqlmodel_metadata_alert(origin=origin, reason=reason)
    except Exception:  # pragma: no cover - best-effort alerting
        logger.debug(
            "Failed to increment SQLModel metadata alert counter", exc_info=True
        )


def _ensure_sqlmodel_metadata(metadata: Any | None) -> MetaData:
    """Return a valid SQLModel metadata instance, creating one if required."""

    if isinstance(metadata, MetaData):
        return metadata

    candidate = getattr(SQLModel, "metadata", None) if metadata is None else metadata
    if candidate is not None and not isinstance(candidate, MetaData):
        create_all_attr = getattr(candidate, "create_all", None)
        if create_all_attr is not None and not callable(create_all_attr):
            logger.error(
                "SQLModel.metadata.create_all is not callable; cannot create schema"
            )
            raise RuntimeError(
                "SQLModel metadata initialisation failed: metadata.create_all is not callable"
            )

    candidate = getattr(SQLModel, "metadata", None)
    if isinstance(candidate, MetaData):
        return candidate

    fallback = MetaData()
    try:
        setattr(SQLModel, "metadata", fallback)
    except Exception:  # pragma: no cover - best-effort assignment
        logger.warning(
            "Unable to attach fallback SQLModel metadata; schema creation may fail"
        )
    return fallback


class FileStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRecord(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256", name="uq_documents_tenant_sha"),
        UniqueConstraint("tenant_slug", "slug", name="uq_documents_tenant_slug"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    tenant_slug: Optional[str] = Field(default=None, index=True)
    file_id: Optional[int] = Field(default=None, foreign_key="files.id", index=True)
    sha256: str = Field(index=True)
    slug: Optional[str] = Field(default=None, index=True)
    title: Optional[str] = Field(default=None)
    mime_type: str = Field(default="application/octet-stream")
    status: str = Field(default=DocumentStatus.QUEUED, index=True)
    error: Optional[str] = Field(default=None)
    meta: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    chunks: Optional[int] = Field(default=None)
    content: str = Field(default="", sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class FileRecord(SQLModel, table=True):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256", name="uq_files_tenant_sha"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    sha256: str = Field(index=True)
    document_id: Optional[int] = Field(default=None, foreign_key="documents.id", index=True)
    path: str
    filename: str
    size: int = Field(default=0, ge=0)
    status: str = Field(default=FileStatus.QUEUED, index=True)
    retries: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    meta: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    chunks: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class PageRecord(SQLModel, table=True):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("tenant_id", "file_id", "number", name="uq_pages_tenant_file_number"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    file_id: int = Field(foreign_key="files.id", index=True)
    number: int = Field(index=True)
    sha256: str = Field(index=True)
    text: str
    tokens: int = Field(default=0, ge=0)
    meta: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class ChunkRecord(SQLModel, table=True):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("tenant_id", "page_id", "index", name="uq_chunks_tenant_page_index"),
        UniqueConstraint("tenant_id", "page_id", "sha256", name="uq_chunks_tenant_page_sha"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    page_id: int = Field(foreign_key="pages.id", index=True)
    index: int = Field(index=True)
    sha256: str = Field(index=True)
    text: str
    batch: Optional[int] = Field(default=None, index=True)
    tokens: int = Field(default=0, ge=0)
    meta: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


def _connect_args(url: str) -> dict[str, object]:
    if url.startswith("sqlite"):
        return {"check_same_thread": False, "timeout": 5.0}
    return {}


def _maybe_configure_sqlite_engine(engine: Engine, url: str) -> None:
    """Apply default SQLite pragmas for durability and concurrency."""

    try:
        dialect = make_url(url)
    except Exception:
        dialect = None

    if isinstance(dialect, str):
        driver_name = dialect
    else:
        driver_name = getattr(dialect, "drivername", "") or ""

    if not str(driver_name).lower().startswith("sqlite"):
        return

    pragmas = (
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA busy_timeout=5000",
    )

    try:
        with engine.connect() as connection:
            for statement in pragmas:
                connection.execute(text(statement))
            connection.commit()
            return
    except Exception:
        logger.debug(
            "SQLite engine connection is unavailable for PRAGMA configuration",
            exc_info=True,
        )

    database: str | None = None

    if (
        dialect is not None
        and not isinstance(dialect, str)
        and hasattr(dialect, "database")
    ):
        database = getattr(dialect, "database", None)

    raw_url = str(url)
    if not database:
        parsed = urlparse(raw_url)
        if parsed.scheme.lower() == "sqlite":
            if parsed.netloc:
                database = f"/{parsed.netloc}{parsed.path}"
            else:
                database = parsed.path
            database = unquote(database or "")

    if not database or database == ":memory:":
        return

    if database.startswith("//"):
        database = "/" + database.lstrip("/")
    elif database.startswith("/") and not raw_url.startswith("sqlite:////"):
        database = database.lstrip("/")

    try:
        with sqlite3.connect(database) as sqlite_conn:
            for statement in pragmas:
                cursor = sqlite_conn.execute(statement)
                try:
                    cursor.fetchall()
                except Exception:
                    pass
            sqlite_conn.commit()
    except Exception:
        logger.exception("Failed to configure SQLite pragmas")


def _sqlite_aiosqlite_to_sync_url(async_url: str) -> str:
    """Downgrade an async ``sqlite+aiosqlite`` URL to its sync counterpart."""

    scheme, separator, remainder = async_url.partition("://")
    if not scheme:
        return async_url.replace("+aiosqlite", "", 1)

    sync_scheme = scheme.replace("+aiosqlite", "", 1)
    return f"{sync_scheme}{separator}{remainder}"


class _SQLAlchemySessionAdapter:
    """Compatibility wrapper providing ``Session.exec`` for SQLAlchemy sessions."""

    __slots__ = ("_session",)

    def __init__(self, engine: Engine) -> None:
        if sessionmaker is None:
            raise RuntimeError(
                "SQLAlchemy sessionmaker is unavailable; install SQLAlchemy to enable database access"
            )

        factory = sessionmaker(
            bind=engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
        self._session = factory()

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "_SQLAlchemySessionAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        closer = getattr(self._session, "close", None)
        if callable(closer):
            closer()

    # ------------------------------------------------------------------
    # SQLModel compatible helpers
    # ------------------------------------------------------------------
    def exec(self, statement):
        execute = getattr(self._session, "execute", None)
        if not callable(execute):
            raise AttributeError("Underlying session does not support execute")
        result = execute(statement)
        scalars = getattr(result, "scalars", None)
        return scalars() if callable(scalars) else result

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------
    def __getattr__(self, item):  # pragma: no cover - simple delegation
        return getattr(self._session, item)


@lru_cache(maxsize=1)
def get_engine(url: Optional[str] = None, *, create_schema: bool = True) -> Engine:
    """Return a synchronous SQLAlchemy engine configured for SQLModel models."""

    db_url = url or os.getenv("DB_URL", "sqlite+aiosqlite:///./var/data/kb.sqlite")
    dialect = make_url(db_url)

    driver_name = getattr(dialect, "drivername", "") or str(dialect)
    driver_name = driver_name.split(":", 1)[0].lower()
    db_url_str = str(db_url)

    # Ensure additional SQLModel definitions are imported before metadata creation
    __import__("app.models.entities")

    metadata = _ensure_sqlmodel_metadata(getattr(SQLModel, "metadata", None))

    engine_url = db_url_str

    if driver_name.endswith("+aiosqlite"):
        engine_url = _sqlite_aiosqlite_to_sync_url(db_url_str)

        def _dispose_async_engine(engine: Any) -> None:
            disposer = getattr(engine, "dispose", None)
            if disposer is None:
                return

            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(disposer())
                finally:
                    loop.close()
                return

            try:
                result = disposer()
            except TypeError:
                return
            if not asyncio.iscoroutine(result):
                return
            try:
                asyncio.run(result)
            except Exception:
                logger.debug(
                    "Failed to dispose async SQLite engine",
                    exc_info=True,
                )

        try:
            async_engine = create_async_engine(db_url, echo=False)
            if asyncio.iscoroutine(async_engine):
                async_engine = asyncio.run(async_engine)
        except ModuleNotFoundError:
            async_engine = None
        except Exception:
            logger.debug(
                "Failed to initialise async SQLite engine; falling back to sync engine",
                exc_info=True,
            )
            async_engine = None
        else:
            try:
                _dispose_async_engine(async_engine)
            except Exception:
                logger.debug(
                    "Error disposing async SQLite engine prior to sync fallback",
                    exc_info=True,
                )

        engine = create_engine(
            engine_url, echo=False, connect_args=_connect_args(engine_url)
        )
    else:
        engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url_str))

    _maybe_configure_sqlite_engine(engine, engine_url)
    engine = SyncEngineGuard(engine, engine_url).ensure_sync()

    if create_schema:
        meta = _ensure_sqlmodel_metadata(metadata)
        create_all = getattr(meta, "create_all", None)
        if callable(create_all):
            create_all(engine)
        else:
            logger.error(
                "SQLModel.metadata.create_all is not callable; cannot create schema"
            )
            raise RuntimeError(
                "SQLModel metadata initialisation failed: metadata.create_all is not callable"
            )
        current_metadata = getattr(SQLModel, "metadata", meta)
        _record_sqlmodel_metadata_health(current_metadata, origin="get_engine")
    else:
        _record_sqlmodel_metadata_health(
            getattr(SQLModel, "metadata", metadata), origin="get_engine"
        )

    return engine



def get_session(url: Optional[str] = None, *, engine: Engine | None = None):
    """Create a SQLModel-compatible session bound to the configured engine."""

    target_engine = engine or get_engine(url)

    session: Any
    try:
        session = Session(target_engine)
    except TypeError:
        session = Session(bind=target_engine)  # type: ignore[call-arg]

    exec_method = getattr(session, "exec", None)
    if callable(exec_method):
        return session

    closer = getattr(session, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # pragma: no cover - defensive best-effort cleanup
            pass

    logger.warning(
        "sqlmodel.Session lacks 'exec'; falling back to SQLAlchemy session adapter"
    )
    return _SQLAlchemySessionAdapter(target_engine)


__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "DocumentStatus",
    "FileRecord",
    "FileStatus",
    "JobRecord",
    "PageRecord",
    "SettingRecord",
    "get_engine",
    "get_session",
]
