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

from app.models.engine_guard import SyncEngineGuard
from app.models.entities import JobRecord, SettingRecord


logger = logging.getLogger(__name__)


if getattr(SQLModel, "metadata", None) is None:
    try:
        SQLModel.metadata = MetaData()  # type: ignore[assignment]
    except Exception:  # pragma: no cover - defensive fallback when assignment fails
        logger.warning("SQLModel.metadata is unavailable; schema creation may be skipped")


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


def _create_schema_if_possible(engine: Engine, metadata: Any | None) -> Any | None:
    """Create database schema when ``SQLModel.metadata`` exposes ``create_all``."""

    meta = metadata if metadata is not None else getattr(SQLModel, "metadata", None)

    if meta is None or not hasattr(meta, "create_all"):
        logger.warning(
            "SQLModel.metadata is missing required API; reinitialising metadata"
        )
        try:
            meta = MetaData()
            setattr(SQLModel, "metadata", meta)
        except Exception:
            logger.exception(
                "Failed to attach fallback MetaData to SQLModel; skipping schema creation"
            )
            return getattr(SQLModel, "metadata", None)

    if meta is None:
        logger.warning("SQLModel.metadata is missing; skipping schema creation")
        return None

    create_all = getattr(meta, "create_all", None)
    if not callable(create_all):
        logger.warning(
            "SQLModel.metadata.create_all is not callable; skipping schema creation"
        )
        return meta

    create_all(engine)
    return meta


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

    metadata = getattr(SQLModel, "metadata", None)
    if metadata is None or not isinstance(metadata, MetaData):
        metadata = MetaData()
        setattr(SQLModel, "metadata", metadata)

    if driver_name.endswith("+aiosqlite"):
        try:
            async_engine = create_async_engine(db_url, echo=False)
            if asyncio.iscoroutine(async_engine):
                async_engine = asyncio.run(async_engine)
        except ModuleNotFoundError:
            engine_url = _sqlite_aiosqlite_to_sync_url(db_url_str)
            engine = create_engine(engine_url, echo=False, connect_args=_connect_args(engine_url))
        else:
            try:
                sync_candidate = getattr(async_engine, "sync_engine")
            except AttributeError:
                sync_candidate = None

            if sync_candidate is None:
                engine_url = _sqlite_aiosqlite_to_sync_url(db_url_str)
                engine = create_engine(
                    engine_url, echo=False, connect_args=_connect_args(engine_url)
                )
            else:
                engine_url = db_url_str
                engine = sync_candidate
        _maybe_configure_sqlite_engine(engine, engine_url)
        engine = SyncEngineGuard(engine, engine_url).ensure_sync()
        _maybe_configure_sqlite_engine(engine, engine_url)
        if create_schema:
            _create_schema_if_possible(engine, metadata)
        engine = SyncEngineGuard(engine, engine_url).ensure_sync()
        _maybe_configure_sqlite_engine(engine, engine_url)
        return engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url_str))
    _maybe_configure_sqlite_engine(engine, db_url_str)
    engine = SyncEngineGuard(engine, db_url_str).ensure_sync()
    _maybe_configure_sqlite_engine(engine, db_url_str)
    if create_schema:
        _create_schema_if_possible(engine, metadata)

    engine = SyncEngineGuard(engine, db_url_str).ensure_sync()
    _maybe_configure_sqlite_engine(engine, db_url_str)

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
