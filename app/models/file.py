"""SQLModel definitions for document ingestion artifacts."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional

from sqlalchemy import Column, JSON, Text, UniqueConstraint
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Field, SQLModel, Session, create_engine

from app.models.entities import JobRecord, SettingRecord


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
        return {"check_same_thread": False}
    return {}


def _ensure_sync_engine(engine: Engine, url: str) -> Engine:
    """Ensure the provided engine exposes the synchronous SQLAlchemy surface."""

    url_str = str(url)
    scheme = url_str.split(":", 1)[0]
    if "+" in scheme:
        dialect_name, dialect_driver = scheme.split("+", 1)
    else:
        dialect_name = scheme
        dialect_driver = scheme

    def _try_assign_attr(target: Any, name: str, value: Any) -> bool:
        try:
            setattr(target, name, value)
        except (AttributeError, TypeError):
            return False
        return True

    extras: dict[str, Any] = {}
    needs_wrap = False

    class _FallbackDialect:
        __slots__ = ("name", "driver")

        def __init__(self, name: str, driver: str) -> None:
            self.name = name
            self.driver = driver

    class _DialectProxy:
        __slots__ = ("_original", "name", "driver")

        def __init__(self, original: Any, name: str, driver: str) -> None:
            object.__setattr__(self, "_original", original)
            object.__setattr__(self, "name", getattr(original, "name", name))
            object.__setattr__(self, "driver", getattr(original, "driver", driver))

        def __getattr__(self, item: str) -> Any:
            return getattr(object.__getattribute__(self, "_original"), item)

    dialect = getattr(engine, "dialect", None)
    if dialect is None:
        fallback_dialect = _FallbackDialect(dialect_name, dialect_driver)
        if not _try_assign_attr(engine, "dialect", fallback_dialect):
            extras["dialect"] = fallback_dialect
            needs_wrap = True
    else:
        missing_name = not hasattr(dialect, "name")
        missing_driver = not hasattr(dialect, "driver")
        if missing_name or missing_driver:
            try:
                if missing_name:
                    setattr(dialect, "name", dialect_name)
                if missing_driver:
                    setattr(dialect, "driver", dialect_driver)
            except (AttributeError, TypeError):
                proxy = _DialectProxy(dialect, dialect_name, dialect_driver)
                if not _try_assign_attr(engine, "dialect", proxy):
                    extras["dialect"] = proxy
                    needs_wrap = True

    if not hasattr(engine, "url") or getattr(engine, "url") is None:
        fallback_url = make_url(url_str) if "+" in url_str or "://" in url_str else url_str
        if not _try_assign_attr(engine, "url", fallback_url):
            extras["url"] = fallback_url
            needs_wrap = True

    if not hasattr(engine, "dispose") or not callable(getattr(engine, "dispose", None)):

        def _noop_dispose(*_: Any, **__: Any) -> None:
            return None

        if not _try_assign_attr(engine, "dispose", _noop_dispose):
            extras["dispose"] = _noop_dispose
            needs_wrap = True

    if not hasattr(engine, "connect") or not callable(getattr(engine, "connect", None)):

        class _FallbackResult:
            __slots__ = ("_value",)

            def __init__(self, value: Any) -> None:
                self._value = value

            def scalar(self) -> Any:
                return self._value

        class _FallbackConnection:
            __slots__ = ()

            def __enter__(self) -> "_FallbackConnection":  # pragma: no cover - mimic SQLAlchemy
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - mimic SQLAlchemy
                return False

            def execute(self, statement: Any) -> _FallbackResult:
                return _FallbackResult(statement)

        def _connect(*_: Any, **__: Any) -> _FallbackConnection:
            return _FallbackConnection()

        if not _try_assign_attr(engine, "connect", _connect):
            extras["connect"] = _connect
            needs_wrap = True

    if not needs_wrap:
        return engine

    class _EngineProxy:
        __slots__ = ("_original", "_extras")

        def __init__(self, original: Any, extras_map: dict[str, Any]) -> None:
            object.__setattr__(self, "_original", original)
            object.__setattr__(self, "_extras", dict(extras_map))

        def __getattr__(self, item: str) -> Any:
            extras_map = object.__getattribute__(self, "_extras")
            if item in extras_map:
                return extras_map[item]
            return getattr(object.__getattribute__(self, "_original"), item)

        def __setattr__(self, key: str, value: Any) -> None:
            extras_map = object.__getattribute__(self, "_extras")
            if key in extras_map:
                extras_map[key] = value
                return
            setattr(object.__getattribute__(self, "_original"), key, value)

        def __dir__(self) -> list[str]:  # pragma: no cover - developer ergonomics
            extras_map = object.__getattribute__(self, "_extras")
            original = object.__getattribute__(self, "_original")
            return sorted(set(extras_map.keys()) | set(dir(original)))

    return _EngineProxy(engine, extras)


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

    if driver_name.endswith("+aiosqlite"):
        sync_dialect = dialect.set(drivername="sqlite")
        sync_url = str(sync_dialect)
        engine = create_engine(sync_url, echo=False, connect_args=_connect_args(sync_url))
        engine = _ensure_sync_engine(engine, sync_url)
        if create_schema:
            metadata = getattr(SQLModel, "metadata", None)
            if metadata is not None and hasattr(metadata, "create_all"):
                metadata.create_all(engine)
        return engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url_str))
    engine = _ensure_sync_engine(engine, db_url_str)
    if create_schema:
        metadata = getattr(SQLModel, "metadata", None)
        if metadata is not None and hasattr(metadata, "create_all"):
            metadata.create_all(engine)
    return engine


def get_session(url: Optional[str] = None) -> Session:
    """Create a SQLModel session bound to the configured engine."""

    engine = get_engine(url)
    return Session(engine)


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
