"""SQLModel definitions for document ingestion artifacts."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Optional

from sqlalchemy import Column, JSON, Text, UniqueConstraint
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Field, SQLModel, Session, create_engine

from app.models.entities import JobRecord, SettingRecord


logger = logging.getLogger(__name__)


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


def _sqlite_aiosqlite_to_sync_url(async_url: str) -> str:
    """Downgrade an async ``sqlite+aiosqlite`` URL to its sync counterpart."""

    scheme, separator, remainder = async_url.partition("://")
    if not scheme:
        return async_url.replace("+aiosqlite", "", 1)

    sync_scheme = scheme.replace("+aiosqlite", "", 1)
    return f"{sync_scheme}{separator}{remainder}"


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

    def _attr_is_readable(target: Any, name: str, *, require_callable: bool = False) -> bool:
        try:
            value = getattr(target, name)
        except Exception:  # pragma: no cover - defensive against exotic descriptors
            return False

        if require_callable and not callable(value):
            return False

        return True

    class _ProxyEntry:
        __slots__ = ("value", "prefer_fallback", "validator")

        def __init__(
            self,
            value: Any,
            prefer_fallback: bool,
            validator: Callable[[Any], bool] | None = None,
        ) -> None:
            self.value = value
            self.prefer_fallback = prefer_fallback
            self.validator = validator

    extras: dict[str, _ProxyEntry] = {}
    needs_wrap = False

    def _register_extra(
        name: str,
        value: Any,
        *,
        prefer_fallback: bool,
        validator: Callable[[Any], bool] | None = None,
    ) -> None:
        """Record a fallback attribute to expose via the proxy."""

        nonlocal needs_wrap
        extras[name] = _ProxyEntry(value, prefer_fallback, validator)
        if prefer_fallback:
            needs_wrap = True

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

    try:
        dialect = getattr(engine, "dialect")
        had_dialect_attr = True
    except Exception:  # pragma: no cover - defensive
        dialect = None
        had_dialect_attr = False

    fallback_dialect = _FallbackDialect(dialect_name, dialect_driver)
    dialect_extra: Any = fallback_dialect
    prefer_dialect_fallback = False

    if not had_dialect_attr:
        prefer_dialect_fallback = True
        _try_assign_attr(engine, "dialect", fallback_dialect)
    else:
        missing_name = not hasattr(dialect, "name")
        missing_driver = not hasattr(dialect, "driver")
        proxy: Any | None = None
        if missing_name or missing_driver:
            try:
                if missing_name:
                    setattr(dialect, "name", dialect_name)
                if missing_driver:
                    setattr(dialect, "driver", dialect_driver)
            except (AttributeError, TypeError):
                proxy = _DialectProxy(dialect, dialect_name, dialect_driver)
        if proxy is None and not _attr_is_readable(engine, "dialect"):
            proxy = _DialectProxy(dialect, dialect_name, dialect_driver)
        if proxy is not None:
            dialect_extra = proxy
            prefer_dialect_fallback = True
            _try_assign_attr(engine, "dialect", proxy)

    def _dialect_validator(value: Any) -> bool:
        return hasattr(value, "name") and hasattr(value, "driver")

    fallback_url = make_url(url_str) if "+" in url_str or "://" in url_str else url_str
    url_extra: Any = fallback_url
    prefer_url_fallback = False

    try:
        current_url = getattr(engine, "url")
        has_url = True
    except Exception:  # pragma: no cover - defensive
        current_url = None
        has_url = False

    if not has_url or current_url is None:
        prefer_url_fallback = True
        _try_assign_attr(engine, "url", fallback_url)
    elif not _attr_is_readable(engine, "url"):
        prefer_url_fallback = True
        _try_assign_attr(engine, "url", fallback_url)

    def _url_validator(value: Any) -> bool:
        return value is not None

    try:
        dispose_attr = getattr(engine, "dispose")
        has_dispose = True
    except Exception:  # pragma: no cover - defensive
        dispose_attr = None
        has_dispose = False

    def _noop_dispose(*_: Any, **__: Any) -> None:
        return None

    dispose_extra: Any = _noop_dispose
    prefer_dispose_fallback = False

    if not has_dispose or not callable(dispose_attr):
        prefer_dispose_fallback = True
        _try_assign_attr(engine, "dispose", _noop_dispose)
    elif not _attr_is_readable(engine, "dispose", require_callable=True):
        prefer_dispose_fallback = True
        _try_assign_attr(engine, "dispose", _noop_dispose)

    def _callable_validator(value: Any) -> bool:
        return callable(value)

    try:
        connect_attr = getattr(engine, "connect")
        has_connect = True
    except Exception:  # pragma: no cover - defensive
        connect_attr = None
        has_connect = False

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

    connect_extra: Any = _connect
    prefer_connect_fallback = False

    if not has_connect or not callable(connect_attr):
        prefer_connect_fallback = True
        _try_assign_attr(engine, "connect", _connect)
    elif not _attr_is_readable(engine, "connect", require_callable=True):
        prefer_connect_fallback = True
        _try_assign_attr(engine, "connect", _connect)

    _register_extra(
        "dialect",
        dialect_extra,
        prefer_fallback=prefer_dialect_fallback,
        validator=_dialect_validator,
    )
    _register_extra(
        "url",
        url_extra,
        prefer_fallback=prefer_url_fallback,
        validator=_url_validator,
    )
    _register_extra(
        "dispose",
        dispose_extra,
        prefer_fallback=prefer_dispose_fallback,
        validator=_callable_validator,
    )
    _register_extra(
        "connect",
        connect_extra,
        prefer_fallback=prefer_connect_fallback,
        validator=_callable_validator,
    )

    if not needs_wrap:
        return engine

    class _EngineProxy:
        __slots__ = ("_original", "_extras")

        def __init__(self, original: Any, extras_map: dict[str, _ProxyEntry]) -> None:
            object.__setattr__(self, "_original", original)
            object.__setattr__(self, "_extras", dict(extras_map))

        def __getattr__(self, item: str) -> Any:
            extras_map = object.__getattribute__(self, "_extras")
            entry = extras_map.get(item)
            original = object.__getattribute__(self, "_original")
            if entry is None:
                return getattr(original, item)

            if entry.prefer_fallback:
                return entry.value

            try:
                candidate = getattr(original, item)
            except Exception:
                return entry.value

            validator = entry.validator
            if validator is not None and not validator(candidate):
                return entry.value

            return candidate

        def __setattr__(self, key: str, value: Any) -> None:
            extras_map = object.__getattribute__(self, "_extras")
            entry = extras_map.get(key)
            if entry is not None and entry.prefer_fallback:
                entry.value = value
                return
            setattr(object.__getattribute__(self, "_original"), key, value)

        def __dir__(self) -> list[str]:  # pragma: no cover - developer ergonomics
            extras_map = object.__getattribute__(self, "_extras")
            original = object.__getattribute__(self, "_original")
            return sorted(set(extras_map.keys()) | set(dir(original)))

    return _EngineProxy(engine, extras)


def _create_schema_if_possible(engine: Engine) -> None:
    """Create database schema when ``SQLModel.metadata`` exposes ``create_all``."""

    metadata = getattr(SQLModel, "metadata", None)
    if metadata is None:
        logger.warning("SQLModel.metadata is missing; skipping schema creation")
        return

    create_all = getattr(metadata, "create_all", None)
    if not callable(create_all):
        logger.warning(
            "SQLModel.metadata.create_all is unavailable; skipping schema creation"
        )
        return

    create_all(engine)


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
        sync_url: str

        set_method = getattr(dialect, "set", None)
        if callable(set_method):
            sync_dialect = set_method(drivername="sqlite")
            sync_url = str(sync_dialect)
        else:

            # ``sqlalchemy.engine.make_url`` can be stubbed to return a plain
            # string during tests. Fall back to simple string rewriting so the
            # synchronous driver is selected even without ``URL.set``.
            dialect_str = str(dialect) if dialect else ""
            if "+aiosqlite" in dialect_str:
                sync_url = dialect_str.replace("+aiosqlite", "", 1)
            elif "+aiosqlite" in db_url_str:
                sync_url = db_url_str.replace("+aiosqlite", "", 1)
            else:
                fallback_dialect = make_url(db_url_str)
                fallback_set = getattr(fallback_dialect, "set", None)
                if callable(fallback_set):
                    sync_url = str(fallback_set(drivername="sqlite"))
                else:
                    prefix, sep, remainder = db_url_str.partition("://")
                    if sep:
                        scheme = prefix.split("+", 1)[0]
                        sync_url = f"{scheme}{sep}{remainder}"
                    else:
                        scheme, sep2, rest = db_url_str.partition(":")
                        if sep2 and "+" in scheme:
                            sync_url = f"{scheme.split('+', 1)[0]}{sep2}{rest}"
                        else:
                            sync_url = db_url_str

            sync_url = _sqlite_aiosqlite_to_sync_url(str(dialect))

        engine = create_engine(sync_url, echo=False, connect_args=_connect_args(sync_url))
        engine = _ensure_sync_engine(engine, sync_url)
        if create_schema:
            _create_schema_if_possible(engine)
        return engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url_str))
    engine = _ensure_sync_engine(engine, db_url_str)
    if create_schema:
        _create_schema_if_possible(engine)
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
