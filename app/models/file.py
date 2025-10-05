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


_FALLBACK_MARKER = "__kb_sync_engine_fallback__"


def _mark_fallback(value: Any) -> None:
    """Mark the provided value as a synthetic fallback."""

    try:
        setattr(value, _FALLBACK_MARKER, True)
    except Exception:  # pragma: no cover - attribute assignment best effort
        return


def _is_fallback_value(value: Any) -> bool:
    """Check whether a value has been flagged as a fallback helper."""

    try:
        return bool(getattr(value, _FALLBACK_MARKER))
    except Exception:  # pragma: no cover - defensive against exotic descriptors
        return False


def _collect_sqlmodel_tables() -> list[tuple[type[Any], Any]]:
    """Capture SQLModel table mappings for later metadata migration."""

    tables: list[tuple[type[Any], Any]] = []
    seen: set[type[Any]] = set()
    stack = list(getattr(SQLModel, "__subclasses__", lambda: [])())

    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)

        table = getattr(cls, "__table__", None)
        if table is not None and hasattr(table, "tometadata"):
            tables.append((cls, table))

        stack.extend(getattr(cls, "__subclasses__", lambda: [])())

    return tables


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


def _ensure_sync_engine(engine: Engine, url: str) -> Engine:
    """Ensure the provided engine exposes the synchronous SQLAlchemy surface."""

    url_str = str(url)
    scheme = url_str.split(":", 1)[0]
    if "+" in scheme:
        dialect_name, dialect_driver = scheme.split("+", 1)
    else:
        dialect_name = scheme
        dialect_driver = scheme

    needs_wrap = False
    attr_failure = False

    def _note_failure() -> None:
        nonlocal needs_wrap, attr_failure
        needs_wrap = True
        attr_failure = True

    def _try_assign_attr(target: Any, name: str, value: Any) -> bool:
        try:
            setattr(target, name, value)
        except (AttributeError, TypeError):
            _note_failure()
            return False
        return True

    def _attr_is_readable(target: Any, name: str, *, require_callable: bool = False) -> bool:
        try:
            value = getattr(target, name)
        except Exception:  # pragma: no cover - defensive against exotic descriptors
            _note_failure()
            return False

        if require_callable and not callable(value):
            _note_failure()
            return False

        return True


    preserved_callables: dict[str, Any] = {}


    originals: dict[str, Any] = {}


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


    def _register_extra(
        name: str,
        value: Any,
        *,
        prefer_fallback: bool,
        validator: Callable[[Any], bool] | None = None,
        assignment_failed: bool = False,
    ) -> None:
        """Record a fallback attribute to expose via the proxy."""

        nonlocal needs_wrap, attr_failure

        if assignment_failed:
            needs_wrap = True
            attr_failure = True

        is_fallback = _is_fallback_value(value)
        effective_prefer_fallback = prefer_fallback or is_fallback

        entry = extras.get(name)
        if entry is None:
            extras[name] = _ProxyEntry(value, effective_prefer_fallback, validator)
            return

        if effective_prefer_fallback:
            if is_fallback or not entry.prefer_fallback:
                entry.value = value
            entry.prefer_fallback = True
            if validator is not None:
                entry.validator = validator
            return

        if entry.prefer_fallback:
            can_downgrade = not is_fallback
            if validator is not None:
                try:
                    can_downgrade = can_downgrade and bool(validator(value))
                except Exception:
                    can_downgrade = False
            if can_downgrade:
                entry.prefer_fallback = False
            if validator is not None:
                entry.validator = validator
            return

        entry.value = value
        if validator is not None:
            entry.validator = validator


    def _preserve_callable(name: str, value: Any) -> None:
        if callable(value):
            preserved_callables[name] = value

    class _FallbackDialect:
        __slots__ = ("name", "driver", _FALLBACK_MARKER)

        def __init__(self, name: str, driver: str) -> None:
            self.name = name
            self.driver = driver
            setattr(self, _FALLBACK_MARKER, True)

    class _DialectProxy:
        __slots__ = ("_original", "name", "driver", _FALLBACK_MARKER)

        def __init__(self, original: Any, name: str, driver: str) -> None:
            object.__setattr__(self, "_original", original)
            object.__setattr__(self, "name", getattr(original, "name", name))
            object.__setattr__(self, "driver", getattr(original, "driver", driver))
            object.__setattr__(self, _FALLBACK_MARKER, True)

        def __getattr__(self, item: str) -> Any:
            return getattr(object.__getattribute__(self, "_original"), item)

    try:
        dialect = getattr(engine, "dialect")
        had_dialect_attr = True
    except Exception:  # pragma: no cover - defensive
        dialect = None
        had_dialect_attr = False

    fallback_dialect = _FallbackDialect(dialect_name, dialect_driver)


    dialect_value: Any
    dialect_extra: Any = fallback_dialect
    prefer_dialect_fallback = False

    if not had_dialect_attr:
        assigned = _try_assign_attr(engine, "dialect", fallback_dialect)
        if not assigned:
            _register_extra(
                "dialect",
                fallback_dialect,
                prefer_fallback=True,
                assignment_failed=True,
            )

        dialect_value = fallback_dialect
        dialect_extra = fallback_dialect
        prefer_dialect_fallback = True

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
            assigned = _try_assign_attr(engine, "dialect", proxy)
            _register_extra(
                "dialect",
                proxy,
                prefer_fallback=True,
                assignment_failed=not assigned,
            )

            dialect_value = proxy
            dialect_extra = proxy
            prefer_dialect_fallback = True
        else:
            dialect_value = dialect if dialect is not None else fallback_dialect
            dialect_extra = dialect_value

    fallback_url = make_url(url_str) if "+" in url_str or "://" in url_str else url_str

    if "dialect_value" not in locals():  # pragma: no cover - defensive fallback
        dialect_value = fallback_dialect

    def _dialect_validator(value: Any) -> bool:
        return hasattr(value, "name") and hasattr(value, "driver")

    url_extra: Any = fallback_url
    prefer_url_fallback = False



    try:
        current_url = getattr(engine, "url")
        has_url = True
    except Exception:  # pragma: no cover - defensive
        current_url = None
        has_url = False


    if has_url and current_url is not None and _attr_is_readable(engine, "url"):
        url_value = current_url
        url_extra = current_url
    else:
        assigned = _try_assign_attr(engine, "url", fallback_url)
        _register_extra(
            "url",
            fallback_url,
            prefer_fallback=True,
            assignment_failed=not assigned,
        )
        url_value = current_url if current_url is not None else fallback_url
        url_extra = fallback_url
        prefer_url_fallback = True

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

    _mark_fallback(_noop_dispose)

    dispose_extra: Any | None = None
    prefer_dispose_fallback = False

    dispose_is_valid = has_dispose and callable(dispose_attr) and _attr_is_readable(
        engine, "dispose", require_callable=True
    )

    if dispose_is_valid:
        dispose_value = dispose_attr
        dispose_extra = dispose_attr
        _preserve_callable("dispose", dispose_attr)
        originals["dispose"] = dispose_attr
    else:
        dispose_value = _noop_dispose

    if not dispose_is_valid:
        assigned = _try_assign_attr(engine, "dispose", _noop_dispose)
        _register_extra(
            "dispose",
            _noop_dispose,
            prefer_fallback=True,
            assignment_failed=not assigned,
        )
        dispose_extra = _noop_dispose
        prefer_dispose_fallback = True

    if dispose_extra is None:
        dispose_extra = dispose_value

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

    _mark_fallback(_connect)

    connect_extra: Any | None = None
    prefer_connect_fallback = False

    connect_is_valid = has_connect and callable(connect_attr) and _attr_is_readable(
        engine, "connect", require_callable=True
    )

    if connect_is_valid:
        connect_value = connect_attr
        connect_extra = connect_attr
        _preserve_callable("connect", connect_attr)
        originals["connect"] = connect_attr
    else:
        connect_value = _connect

    if not connect_is_valid:
        assigned = _try_assign_attr(engine, "connect", _connect)
        _register_extra(
            "connect",
            _connect,
            prefer_fallback=True,
            assignment_failed=not assigned,
        )
        connect_extra = _connect
        prefer_connect_fallback = True

    if connect_extra is None:
        connect_extra = connect_value


    _register_extra(
        "dialect",
        dialect_value,
        prefer_fallback=False,
        validator=_dialect_validator,
    )
    _register_extra(
        "url",
        url_value,
        prefer_fallback=False,
        validator=_url_validator,
    )
    _register_extra(
        "dispose",
        dispose_value,
        prefer_fallback=False,
        validator=_callable_validator,
    )
    _register_extra(
        "connect",
        connect_value,
        prefer_fallback=False,
        validator=_callable_validator,
    )

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
    if dispose_extra is not None:
        _register_extra(
            "dispose",
            dispose_extra,
            prefer_fallback=prefer_dispose_fallback,
            validator=_callable_validator,
        )
    if connect_extra is not None:
        _register_extra(
            "connect",
            connect_extra,
            prefer_fallback=prefer_connect_fallback,
            validator=_callable_validator,
        )


    for name, entry in extras.items():
        if entry.prefer_fallback:
            needs_wrap = True
            continue

        try:
            candidate = getattr(engine, name)
        except Exception:
            entry.prefer_fallback = True
            needs_wrap = True
            attr_failure = True
            continue

        validator = entry.validator
        if validator is not None and not validator(candidate):
            entry.prefer_fallback = True
            needs_wrap = True

    def _final_fallback(name: str) -> Any:
        return {
            "dialect": dialect_extra or fallback_dialect,
            "url": url_extra or fallback_url,
            "dispose": dispose_extra or _noop_dispose,
            "connect": connect_extra or _connect,
        }[name]

    _core_validators: dict[str, Callable[[Any], bool]] = {
        "dialect": _dialect_validator,
        "url": _url_validator,
        "dispose": _callable_validator,
        "connect": _callable_validator,
    }

    def _force_core_fallback(attr_name: str) -> None:
        nonlocal needs_wrap

        fallback_value = _final_fallback(attr_name)
        validator = _core_validators[attr_name]

        _register_extra(
            attr_name,
            fallback_value,
            prefer_fallback=True,
            validator=validator,
        )

        needs_wrap = True

    for attr_name, require_callable in (
        ("dialect", False),
        ("url", False),
        ("dispose", True),
        ("connect", True),
    ):
        validator = _core_validators[attr_name]

        try:
            attr_value = getattr(engine, attr_name)
        except Exception:
            attr_failure = True
            _force_core_fallback(attr_name)
            continue

        if require_callable and not callable(attr_value):
            _force_core_fallback(attr_name)
            continue

        if attr_name == "dialect":
            try:
                getattr(attr_value, "name")
                getattr(attr_value, "driver")
            except Exception:
                attr_failure = True
                _force_core_fallback(attr_name)
                continue

        try:
            if not validator(attr_value):
                raise ValueError(attr_name)
        except Exception:
            _force_core_fallback(attr_name)



    if not needs_wrap and not attr_failure:
        return engine

    def _ensure_proxy_entry(
        name: str,
        value: Any,
        *,
        prefer_fallback: bool,
        validator: Callable[[Any], bool] | None,
    ) -> None:
        if value is None:
            return

        entry = extras.get(name)
        if entry is None:
            extras[name] = _ProxyEntry(value, prefer_fallback, validator)
            return

        entry.value = value
        if prefer_fallback:
            entry.prefer_fallback = True
        if validator is not None:
            entry.validator = validator

    _ensure_proxy_entry(
        "dialect",
        dialect_extra,
        prefer_fallback=prefer_dialect_fallback,
        validator=_dialect_validator,
    )
    _ensure_proxy_entry(
        "url",
        url_extra,
        prefer_fallback=prefer_url_fallback,
        validator=_url_validator,
    )
    _ensure_proxy_entry(
        "dispose",
        dispose_extra,
        prefer_fallback=prefer_dispose_fallback,
        validator=_callable_validator,
    )
    _ensure_proxy_entry(
        "connect",
        connect_extra,
        prefer_fallback=prefer_connect_fallback,
        validator=_callable_validator,
    )

    for name, value in originals.items():
        extras.setdefault(
            name,
            _ProxyEntry(
                value,
                False,
                _callable_validator if callable(value) else None,
            ),
        )

    class _EngineProxy:
        __slots__ = ("_original", "_extras", "_preserved")

        def __init__(
            self,
            original: Any,
            extras_map: dict[str, _ProxyEntry],
            preserved: dict[str, Any],
        ) -> None:
            object.__setattr__(self, "_original", original)
            object.__setattr__(self, "_extras", dict(extras_map))
            object.__setattr__(self, "_preserved", dict(preserved))

        def _fallback_value(self, item: str, entry: _ProxyEntry) -> Any:
            preserved = object.__getattribute__(self, "_preserved").get(item)
            if preserved is not None:
                return preserved
            return entry.value

        def _resolve_extra(self, item: str, entry: _ProxyEntry) -> Any:
            if entry.prefer_fallback:
                return self._fallback_value(item, entry)

            original = object.__getattribute__(self, "_original")
            try:
                candidate = getattr(original, item)
            except Exception:
                return self._fallback_value(item, entry)

            validator = entry.validator
            if validator is not None and not validator(candidate):
                return self._fallback_value(item, entry)

            preserved = object.__getattribute__(self, "_preserved").get(item)
            if preserved is not None and callable(preserved):
                return preserved

            return candidate

        def __getattribute__(self, item: str) -> Any:
            if item in {
                "_original",
                "_extras",
                "_preserved",
                "__setattr__",
                "__getattribute__",
                "__dir__",
                "__repr__",
                "_fallback_value",
                "_resolve_extra",
            }:
                return object.__getattribute__(self, item)

            if item == "__class__":
                original = object.__getattribute__(self, "_original")
                return type(original)

            if item == "__wrapped__":
                return object.__getattribute__(self, "_original")

            extras_map = object.__getattribute__(self, "_extras")
            entry = extras_map.get(item)
            if entry is not None:
                return self._resolve_extra(item, entry)

            original = object.__getattribute__(self, "_original")
            return getattr(original, item)

        def __getattr__(self, item: str) -> Any:
            extras_map = object.__getattribute__(self, "_extras")
            entry = extras_map.get(item)
            if entry is None:
                original = object.__getattribute__(self, "_original")
                return getattr(original, item)

            return self._resolve_extra(item, entry)

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

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            original = object.__getattribute__(self, "_original")
            return f"EngineProxy({original!r})"

    proxy_extras = dict(extras)

    for name, value in originals.items():
        entry = proxy_extras.get(name)
        if isinstance(entry, _ProxyEntry):
            entry.value = value
        else:
            proxy_extras[name] = _ProxyEntry(value, False, _callable_validator if callable(value) else None)

    for name, preserved in preserved_callables.items():
        entry = proxy_extras.get(name)
        if isinstance(entry, _ProxyEntry):
            if callable(preserved) and entry.validator is None:
                entry.validator = _callable_validator
            entry.value = preserved
        else:
            proxy_extras[name] = _ProxyEntry(
                preserved,
                False,
                _callable_validator if callable(preserved) else None,
            )

    return _EngineProxy(engine, proxy_extras, preserved_callables)


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


def _create_schema_if_possible(engine: Engine, metadata: Any | None) -> MetaData | None:
    """Create database schema when ``SQLModel.metadata`` exposes ``create_all``."""

    meta: MetaData | None
    tables_snapshot: list[tuple[type[Any], Any]] | None = None

    if isinstance(metadata, MetaData):
        meta = metadata
    else:
        candidate = getattr(SQLModel, "metadata", None)
        meta = candidate if isinstance(candidate, MetaData) else None

    if meta is None:
        tables_snapshot = _collect_sqlmodel_tables()
        logger.warning(
            "SQLModel.metadata is missing or invalid; reinitialising metadata"
        )
        try:
            meta = MetaData()
            setattr(SQLModel, "metadata", meta)
        except Exception:
            logger.exception(
                "Failed to attach fallback MetaData to SQLModel; skipping schema creation"
            )
            return None

        if tables_snapshot:
            for model_cls, table in tables_snapshot:
                migrate = getattr(table, "to_metadata", None)
                if not callable(migrate):
                    migrate = getattr(table, "tometadata", None)
                if not callable(migrate):
                    continue
                try:
                    new_table = migrate(meta)
                except Exception:
                    logger.exception(
                        "Failed to migrate SQLModel table %s to new metadata",
                        getattr(model_cls, "__name__", repr(model_cls)),
                    )
                    continue
                try:
                    setattr(model_cls, "__table__", new_table)
                except Exception:
                    logger.debug(
                        "Unable to update __table__ for %s during metadata migration",
                        getattr(model_cls, "__name__", repr(model_cls)),
                        exc_info=True,
                    )

    create_all = getattr(meta, "create_all", None)
    if not callable(create_all):
        logger.error(
            "SQLModel.metadata.create_all is not callable; cannot create schema"
        )
        return None

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
            schema_metadata = _create_schema_if_possible(engine, metadata)
            if schema_metadata is None:
                logger.error(
                    "Unable to initialise SQLModel metadata; aborting engine setup"
                )
                raise RuntimeError(
                    "SQLModel metadata initialisation failed; cannot create schema"
                )
            setattr(SQLModel, "metadata", schema_metadata)
            metadata = schema_metadata
        ensured_engine = _ensure_sync_engine(engine, engine_url)
        if ensured_engine is not engine:
            _maybe_configure_sqlite_engine(ensured_engine, engine_url)
        return ensured_engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url_str))
    _maybe_configure_sqlite_engine(engine, db_url_str)
    engine = SyncEngineGuard(engine, db_url_str).ensure_sync()
    _maybe_configure_sqlite_engine(engine, db_url_str)
    if create_schema:
        schema_metadata = _create_schema_if_possible(engine, metadata)
        if schema_metadata is None:
            logger.error(
                "Unable to initialise SQLModel metadata; aborting engine setup"
            )
            raise RuntimeError(
                "SQLModel metadata initialisation failed; cannot create schema"
            )
        setattr(SQLModel, "metadata", schema_metadata)
        metadata = schema_metadata

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
