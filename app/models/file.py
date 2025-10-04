"""SQLModel definitions for document ingestion artifacts."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Optional

from sqlalchemy import Column, JSON, MetaData, Text, UniqueConstraint
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


    extras: dict[str, Any] = {}

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

    needs_wrap = False


    def _register_extra(
        name: str,
        value: Any,
        *,
        prefer_fallback: bool,
        validator: Callable[[Any], bool] | None = None,
    ) -> None:
        """Record a fallback attribute to expose via the proxy."""


        extras[name] = value

        nonlocal needs_wrap
        extras[name] = _ProxyEntry(value, prefer_fallback, validator)


    def _preserve_callable(name: str, value: Any) -> None:
        if callable(value):
            preserved_callables[name] = value

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


    dialect_value: Any
    dialect_extra: Any = fallback_dialect
    prefer_dialect_fallback = False

    if not had_dialect_attr:
        if not _try_assign_attr(engine, "dialect", fallback_dialect):
            _register_extra("dialect", fallback_dialect, prefer_fallback=True)

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
            if not _try_assign_attr(engine, "dialect", proxy):
                _register_extra("dialect", proxy, prefer_fallback=True)

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


    if not has_url or current_url is None or not _attr_is_readable(engine, "url"):
        _try_assign_attr(engine, "url", fallback_url)
        _register_extra("url", fallback_url, prefer_fallback=True)


    if not has_url or current_url is None or not _attr_is_readable(engine, "url"):
        url_value = fallback_url

    if not has_url or current_url is None:

        url_extra = fallback_url
        if not _try_assign_attr(engine, "url", fallback_url):
            _register_extra("url", fallback_url, prefer_fallback=True)
    elif not _attr_is_readable(engine, "url"):
        url_extra = fallback_url
        if not _try_assign_attr(engine, "url", fallback_url):
            _register_extra("url", fallback_url, prefer_fallback=True)

        prefer_url_fallback = True
        _try_assign_attr(engine, "url", fallback_url)
    elif not _attr_is_readable(engine, "url"):
        prefer_url_fallback = True

        _try_assign_attr(engine, "url", fallback_url)
    else:
        url_value = current_url

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

    dispose_extra: Any | None = None
    prefer_dispose_fallback = False

    if has_dispose and callable(dispose_attr) and _attr_is_readable(
        engine, "dispose", require_callable=True
    ):
        dispose_value = dispose_attr
        dispose_extra = dispose_attr
        _preserve_callable("dispose", dispose_attr)
        originals["dispose"] = dispose_attr
    else:
        dispose_value = _noop_dispose

    if not has_dispose or not callable(dispose_attr):
        if not _try_assign_attr(engine, "dispose", _noop_dispose):
            _register_extra("dispose", _noop_dispose, prefer_fallback=True)
        dispose_extra = _noop_dispose
        prefer_dispose_fallback = True
    elif not _attr_is_readable(engine, "dispose", require_callable=True):
        if not _try_assign_attr(engine, "dispose", _noop_dispose):
            _register_extra("dispose", _noop_dispose, prefer_fallback=True)
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

    connect_extra: Any | None = None
    prefer_connect_fallback = False

    if has_connect and callable(connect_attr) and _attr_is_readable(
        engine, "connect", require_callable=True
    ):
        connect_value = connect_attr
        connect_extra = connect_attr
        _preserve_callable("connect", connect_attr)
        originals["connect"] = connect_attr
    else:
        connect_value = _connect

    if not has_connect or not callable(connect_attr):
        if not _try_assign_attr(engine, "connect", _connect):
            _register_extra("connect", _connect, prefer_fallback=True)
        connect_extra = _connect
        prefer_connect_fallback = True
    elif not _attr_is_readable(engine, "connect", require_callable=True):
        if not _try_assign_attr(engine, "connect", _connect):
            _register_extra("connect", _connect, prefer_fallback=True)
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

    for attr_name, require_callable in (
        ("dialect", False),
        ("url", False),
        ("dispose", True),
        ("connect", True),
    ):
        try:
            attr_value = getattr(engine, attr_name)
            if require_callable and not callable(attr_value):
                raise TypeError(attr_name)
            if attr_name == "dialect":
                getattr(attr_value, "name")
                getattr(attr_value, "driver")
        except Exception:
            _register_extra(
                attr_name,
                _final_fallback(attr_name),
                prefer_fallback=True,
                validator=(
                    _dialect_validator
                    if attr_name == "dialect"
                    else _url_validator
                    if attr_name == "url"
                    else _callable_validator
                ),
            )



    if not needs_wrap:
        return engine

    if "dialect" not in extras and dialect_extra is not None:
        extras["dialect"] = dialect_extra
    if "url" not in extras and url_extra is not None:
        extras["url"] = url_extra
    if "dispose" not in extras and dispose_extra is not None:
        extras["dispose"] = dispose_extra
    if "connect" not in extras and connect_extra is not None:
        extras["connect"] = connect_extra

    for name, value in originals.items():
        extras.setdefault(name, value)

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


def _create_schema_if_possible(engine: Engine, metadata: Any | None) -> None:
    """Create database schema when ``SQLModel.metadata`` exposes ``create_all``."""

    metadata = getattr(SQLModel, "metadata", None)
    if metadata is None or not hasattr(metadata, "create_all"):
        logger.warning(
            "SQLModel.metadata is missing required API; reinitialising metadata"
        )
        try:
            metadata = MetaData()
            setattr(SQLModel, "metadata", metadata)
        except Exception:
            logger.exception(
                "Failed to attach fallback MetaData to SQLModel; skipping schema creation"
            )
            return
    if metadata is None:
        logger.warning("SQLModel.metadata is missing; skipping schema creation")
        return

    try:
        create_all = getattr(metadata, "create_all")
    except AttributeError:
        logger.warning(
            "SQLModel.metadata has no 'create_all'; skipping schema creation"
        )
        return

    if not callable(create_all):
        logger.warning(
            "SQLModel.metadata.create_all is unavailable even after fallback; skipping schema creation"
            "SQLModel.metadata.create_all is not callable; skipping schema creation"
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

    metadata = getattr(SQLModel, "metadata", None)

    if driver_name.endswith("+aiosqlite"):
        dialect_str = str(dialect) if dialect is not None else ""
        sync_url = _sqlite_aiosqlite_to_sync_url(db_url_str)
        if sync_url == db_url_str and "+aiosqlite" in dialect_str:
            sync_url = _sqlite_aiosqlite_to_sync_url(dialect_str)
        if "+aiosqlite" in sync_url:
            sync_url = sync_url.replace("+aiosqlite", "", 1)

        sync_url = db_url_str.replace("+aiosqlite", "", 1)
        engine = create_engine(sync_url, echo=False, connect_args=_connect_args(sync_url))
        engine = _ensure_sync_engine(engine, sync_url)
        if create_schema:
            metadata = getattr(SQLModel, "metadata", None)
            create_all = getattr(metadata, "create_all", None) if metadata is not None else None
            if callable(create_all):
                create_all(engine)

        sync_url: str

        if hasattr(dialect, "set"):
            # ``sqlalchemy.engine.URL`` exposes ``set`` for copying with driver
            # overrides. When available, it provides the canonical sync URL.
            sync_url = str(dialect.set(drivername="sqlite"))
        else:
            # ``make_url`` may be stubbed to return a string without ``set``.
            # Fall back to string rewriting and retry URL reconstruction so a
            # synchronous driver is always selected.
            dialect_str = str(dialect) if dialect is not None else ""
            sync_url = _sqlite_aiosqlite_to_sync_url(dialect_str)
            if not sync_url or sync_url == dialect_str:
                sync_url = _sqlite_aiosqlite_to_sync_url(db_url_str)

            try:
                rebuilt_dialect = make_url(sync_url)
            except Exception:  # pragma: no cover - defensive against bad URLs
                rebuilt_dialect = None
            else:
                rebuilt_set = getattr(rebuilt_dialect, "set", None)
                if callable(rebuilt_set):
                    sync_url = str(rebuilt_set(drivername="sqlite"))
            # ``sqlalchemy.engine.make_url`` may be stubbed to return a simple
            # string that lacks the ``set`` method. Fall back to rewriting the
            # URL manually so that we always select the synchronous driver.

            candidates = [str(dialect) if dialect is not None else "", db_url_str]

            sync_url = ""
            for candidate in candidates:
                if not candidate:
                    continue

                downgraded = _sqlite_aiosqlite_to_sync_url(candidate)
                if "+aiosqlite" in candidate or downgraded != candidate:
                    sync_url = downgraded
                    break

            if not sync_url:
                sync_url = _sqlite_aiosqlite_to_sync_url(db_url_str)
            # ``dialect`` can be a bare string when the SQLAlchemy dependency
            # is stubbed during tests. Fall back to deterministic string
            # rewriting so the synchronous driver is selected without relying
            # on the ``URL`` helpers.
            dialect_str = str(dialect)
            source = dialect_str if "+aiosqlite" in dialect_str else db_url_str
            sync_url = _sqlite_aiosqlite_to_sync_url(source)

        engine = create_engine(sync_url, echo=False, connect_args=_connect_args(sync_url))
        engine = _ensure_sync_engine(engine, sync_url)
        if create_schema:
            _create_schema_if_possible(engine, metadata)

            metadata = getattr(SQLModel, "metadata", None)
            if metadata is None or not hasattr(metadata, "create_all"):
                metadata = MetaData()
                setattr(SQLModel, "metadata", metadata)
            if hasattr(metadata, "create_all"):
                metadata.create_all(engine)

            _create_schema_if_possible(engine)


        return engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url_str))
    engine = _ensure_sync_engine(engine, db_url_str)
    if create_schema:
        _create_schema_if_possible(engine, metadata)

        metadata = getattr(SQLModel, "metadata", None)

        if metadata is None or not hasattr(metadata, "create_all"):
            metadata = MetaData()
            setattr(SQLModel, "metadata", metadata)
        if hasattr(metadata, "create_all"):
            metadata.create_all(engine)

        create_all = getattr(metadata, "create_all", None) if metadata is not None else None
        if callable(create_all):
            create_all(engine)

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
