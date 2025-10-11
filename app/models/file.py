"""SQLModel definitions for document ingestion artifacts."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Optional

import sqlite3
from urllib.parse import unquote, urlparse

from sqlalchemy import Column, JSON, MetaData, Text, UniqueConstraint, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import create_async_engine
try:  # pragma: no cover - optional dependency during lightweight test runs
    from sqlalchemy.exc import NoSuchModuleError as _SQLAlchemyNoSuchModuleError
except Exception:  # pragma: no cover - fallback when SQLAlchemy is stubbed
    class _SQLAlchemyNoSuchModuleError(Exception):
        """Placeholder exception when SQLAlchemy is unavailable."""

        pass

try:  # pragma: no cover - optional dependency during lightweight test runs
    from sqlalchemy.util import FacadeDict as _SQLAlchemyFacadeDict
except Exception:  # pragma: no cover - exercised when SQLAlchemy is stubbed out
    class _SQLAlchemyFacadeDict(dict):
        """Fallback mapping used when ``sqlalchemy.util.FacadeDict`` is unavailable."""

        pass

    FacadeDict = _SQLAlchemyFacadeDict  # type: ignore[assignment]
else:
    FacadeDict = _SQLAlchemyFacadeDict
try:  # pragma: no cover - optional dependency during lightweight test runs
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - fallback when SQLAlchemy ORM is absent
    sessionmaker = None  # type: ignore[assignment]
from sqlmodel import Field, SQLModel, Session, create_engine

from app.core.datetime_utils import utc_now
from app.observability.metrics import (
    record_sqlmodel_metadata_alert,
    record_sqlmodel_metadata_state,
)
from app.models.engine_guard import SyncEngineGuard, is_fallback_value, mark_fallback
from app.models.entities import JobRecord, SettingRecord
from app.models.sqlmodel_compat import (
    collect_sqlmodel_tables,
    install_stub_model_initializers,
)
from app.models.sqlite_datetime import register_sqlite_datetime_support


logger = logging.getLogger(__name__)


register_sqlite_datetime_support()


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


def _sanitize_metadata_tables(metadata: MetaData) -> MetaData:
    """Return metadata with an intact table registry, rebuilding it if required."""

    tables_attr = getattr(metadata, "tables", None)

    if not isinstance(tables_attr, FacadeDict):
        logger.warning(
            "SQLModel metadata tables registry is corrupt; rebuilding metadata"
        )
        return _rebuild_sqlmodel_metadata()

    invalid_keys: list[str] = []
    for table_name, table in list(tables_attr.items()):
        if getattr(table, "metadata", None) is not metadata:
            invalid_keys.append(table_name)
            continue
        if not getattr(table, "name", None):
            invalid_keys.append(table_name)

    for table_name in invalid_keys:
        tables_attr.pop(table_name, None)

    if invalid_keys:
        logger.warning(
            "Removed %s invalid table entries from SQLModel metadata", len(invalid_keys)
        )

    if not tables_attr:
        snapshot = collect_sqlmodel_tables()
        if snapshot:
            return _rebuild_sqlmodel_metadata(snapshot)

    return metadata


def _rebuild_sqlmodel_metadata(
    snapshot: list[tuple[type[Any], Any]] | None = None,
) -> MetaData:
    """Recreate ``SQLModel.metadata`` from known model definitions."""

    if snapshot is None:
        snapshot = collect_sqlmodel_tables()

    new_metadata = MetaData()

    try:
        setattr(SQLModel, "metadata", new_metadata)
    except Exception:  # pragma: no cover - defensive best-effort assignment
        logger.warning(
            "Failed to assign rebuilt metadata to SQLModel; using local instance"
        )

    for model_cls, table in snapshot:
        migrate = getattr(table, "to_metadata", None)
        if not callable(migrate):
            migrate = getattr(table, "tometadata", None)
        if not callable(migrate):
            logger.debug(
                "Model %s table %r cannot migrate to rebuilt metadata",
                getattr(model_cls, "__name__", repr(model_cls)),
                getattr(table, "name", repr(table)),
            )
            continue

        try:
            new_table = migrate(new_metadata)
        except Exception:  # pragma: no cover - defensive migration
            logger.exception(
                "Failed to migrate table %s during metadata rebuild",
                getattr(table, "name", repr(table)),
            )
            continue

        try:
            setattr(model_cls, "__table__", new_table)
        except Exception:  # pragma: no cover - best-effort attribute update
            logger.debug(
                "Unable to refresh __table__ for %s after metadata rebuild",
                getattr(model_cls, "__name__", repr(model_cls)),
                exc_info=True,
            )

    return new_metadata


def _recover_sqlmodel_metadata(*, reason: str) -> MetaData:
    """Rebuild SQLModel metadata when the current instance is unusable."""

    logger.warning("Rebuilding SQLModel metadata due to %s", reason)
    rebuilt = _rebuild_sqlmodel_metadata()
    return _sanitize_metadata_tables(rebuilt)


def _ensure_metadata_with_recovery(metadata: Any | None, *, reason: str) -> MetaData:
    """Validate SQLModel metadata, rebuilding it on failure."""

    try:
        return _ensure_sqlmodel_metadata(metadata)
    except RuntimeError:
        logger.debug(
            "SQLModel metadata validation failed during %s; attempting recovery", reason
        )
        return _recover_sqlmodel_metadata(reason=reason)


def _ensure_sqlmodel_metadata(metadata: Any | None) -> MetaData:
    """Return a valid SQLModel metadata instance, creating one if required."""

    if isinstance(metadata, MetaData):
        return _sanitize_metadata_tables(metadata)

    candidate = getattr(SQLModel, "metadata", None) if metadata is None else metadata
    if candidate is not None and not isinstance(candidate, MetaData):
        create_all_attr = getattr(candidate, "create_all", None)
        if create_all_attr is not None and not callable(create_all_attr):
            logger.error(
                "SQLModel.metadata.create_all is not callable; refusing schema creation"
            )
            raise RuntimeError(
                "SQLModel metadata initialisation failed: metadata.create_all is not callable"
            )

        logger.error(
            "SQLModel metadata object is not an instance of sqlalchemy.MetaData"
        )
        raise RuntimeError(
            "SQLModel metadata initialisation failed: metadata must be an instance of sqlalchemy.MetaData"
        )

    candidate = getattr(SQLModel, "metadata", None)
    if isinstance(candidate, MetaData):
        return _sanitize_metadata_tables(candidate)

    fallback = MetaData()
    try:
        setattr(SQLModel, "metadata", fallback)
    except Exception:  # pragma: no cover - best-effort assignment
        logger.warning(
            "Unable to attach fallback SQLModel metadata; schema creation may fail"
        )
    return _sanitize_metadata_tables(fallback)


def _collect_sqlmodel_tables() -> list[tuple[type[Any], Any]]:
    """Return pairs of ``(model_class, table)`` registered with SQLModel."""

    try:
        subclasses = list(getattr(SQLModel, "__subclasses__", lambda: [])())
    except Exception:  # pragma: no cover - defensive fallback when introspection fails
        return []

    tables: list[tuple[type[Any], Any]] = []
    seen: set[type[Any]] = set()

    def _visit(model: type[Any]) -> None:
        if model in seen:
            return
        seen.add(model)

        table = getattr(model, "__table__", None)
        if table is not None:
            tables.append((model, table))

        for child in getattr(model, "__subclasses__", lambda: [])():
            _visit(child)

    for model_cls in subclasses:
        _visit(model_cls)

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
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


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
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


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
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


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
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


install_stub_model_initializers([DocumentRecord, FileRecord, PageRecord, ChunkRecord])


def _connect_args(url: str) -> dict[str, object]:
    if url.startswith("sqlite"):
        return {
            "check_same_thread": False,
            "timeout": 5.0,
        }
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
        dbapi_connection = engine.raw_connection()
    except Exception:
        logger.debug(
            "SQLite engine connection is unavailable for PRAGMA configuration",
            exc_info=True,
        )
    else:
        try:
            cursor = dbapi_connection.cursor()
            for statement in pragmas:
                cursor.execute(statement)
            dbapi_connection.commit()
        except Exception:
            logger.debug(
                "Failed to apply SQLite PRAGMAs via engine connection",
                exc_info=True,
            )
        finally:
            try:
                dbapi_connection.close()
            except Exception:
                logger.debug("Failed to close SQLite engine connection", exc_info=True)

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


class _EngineFallbackResult:
    """Lightweight result wrapper mirroring ``Result.scalar``."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _EngineFallbackConnection:
    """Context manager compatible connection used by sync fallbacks."""

    __slots__ = ()

    def __enter__(self) -> "_EngineFallbackConnection":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False

    def execute(self, statement: Any) -> _EngineFallbackResult:
        return _EngineFallbackResult(statement)


def _build_engine_fallback(url: str) -> Engine:
    """Create a lightweight engine replacement when SQLAlchemy is unavailable."""

    fallback_engine = SimpleNamespace()

    def _fallback_connect(*_: Any, **__: Any) -> _EngineFallbackConnection:
        return _EngineFallbackConnection()

    def _fallback_dispose(*_: Any, **__: Any) -> None:
        return None

    mark_fallback(_fallback_connect)
    mark_fallback(_fallback_dispose)

    fallback_engine.connect = _fallback_connect  # type: ignore[attr-defined]
    fallback_engine.dispose = _fallback_dispose  # type: ignore[attr-defined]
    fallback_engine.dialect = _synthesise_dialect(url)  # type: ignore[attr-defined]
    try:
        fallback_engine.url = make_url(url)  # type: ignore[attr-defined]
    except (_SQLAlchemyNoSuchModuleError, ModuleNotFoundError):
        fallback_engine.url = url  # type: ignore[attr-defined]
    except Exception:
        fallback_engine.url = url  # type: ignore[attr-defined]

    mark_fallback(fallback_engine)
    return _ensure_engine_surface(fallback_engine, url)


def _synthesise_dialect(url: str) -> Any:
    """Create a minimal dialect object exposing ``name``/``driver`` attributes."""

    try:
        parsed = make_url(url)
    except Exception:
        parsed = url

    backend: str
    driver: str

    if parsed is not None and hasattr(parsed, "get_backend_name"):
        backend = parsed.get_backend_name()  # type: ignore[assignment]
        get_driver = getattr(parsed, "get_driver_name", lambda: backend)
        try:
            driver = get_driver() or backend
        except (_SQLAlchemyNoSuchModuleError, ModuleNotFoundError):
            logger.debug(
                "Unable to resolve SQLAlchemy driver for %s; defaulting to backend", url
            )
            driver = backend
    else:
        scheme = str(parsed).split(":", 1)[0]
        if "+" in scheme:
            backend, driver = scheme.split("+", 1)
        else:
            backend = driver = scheme or "sqlite"

    dialect = SimpleNamespace(name=backend or "sqlite", driver=driver or backend)
    setattr(dialect, "is_async", False)
    mark_fallback(dialect)
    return dialect


def _ensure_engine_surface(engine: Engine, url: str) -> Engine:
    """Guarantee essential SQLModel-compatible attributes on ``engine``."""

    try:
        dialect = engine.dialect  # type: ignore[attr-defined]
    except AttributeError:
        dialect = None

    if not (hasattr(dialect, "name") and hasattr(dialect, "driver")):
        fallback_dialect = _synthesise_dialect(url)
        try:
            setattr(engine, "dialect", fallback_dialect)
        except Exception:
            pass
        else:
            dialect = fallback_dialect

    has_url = False
    try:
        candidate_url = engine.url  # type: ignore[attr-defined]
    except AttributeError:
        candidate_url = None
    else:
        has_url = candidate_url is not None

    if not has_url:
        try:
            fallback_url = make_url(url)
        except Exception:
            fallback_url = url
        try:
            setattr(engine, "url", fallback_url)
        except Exception:
            pass

    try:
        dispose = engine.dispose  # type: ignore[attr-defined]
    except AttributeError:
        dispose = None

    if not callable(dispose):

        def _noop_dispose(*_: Any, **__: Any) -> None:
            return None

        mark_fallback(_noop_dispose)
        try:
            setattr(engine, "dispose", _noop_dispose)
        except Exception:
            pass

    try:
        connect = engine.connect  # type: ignore[attr-defined]
    except AttributeError:
        connect = None

    if not callable(connect):

        def _fallback_connect(*_: Any, **__: Any) -> _EngineFallbackConnection:
            return _EngineFallbackConnection()

        mark_fallback(_fallback_connect)
        try:
            setattr(engine, "connect", _fallback_connect)
        except Exception:
            pass

    return engine


def _ensure_sync_engine(engine: Engine, url: str) -> Engine:
    """Return an engine guaranteed to expose the synchronous SQLModel API."""

    guarded = SyncEngineGuard(engine, url).ensure_sync()
    dialect = getattr(guarded, "dialect", None)
    if dialect is not None and is_fallback_value(dialect):
        driver = getattr(dialect, "driver", None)
        if isinstance(driver, str) and driver.lower() == "pysqlite":
            try:
                setattr(dialect, "driver", "sqlite")
            except Exception:
                pass
    return _ensure_engine_surface(guarded, url)


def _collect_sqlmodel_tables() -> list[tuple[type[Any], Any]]:
    """Return a snapshot of currently declared SQLModel tables."""

    tables: list[tuple[type[Any], Any]] = []

    try:
        subclasses = list(SQLModel.__subclasses__())
    except Exception:
        return tables

    for model_cls in subclasses:
        table = getattr(model_cls, "__table__", None)
        if table is not None:
            tables.append((model_cls, table))

    return tables


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
        candidate = metadata if metadata is not None else getattr(SQLModel, "metadata", None)
        if candidate is not None and not isinstance(candidate, MetaData):
            create_all_attr = getattr(candidate, "create_all", None)
            if create_all_attr is not None and not callable(create_all_attr):
                logger.error(
                    "SQLModel.metadata.create_all is not callable; skipping schema creation"
                )
                return None
        meta = candidate if isinstance(candidate, MetaData) else None

        if meta is None:
            tables_snapshot = collect_sqlmodel_tables()
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
            "SQLModel.metadata.create_all is not callable; skipping schema creation"
        )
        return None

    try:
        create_all(engine)
    except Exception:
        logger.exception("Schema creation failed; continuing without database schema")
        return None
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

    metadata = _ensure_metadata_with_recovery(
        getattr(SQLModel, "metadata", None), reason="engine initialisation"
    )

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

        try:
            engine = create_engine(
                engine_url, echo=False, connect_args=_connect_args(engine_url)
            )
        except (_SQLAlchemyNoSuchModuleError, ModuleNotFoundError):
            logger.error(
                "SQLAlchemy is missing the '%s' dialect; falling back to stub engine",
                engine_url,
                exc_info=True,
            )
            engine = _build_engine_fallback(engine_url)
    else:
        try:
            engine = create_engine(
                db_url, echo=False, connect_args=_connect_args(db_url_str)
            )
        except (_SQLAlchemyNoSuchModuleError, ModuleNotFoundError):
            logger.error(
                "SQLAlchemy is missing the '%s' dialect; falling back to stub engine",
                db_url_str,
                exc_info=True,
            )
            engine = _build_engine_fallback(db_url_str)

    _maybe_configure_sqlite_engine(engine, engine_url)
    engine = SyncEngineGuard(engine, engine_url).ensure_sync()
    engine = _ensure_engine_surface(engine, engine_url)

    if create_schema:
        meta = _ensure_metadata_with_recovery(metadata, reason="schema creation")
        create_all = getattr(meta, "create_all", None)
        if callable(create_all) and not is_fallback_value(engine):
            create_all(engine)
        elif not callable(create_all):
            logger.error(
                "SQLModel.metadata.create_all is not callable; cannot create schema"
            )
            raise RuntimeError(
                "SQLModel metadata initialisation failed: metadata.create_all is not callable"
            )
        else:
            logger.warning(
                "Skipping schema creation because SQLAlchemy engine is operating in fallback mode"
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
    "MetaData",
    "create_engine",
    "text",
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
    "_ensure_sync_engine",
]
