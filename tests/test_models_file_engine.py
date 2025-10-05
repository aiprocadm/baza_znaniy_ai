"""Focused tests for synchronous engine helpers in ``app.models.file``."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from app.models import file as file_module

MetaData = file_module.MetaData
create_engine = file_module.create_engine
text = file_module.text


@contextmanager
def _stubbed_file_module() -> Iterator[Any]:
    """Reload ``app.models.file`` using the lightweight SQLAlchemy/SQLModel stubs."""

    from tests.stubs import sqlalchemy as sqlalchemy_stub
    from tests.stubs import sqlmodel as sqlmodel_stub

    stubbed_module: Any | None = None
    original_modules = {
        "sqlalchemy": sys.modules.get("sqlalchemy"),
        "sqlalchemy.engine": sys.modules.get("sqlalchemy.engine"),
        "sqlmodel": sys.modules.get("sqlmodel"),
        "app.models.file": sys.modules.get("app.models.file"),
    }
    app_models_pkg = sys.modules.get("app.models")

    try:
        sys.modules["sqlalchemy"] = sqlalchemy_stub
        if hasattr(sqlalchemy_stub, "engine_module"):
            sys.modules["sqlalchemy.engine"] = sqlalchemy_stub.engine_module
        sys.modules["sqlmodel"] = sqlmodel_stub

        sys.modules.pop("app.models.file", None)
        if app_models_pkg is not None and hasattr(app_models_pkg, "file"):
            delattr(app_models_pkg, "file")

        stubbed_module = importlib.import_module("app.models.file")
        if app_models_pkg is not None:
            setattr(app_models_pkg, "file", stubbed_module)

        stubbed_module.get_engine.cache_clear()
        yield stubbed_module
    finally:
        if stubbed_module is not None:
            stubbed_module.get_engine.cache_clear()

        sys.modules.pop("app.models.file", None)
        if original_modules["app.models.file"] is not None:
            sys.modules["app.models.file"] = original_modules["app.models.file"]
            if app_models_pkg is not None:
                setattr(app_models_pkg, "file", original_modules["app.models.file"])
        elif app_models_pkg is not None and hasattr(app_models_pkg, "file"):
            delattr(app_models_pkg, "file")

        for name in ("sqlalchemy", "sqlalchemy.engine", "sqlmodel"):
            original = original_modules[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

        original_file_module = original_modules["app.models.file"]
        if original_file_module is not None and hasattr(original_file_module, "get_engine"):
            original_file_module.get_engine.cache_clear()


@pytest.fixture(autouse=True)
def _clear_engine_cache() -> Iterator[None]:
    """Ensure engine cache isolation between tests."""

    file_module.get_engine.cache_clear()
    try:
        yield
    finally:
        file_module.get_engine.cache_clear()


def _connect_and_scalar(engine: Any) -> Any:
    with engine.connect() as connection:
        execution = connection.execute(text("SELECT 1"))
        scalar = getattr(execution, "scalar", None)
        return scalar() if callable(scalar) else execution


def test_get_engine_handles_missing_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_engine`` should rebuild metadata when it is ``None``."""

    db_path = tmp_path / "missing-metadata.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    original_metadata = file_module.SQLModel.metadata
    file_module.SQLModel.metadata = None  # type: ignore[assignment]

    engine = None
    try:
        engine = file_module.get_engine(create_schema=True)

        assert isinstance(file_module.SQLModel.metadata, MetaData)
        assert hasattr(engine, "dialect")
        assert hasattr(engine, "url")
        assert callable(getattr(engine, "dispose", None))
        assert callable(getattr(engine, "connect", None))
        assert _connect_and_scalar(engine) in {1, "SELECT 1"}
    finally:
        file_module.SQLModel.metadata = original_metadata


def test_get_engine_exposes_core_attributes(tmp_path: Path) -> None:
    """``get_engine`` must provide the synchronous SQLAlchemy API for SQLite URLs."""

    engine = file_module.get_engine(f"sqlite:///{tmp_path/'core.sqlite'}", create_schema=False)

    assert getattr(engine.dialect, "name") == "sqlite"
    assert getattr(engine.dialect, "driver") in {"sqlite", "pysqlite"}
    assert str(engine.url).startswith("sqlite")
    assert callable(engine.dispose)
    assert callable(engine.connect)
    assert _connect_and_scalar(engine) in {1, "SELECT 1"}


def test_stubbed_engine_exposes_fallback_api(tmp_path: Path) -> None:
    """Stubbed SQLAlchemy/SQLModel should still expose the sync API via fallbacks."""

    with _stubbed_file_module() as stubbed:
        engine = stubbed.get_engine(
            f"sqlite:///{tmp_path/'stubbed.sqlite'}",
            create_schema=False,
        )

        assert getattr(engine.dialect, "name") == "sqlite"
        assert getattr(engine.dialect, "driver") == "sqlite"
        assert str(engine.url).startswith("sqlite")
        assert callable(engine.dispose)
        assert callable(engine.connect)

        connection = engine.connect()
        try:
            result = connection.execute("SELECT 1")
            scalar = getattr(result, "scalar", None)
            assert callable(scalar)
            assert scalar() == "SELECT 1"
        finally:
            closer = getattr(connection, "close", None)
            if callable(closer):
                closer()


def test_get_engine_sqlite_aiosqlite_conversion(tmp_path: Path) -> None:
    """Async SQLite URLs should produce a synchronous SQLite engine with fallbacks."""

    engine = file_module.get_engine(
        f"sqlite+aiosqlite:///{tmp_path/'async.sqlite'}",
        create_schema=False,
    )

    assert getattr(engine.dialect, "name") == "sqlite"
    assert getattr(engine.dialect, "driver") in {"sqlite", "pysqlite"}
    assert str(engine.url).startswith("sqlite:///")
    assert callable(engine.dispose)
    assert callable(engine.connect)
    assert _connect_and_scalar(engine) in {1, "SELECT 1"}


def test_ensure_sync_engine_preserves_real_methods(tmp_path: Path) -> None:
    """Wrapping an engine must retain original ``connect`` and ``dispose`` callables."""

    target_url = f"sqlite:///{tmp_path/'preserve.sqlite'}"

    class DummyResult:
        def scalar(self) -> int:
            return 1

    class DummyConnection:
        def __enter__(self) -> "DummyConnection":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def execute(self, statement: Any) -> DummyResult:
            assert "SELECT 1" in str(statement)
            return DummyResult()

    class DummyEngine:
        def __init__(self) -> None:
            self.dialect = type("Dialect", (), {"name": "sqlite", "driver": "sqlite"})()
            self.url = target_url
            self.disposed = False

        def connect(self) -> DummyConnection:
            return DummyConnection()

        def dispose(self) -> None:
            self.disposed = True

    class WrappedEngine:
        def __init__(self, inner: DummyEngine) -> None:
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, item: str) -> Any:
            if item == "url":
                raise AttributeError("url attribute missing")
            return getattr(self._inner, item)

        def __setattr__(self, key: str, value: Any) -> None:
            if key == "url":
                raise AttributeError("cannot assign url")
            setattr(self._inner, key, value)

        def connect(self) -> DummyConnection:
            return self._inner.connect()

        def dispose(self) -> None:
            return self._inner.dispose()

    wrapped = WrappedEngine(DummyEngine())

    proxied = file_module._ensure_sync_engine(wrapped, target_url)

    assert proxied is not wrapped
    assert getattr(proxied.connect, "__self__", None) is wrapped
    assert getattr(proxied.dispose, "__self__", None) is wrapped
    assert _connect_and_scalar(proxied) in {1, "SELECT 1"}
    proxied.dispose()


def test_ensure_sync_engine_wraps_on_attribute_failure(tmp_path: Path) -> None:
    """Attribute assignment/read failures should produce a proxy exposing the sync API."""

    url = f"sqlite:///{tmp_path/'fallback.sqlite'}"

    class RejectingEngine:
        def __setattr__(self, key: str, value: Any) -> None:
            if key in {"dialect", "url", "dispose", "connect"}:
                raise AttributeError(key)
            object.__setattr__(self, key, value)

        def __getattr__(self, item: str) -> Any:
            raise AttributeError(item)

    rejecting = RejectingEngine()

    proxied = file_module._ensure_sync_engine(rejecting, url)

    assert hasattr(proxied, "dialect")
    assert getattr(proxied.dialect, "name") == "sqlite"
    assert getattr(proxied.dialect, "driver") == "sqlite"
    assert hasattr(proxied, "url")
    assert str(proxied.url) == url
    assert callable(proxied.dispose)
    assert callable(proxied.connect)

    connection = proxied.connect()
    try:
        result = connection.execute("SELECT 1")
        scalar = getattr(result, "scalar", None)
        assert callable(scalar)
        assert scalar() == "SELECT 1"
    finally:
        closer = getattr(connection, "close", None)
        if callable(closer):
            closer()

