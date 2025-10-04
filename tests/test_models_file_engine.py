"""Tests for the synchronous engine helpers in ``app.models.file``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

STUBS_PATH = Path(__file__).resolve().parent / "stubs"

if "sqlalchemy" in sys.modules and getattr(
    sys.modules["sqlalchemy"], "__file__", ""
).endswith("tests/stubs/sqlalchemy/__init__.py"):
    sys.modules.pop("sqlalchemy", None)
    sys.modules.pop("sqlalchemy.engine", None)

removed_stub = False
if STUBS_PATH.exists():
    try:
        sys.path.remove(str(STUBS_PATH))
    except ValueError:
        pass
    else:
        removed_stub = True

importlib.import_module("sqlalchemy")

if removed_stub:
    sys.path.insert(0, str(STUBS_PATH))

from sqlalchemy import create_engine, text

from app.models import file as file_module


def test_get_engine_handles_missing_create_all(tmp_path, monkeypatch) -> None:
    """``get_engine`` should ignore metadata without ``create_all``."""

    file_module.get_engine.cache_clear()

    monkeypatch.setattr(file_module, "create_engine", create_engine)

    original_metadata = file_module.SQLModel.metadata
    dummy_metadata = object()
    file_module.SQLModel.metadata = dummy_metadata  # type: ignore[assignment]
    try:
        engine = file_module.get_engine(
            f"sqlite:///{tmp_path/'missing-create-all.db'}",
            create_schema=True,
        )

        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()



def test_get_engine_exposes_core_engine_attributes(tmp_path, monkeypatch):
    file_module.get_engine.cache_clear()
    db_path = tmp_path / "engine.sqlite"
    db_url = f"sqlite:///{db_path}"

    monkeypatch.setenv("DB_URL", db_url)
    monkeypatch.setattr(file_module, "create_engine", create_engine)

    try:
        engine = file_module.get_engine(create_schema=False)

        assert hasattr(engine, "dialect")
        assert getattr(engine.dialect, "name") == "sqlite"
        assert getattr(engine.dialect, "driver") in {"sqlite", "pysqlite"}

        assert hasattr(engine, "url")
        assert str(engine.url) == db_url

        dispose = getattr(engine, "dispose")
        assert callable(dispose)
        dispose()
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)


def test_get_engine_preserves_sqlalchemy_methods(tmp_path, monkeypatch):
    file_module.get_engine.cache_clear()
    db_path = tmp_path / "preserve.sqlite"
    db_url = f"sqlite:///{db_path}"

    monkeypatch.setenv("DB_URL", db_url)
    monkeypatch.setattr(file_module, "create_engine", create_engine)

    try:
        engine = file_module.get_engine(create_schema=False)
        dispose = engine.dispose
        connect = engine.connect

        assert callable(dispose)
        assert callable(connect)
        assert getattr(dispose, "__name__", "") == "dispose"
        assert getattr(connect, "__name__", "") == "connect"
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()

def test_get_engine_exposes_sync_attributes(tmp_path, monkeypatch) -> None:
    """Ensure ``get_engine`` returns an engine with expected sync API."""

    from app.models import file as file_module

    file_module.get_engine.cache_clear()
    db_path = Path(tmp_path) / "engine.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(file_module, "create_engine", create_engine)

    engine = file_module.get_engine(create_schema=False)

    try:
        assert hasattr(engine, "dialect")
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) in {"sqlite", "pysqlite"}
        assert hasattr(engine, "url")
        assert str(engine.url).startswith("sqlite")
        assert callable(engine.dispose)
        assert callable(engine.connect)

        with engine.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}

        engine.dispose()
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()


def test_get_engine_sqlite_regression(tmp_path, monkeypatch) -> None:
    """Regression: the fallback engine proxy exposes the sync SQLAlchemy API."""

    from app.models import file as file_module

    file_module.get_engine.cache_clear()

    db_path = Path(tmp_path) / "regression.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(file_module, "create_engine", create_engine)

    engine = file_module.get_engine(create_schema=False)

    try:
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) in {"sqlite", "pysqlite"}

        dispose = getattr(engine, "dispose", None)
        assert callable(dispose)
        dispose()

        connection = engine.connect()
        try:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            assert callable(scalar)
            assert scalar() in {1, "SELECT 1"}
        finally:
            if hasattr(connection, "close"):
                connection.close()
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()


def test_ensure_sync_engine_proxy_preserves_original_methods(tmp_path) -> None:
    """When wrapping an engine the proxy should retain SQLAlchemy callables."""

    base_engine = create_engine(f"sqlite:///{tmp_path/'wrapped.sqlite'}")

    class MissingDialectEngine:
        def __init__(self, original):
            self._original = original
            self.dispose = original.dispose
            self.connect = original.connect

        @property
        def dialect(self):  # pragma: no cover - exercised indirectly
            raise AttributeError("dialect unavailable")

        @property
        def url(self):  # pragma: no cover - exercised indirectly
            raise AttributeError("url unavailable")

        def __getattr__(self, item):
            return getattr(self._original, item)

    wrapped = MissingDialectEngine(base_engine)
    original_dispose = wrapped.dispose
    original_connect = wrapped.connect

    proxied = file_module._ensure_sync_engine(wrapped, str(base_engine.url))

    try:
        assert proxied.dispose is original_dispose
        assert proxied.connect is original_connect

        proxied.dispose()

        with proxied.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
    finally:
        base_engine.dispose()

