"""Tests for the synchronous engine helpers in ``app.models.file``."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text

from app.models import file as file_module


def test_get_engine_handles_missing_create_all(tmp_path) -> None:
    """``get_engine`` should ignore metadata without ``create_all``."""

    file_module.get_engine.cache_clear()

    original_metadata = file_module.SQLModel.metadata
    dummy_metadata = object()
    file_module.SQLModel.metadata = dummy_metadata  # type: ignore[assignment]
    try:
        engine = file_module.get_engine(
            f"sqlite:///{tmp_path/'missing-create-all.db'}",
            create_schema=True,
        )

        with engine.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value == 1 or str(value) == "SELECT 1"
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()



def test_ensure_sync_engine_preserves_real_methods(tmp_path) -> None:
    """Proxy should delegate ``connect``/``dispose`` to the wrapped engine."""

    sqlite_path = tmp_path / "proxy.sqlite"
    url = f"sqlite:///{sqlite_path}"

    engine = create_engine(url)
    connect_calls: list[bool] = []
    dispose_calls: list[bool] = []

    class WrappedEngine:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, item: str):
            if item == "url":
                raise AttributeError("url attribute missing")
            return getattr(self._inner, item)

        def __setattr__(self, key: str, value):
            if key == "_inner":
                object.__setattr__(self, key, value)
            elif key == "url":
                object.__setattr__(self, key, value)
            else:
                setattr(self._inner, key, value)

        def connect(self, *args, **kwargs):
            connect_calls.append(True)
            return self._inner.connect(*args, **kwargs)

        def dispose(self, *args, **kwargs):
            dispose_calls.append(True)
            return self._inner.dispose(*args, **kwargs)

    wrapped = WrappedEngine(engine)

    try:
        proxy = file_module._ensure_sync_engine(wrapped, url)

        # ``url`` fallback forces a proxy even though ``connect``/``dispose`` exist.
        assert proxy is not wrapped

        connect_method = proxy.connect
        dispose_method = proxy.dispose

        assert getattr(connect_method, "__self__", None) is wrapped
        assert getattr(dispose_method, "__self__", None) is wrapped

        with proxy.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            assert execution.scalar() == 1

        proxy.dispose()

        assert len(connect_calls) == 1
        assert len(dispose_calls) == 1
    finally:
        engine.dispose()
        if sqlite_path.exists():
            sqlite_path.unlink()


def test_get_engine_exposes_core_engine_attributes(tmp_path, monkeypatch):
    file_module.get_engine.cache_clear()
    db_path = tmp_path / "engine.sqlite"
    db_url = f"sqlite:///{db_path}"

    monkeypatch.setenv("DB_URL", db_url)

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

def test_get_engine_exposes_sync_attributes(tmp_path, monkeypatch) -> None:
    """Ensure ``get_engine`` returns an engine with expected sync API."""

    from app.models import file as file_module

    file_module.get_engine.cache_clear()
    db_path = Path(tmp_path) / "engine.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

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
            assert value == 1 or str(value) == "SELECT 1"

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

    engine = file_module.get_engine(create_schema=False)

    try:
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) in {"sqlite", "pysqlite"}

        dispose = getattr(engine, "dispose", None)
        assert callable(dispose)
        dispose()

        connection = engine.connect()
        try:
            execution = connection.execute("SELECT 1")
            scalar = getattr(execution, "scalar", None)
            assert callable(scalar)
            result = scalar()
            assert result == 1 or str(result) == "SELECT 1"
        finally:
            if hasattr(connection, "close"):
                connection.close()
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()



def test_get_engine_stub_engine_proxy(tmp_path, monkeypatch) -> None:
    """Regression: ensure the SQLModel stub engine receives fallback attributes."""

    from tests.stubs import sqlmodel as sqlmodel_stub

    from app.models import file as file_module

    file_module.get_engine.cache_clear()

    db_path = Path(tmp_path) / "stub-engine.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    monkeypatch.setattr(file_module, "create_engine", sqlmodel_stub.create_engine)
    monkeypatch.setattr(file_module, "SQLModel", sqlmodel_stub.SQLModel)

    engine = file_module.get_engine(create_schema=False)

    try:
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) == "sqlite"

        dispose = getattr(engine, "dispose", None)
        assert callable(dispose)
        dispose()

        connection = engine.connect()
        try:
            execution = connection.execute("SELECT 1")
            scalar = getattr(execution, "scalar", None)
            assert callable(scalar)
            result = scalar()
            assert result == 1 or str(result) == "SELECT 1"
        finally:
            if hasattr(connection, "close"):
                connection.close()
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()

def test_get_engine_aiosqlite_stub_regression(tmp_path) -> None:
    """``get_engine`` should cope with stubbed ``make_url`` lacking ``set``."""

    import importlib
    import sys

    from tests.stubs import sqlalchemy as sqlalchemy_stub
    from tests.stubs import sqlmodel as sqlmodel_stub

    original_sqlalchemy = sys.modules.get("sqlalchemy")
    original_sqlalchemy_engine = sys.modules.get("sqlalchemy.engine")
    original_sqlmodel = sys.modules.get("sqlmodel")
    original_file_module = sys.modules.get("app.models.file")
    app_models_pkg = sys.modules.get("app.models")

    target_url = f"sqlite+aiosqlite:///{tmp_path/'stub-aiosqlite.db'}"
    file_module = None

    try:
        sys.modules["sqlalchemy"] = sqlalchemy_stub
        if hasattr(sqlalchemy_stub, "engine_module"):
            sys.modules["sqlalchemy.engine"] = sqlalchemy_stub.engine_module
        sys.modules["sqlmodel"] = sqlmodel_stub
        sys.modules.pop("app.models.file", None)

        file_module = importlib.import_module("app.models.file")
        file_module.get_engine.cache_clear()

        engine = file_module.get_engine(target_url, create_schema=False)

        assert hasattr(engine, "dialect")
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) == "sqlite"

        assert hasattr(engine, "dispose")
        dispose = getattr(engine, "dispose")
        assert callable(dispose)
        dispose()

        assert hasattr(engine, "url")
        assert str(engine.url).startswith("sqlite:///")
    finally:
        sys.modules.pop("app.models.file", None)
        if file_module is not None:
            file_module.get_engine.cache_clear()
        if original_file_module is not None:
            sys.modules["app.models.file"] = original_file_module
            if app_models_pkg is not None:
                setattr(app_models_pkg, "file", original_file_module)
        elif app_models_pkg is not None and hasattr(app_models_pkg, "file"):
            delattr(app_models_pkg, "file")

        if original_sqlalchemy is None:
            sys.modules.pop("sqlalchemy", None)
        else:
            sys.modules["sqlalchemy"] = original_sqlalchemy

        if original_sqlalchemy_engine is None:
            sys.modules.pop("sqlalchemy.engine", None)
        else:
            sys.modules["sqlalchemy.engine"] = original_sqlalchemy_engine

        if original_sqlmodel is None:
            sys.modules.pop("sqlmodel", None)
        else:
            sys.modules["sqlmodel"] = original_sqlmodel

        if original_file_module is not None and hasattr(original_file_module, "get_engine"):
            original_file_module.get_engine.cache_clear()


