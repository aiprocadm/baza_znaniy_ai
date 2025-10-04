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

from sqlalchemy import MetaData, text


from app.models import file as file_module



def test_get_engine_handles_missing_create_all(tmp_path, monkeypatch) -> None:
    """``get_engine`` should ignore metadata without ``create_all``."""

def test_get_engine_handles_missing_create_all(tmp_path) -> None:
    """``get_engine`` should replace unusable metadata before schema creation."""


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

        assert isinstance(file_module.SQLModel.metadata, MetaData)

        with engine.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()



def test_get_engine_initializes_metadata_when_missing(tmp_path) -> None:
    file_module.get_engine.cache_clear()

def test_get_engine_handles_missing_metadata(tmp_path, monkeypatch) -> None:
    """``get_engine`` should tolerate ``SQLModel.metadata`` being ``None``."""

    file_module.get_engine.cache_clear()

    db_path = Path(tmp_path) / "missing-metadata.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")


    original_metadata = file_module.SQLModel.metadata
    file_module.SQLModel.metadata = None  # type: ignore[assignment]

    try:

        engine = file_module.get_engine(
            f"sqlite:///{tmp_path/'missing-metadata.db'}",
            create_schema=True,
        )

        assert isinstance(file_module.SQLModel.metadata, MetaData)

        with engine.connect() as connection:

        engine = file_module.get_engine(create_schema=True)

        assert getattr(engine, "dialect", None) is not None
        assert getattr(engine, "url", None) is not None

        dispose = getattr(engine, "dispose", None)
        connect = getattr(engine, "connect", None)

        assert callable(dispose)
        assert callable(connect)

        with connect() as connection:  # type: ignore[operator]

            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}



        dispose()

    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()



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

def test_get_engine_downgrades_aiosqlite(tmp_path, monkeypatch) -> None:


def test_get_engine_stub_engine_proxy(tmp_path, monkeypatch) -> None:
    """Regression: ensure the SQLModel stub engine receives fallback attributes."""

    from tests.stubs import sqlmodel as sqlmodel_stub


    from app.models import file as file_module

    file_module.get_engine.cache_clear()


    db_path = Path(tmp_path) / "async.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{db_path}")

    db_path = Path(tmp_path) / "stub-engine.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    monkeypatch.setattr(file_module, "create_engine", sqlmodel_stub.create_engine)
    monkeypatch.setattr(file_module, "SQLModel", sqlmodel_stub.SQLModel)


    engine = file_module.get_engine(create_schema=False)

    try:
        assert getattr(engine.dialect, "name", None) == "sqlite"

        assert getattr(engine.dialect, "driver", None) in {"sqlite", "pysqlite"}
        assert str(engine.url).startswith("sqlite:")
        assert callable(engine.dispose)
        assert callable(engine.connect)

        with engine.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}

        assert getattr(engine.dialect, "driver", None) == "sqlite"

        dispose = getattr(engine, "dispose", None)
        assert callable(dispose)
        dispose()

        connection = engine.connect()
        try:
            execution = connection.execute("SELECT 1")
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



def test_ensure_sync_engine_preserves_sqlalchemy_methods(tmp_path) -> None:
    """``_ensure_sync_engine`` should preserve existing SQLAlchemy callables."""

    from app.models import file as file_module

    db_path = tmp_path / "preserve.sqlite"
    url = f"sqlite:///{db_path}"

    class _DummyResult:
        def __init__(self, value: int) -> None:
            self._value = value

        def scalar(self) -> int:
            return self._value

    class _DummyConnection:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> "_DummyConnection":  # pragma: no cover - mirrors SQLAlchemy
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - mirrors SQLAlchemy
            self.close()
            return False

        def exec_driver_sql(self, statement: str) -> _DummyResult:
            assert statement == "SELECT 1"
            return _DummyResult(1)

        def close(self) -> None:
            self.closed = True

    class EngineWithoutDialect:
        def __init__(self, target_url: str) -> None:
            self.url = target_url
            self._disposed = False

        @property
        def dialect(self) -> None:  # pragma: no cover - exercised via getattr
            raise AttributeError("dialect unavailable")

        def connect(self) -> _DummyConnection:
            return _DummyConnection()

        def dispose(self) -> None:
            self._disposed = True

    base_engine = EngineWithoutDialect(url)

    original_connect = base_engine.connect
    original_dispose = base_engine.dispose

    proxied = file_module._ensure_sync_engine(base_engine, url)

    assert proxied is not base_engine
    assert getattr(proxied.connect, "__self__", None) is base_engine
    assert getattr(proxied.connect, "__func__", None) is getattr(original_connect, "__func__", None)
    assert getattr(proxied.dispose, "__self__", None) is base_engine
    assert getattr(proxied.dispose, "__func__", None) is getattr(original_dispose, "__func__", None)

    with proxied.connect() as connection:
        execution = connection.exec_driver_sql("SELECT 1")
        scalar = getattr(execution, "scalar", None)
        value = scalar() if callable(scalar) else execution
        assert value in {1, "SELECT 1"}

    proxied.dispose()

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



