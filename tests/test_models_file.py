"""Focused tests for the SQLModel engine helpers when stubs are active."""

from __future__ import annotations

import sys
from pathlib import Path


def _restore_module(name: str, original: object | None) -> None:
    if original is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = original


def test_get_engine_with_sqlmodel_stub(monkeypatch, tmp_path: Path) -> None:
    """Ensure the engine proxy surfaces the sync SQLAlchemy API under the stub."""

    from tests.stubs import sqlmodel as sqlmodel_stub

    # Preserve the real module (if present) before swapping in the stub.
    original_sqlmodel = sys.modules.get("sqlmodel")
    original_file_module = sys.modules.get("app.models.file")

    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path/'stub.db'}")

    app_models_pkg = sys.modules.get("app.models")

    file_module = None

    try:
        sys.modules["sqlmodel"] = sqlmodel_stub
        sys.modules.pop("app.models.file", None)

        import importlib

        file_module = importlib.import_module("app.models.file")
        file_module.get_engine.cache_clear()

        engine = file_module.get_engine(create_schema=False)

        assert engine.dialect.name == "sqlite"
        assert engine.dialect.driver == "sqlite"
        assert str(engine.url).startswith("sqlite:")

        with engine.connect() as connection:
            assert connection.execute("SELECT 1").scalar() == "SELECT 1"

        engine.dispose()
    finally:
        if file_module is not None:
            file_module.get_engine.cache_clear()
        sys.modules.pop("app.models.file", None)
        if original_file_module is not None:
            sys.modules["app.models.file"] = original_file_module
            if app_models_pkg is not None:
                setattr(app_models_pkg, "file", original_file_module)
        if original_sqlmodel is None:
            sys.modules.pop("sqlmodel", None)
        else:
            sys.modules["sqlmodel"] = original_sqlmodel
            if original_file_module is not None and hasattr(original_file_module, "get_engine"):
                original_file_module.get_engine.cache_clear()
        if (
            original_file_module is None
            and app_models_pkg is not None
            and hasattr(app_models_pkg, "file")
        ):
            delattr(app_models_pkg, "file")
        monkeypatch.delenv("DB_URL", raising=False)


def test_get_engine_sqlite_aiosqlite_with_sqlalchemy_stub(monkeypatch) -> None:
    """The async sqlite URL should fall back to a sync variant when ``set`` is missing."""

    import importlib

    sqlite_async_url = "sqlite+aiosqlite:///./var/data/kb.sqlite"

    app_models_pkg = sys.modules.get("app.models")
    original_sqlmodel = sys.modules.get("sqlmodel")
    original_sqlalchemy = sys.modules.get("sqlalchemy")
    original_file_module = sys.modules.get("app.models.file")

    file_module = None

    try:
        sqlmodel_stub = importlib.import_module("tests.stubs.sqlmodel")
        sqlalchemy_stub = importlib.import_module("tests.stubs.sqlalchemy")

        sys.modules["sqlmodel"] = sqlmodel_stub
        sys.modules["sqlalchemy"] = sqlalchemy_stub
        sys.modules.pop("app.models.file", None)
        if app_models_pkg is not None and hasattr(app_models_pkg, "file"):
            delattr(app_models_pkg, "file")

        monkeypatch.setenv("DB_URL", sqlite_async_url)

        file_module = importlib.import_module("app.models.file")
        file_module.get_engine.cache_clear()

        engine = file_module.get_engine(create_schema=False)

        assert hasattr(engine, "dialect")
        assert hasattr(engine, "url")
        assert str(engine.url).startswith("sqlite:")
        assert engine.dialect.name == "sqlite"
        assert engine.dialect.driver == "sqlite"

        with engine.connect() as connection:
            assert connection.execute("SELECT 1").scalar() == "SELECT 1"

        engine.dispose()
    finally:
        if file_module is not None:
            file_module.get_engine.cache_clear()
        _restore_module("app.models.file", original_file_module)
        if (
            app_models_pkg is not None
            and hasattr(app_models_pkg, "file")
            and original_file_module is None
        ):
            delattr(app_models_pkg, "file")
        _restore_module("sqlalchemy", original_sqlalchemy)
        _restore_module("sqlmodel", original_sqlmodel)
        monkeypatch.delenv("DB_URL", raising=False)
