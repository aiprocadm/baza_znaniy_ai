"""Tests for ``get_engine`` when SQLModel metadata lacks ``create_all``."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from app.models import file as file_module


def test_get_engine_warns_when_metadata_has_no_callable_create_all(
    tmp_path, caplog
) -> None:
    """``get_engine`` should warn and continue when ``create_all`` is missing."""

    file_module.get_engine.cache_clear()

    original_metadata = file_module.SQLModel.metadata

    class _MetadataStub:
        create_all = "not-callable"

    file_module.SQLModel.metadata = _MetadataStub()  # type: ignore[assignment]

    db_path = Path(tmp_path) / "missing-create-all.db"

    try:
        with caplog.at_level("WARNING", logger=file_module.logger.name):
            engine = file_module.get_engine(f"sqlite:///{db_path}", create_schema=True)

        assert "metadata.create_all is not callable" in caplog.text

        with engine.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()


def test_get_engine_skips_schema_when_metadata_is_none(tmp_path, monkeypatch) -> None:
    """``get_engine`` should return an engine with sync API even without metadata."""

    file_module.get_engine.cache_clear()

    original_metadata = file_module.SQLModel.metadata
    file_module.SQLModel.metadata = None  # type: ignore[assignment]

    db_path = Path(tmp_path) / "metadata-none.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    try:
        engine = file_module.get_engine(create_schema=True)

        assert hasattr(engine, "dialect")
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) in {"sqlite", "pysqlite"}

        assert hasattr(engine, "dispose")
        dispose = getattr(engine, "dispose")
        assert callable(dispose)
        dispose()

        assert hasattr(engine, "connect")
        connect = getattr(engine, "connect")
        assert callable(connect)
        with connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink(missing_ok=True)


def test_get_engine_creates_schema_when_metadata_callable(tmp_path, monkeypatch) -> None:
    """``get_engine`` should create database tables when metadata is valid."""

    file_module.get_engine.cache_clear()

    db_path = Path(tmp_path) / "metadata-valid.sqlite"

    metadata = file_module.SQLModel.metadata
    original_create_all = getattr(metadata, "create_all", None)
    assert callable(original_create_all)

    calls: list[object] = []

    def _counting_create_all(engine: object, _orig=original_create_all) -> object:
        calls.append(engine)
        return _orig(engine)

    monkeypatch.setattr(metadata, "create_all", _counting_create_all, raising=False)

    engine = file_module.get_engine(f"sqlite:///{db_path}", create_schema=True)

    assert calls == [engine]

    engine.dispose()

    file_module.get_engine.cache_clear()
