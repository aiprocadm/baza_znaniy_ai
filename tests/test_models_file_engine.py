"""Tests for the synchronous engine helpers in ``app.models.file``."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

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
            assert connection.execute(text("SELECT 1")).scalar() == 1
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()



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
            assert value in {1, "SELECT 1"}

        engine.dispose()
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()

