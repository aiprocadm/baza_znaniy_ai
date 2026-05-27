"""Schema creation tests for ``app.models.file.get_engine``."""

from __future__ import annotations

import gc
from pathlib import Path

from sqlalchemy import text

from app.models import file as file_module


def test_get_engine_creates_schema(tmp_path, monkeypatch) -> None:
    """``get_engine`` should create tables without raising when requested."""

    file_module.get_engine.cache_clear()

    db_path = Path(tmp_path) / "schema-success.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    engine = None
    try:
        engine = file_module.get_engine(create_schema=True)

        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            )
            assert result.scalar() == "documents"
    finally:
        if engine is not None:
            engine.dispose()
        engine = None
        # On Windows the SQLite DLL holds the file handle until the
        # connection object is collected — even after engine.dispose().
        gc.collect()
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()
