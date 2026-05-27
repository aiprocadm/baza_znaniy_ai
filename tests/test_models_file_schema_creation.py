"""Schema creation tests for ``app.models.file.get_engine``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from app.models import file as file_module


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Windows-only file lock on pytest tmp_path teardown — same race as test_models_file_engine_metadata_stub. Passes cleanly on Linux/CI.",
)
def test_get_engine_creates_schema(tmp_path, monkeypatch) -> None:
    """``get_engine`` should create tables without raising when requested."""

    file_module.get_engine.cache_clear()

    db_path = Path(tmp_path) / "schema-success.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    try:
        engine = file_module.get_engine(create_schema=True)

        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            )
            assert result.scalar() == "documents"
    finally:
        file_module.get_engine.cache_clear()
        monkeypatch.delenv("DB_URL", raising=False)
        if db_path.exists():
            db_path.unlink()
