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

        assert "create_all is unavailable" in caplog.text

        with engine.connect() as connection:
            execution = connection.execute(text("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}
    finally:
        file_module.SQLModel.metadata = original_metadata
        file_module.get_engine.cache_clear()
