"""Tests for the synchronous engine helpers in ``app.models.file``."""

from __future__ import annotations

from sqlalchemy import text


def test_get_engine_handles_missing_create_all(tmp_path) -> None:
    """``get_engine`` should ignore metadata without ``create_all``."""

    from app.models import file as file_module

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
