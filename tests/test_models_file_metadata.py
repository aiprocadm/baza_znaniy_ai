"""Tests for SQLModel metadata utilities in app.models.file."""

from __future__ import annotations

from unittest.mock import patch

from sqlalchemy import MetaData

from app.models import file as file_module


def test_sanitize_metadata_tables_returns_metadata_when_facade_dict_present() -> None:
    """If the metadata.tables mapping is a FacadeDict, the metadata is returned unchanged."""

    metadata = MetaData()

    with patch.object(file_module, "collect_sqlmodel_tables", return_value=[]):
        with patch.object(file_module, "_rebuild_sqlmodel_metadata") as rebuild:
            result = file_module._sanitize_metadata_tables(metadata)  # type: ignore[attr-defined]

    rebuild.assert_not_called()
    assert result is metadata


def test_sanitize_metadata_tables_triggers_rebuild_for_invalid_mapping() -> None:
    """When metadata.tables is not a FacadeDict, the registry is rebuilt."""

    metadata = MetaData()
    sentinel = MetaData()

    with patch.object(file_module, "FacadeDict", tuple):
        with patch.object(
            file_module,
            "_rebuild_sqlmodel_metadata",
            return_value=sentinel,
        ) as rebuild:
            result = file_module._sanitize_metadata_tables(metadata)  # type: ignore[attr-defined]

    rebuild.assert_called_once_with()
    assert result is sentinel
