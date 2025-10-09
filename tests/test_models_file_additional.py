from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, MetaData, Table

from app.models import file as file_module


def test_record_sqlmodel_metadata_health_emits_alert(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_state(metadata, *, origin):
        return False, "corrupt"

    def fake_alert(*, origin, reason):
        calls.append((origin, reason))

    monkeypatch.setattr(file_module, "record_sqlmodel_metadata_state", fake_state)
    monkeypatch.setattr(file_module, "record_sqlmodel_metadata_alert", fake_alert)

    file_module._record_sqlmodel_metadata_health(object(), origin="unit-test")

    assert calls == [("unit-test", "corrupt")]


def test_record_sqlmodel_metadata_health_handles_metric_errors(monkeypatch):
    def failing_state(metadata, *, origin):
        raise RuntimeError("boom")

    monkeypatch.setattr(file_module, "record_sqlmodel_metadata_state", failing_state)

    file_module._record_sqlmodel_metadata_health(object(), origin="unit-test")


def test_ensure_sqlmodel_metadata_rejects_invalid_metadata():
    class BadMetadata:
        create_all = "not-callable"

    with pytest.raises(RuntimeError):
        file_module._ensure_sqlmodel_metadata(BadMetadata())


def test_collect_sqlmodel_tables_includes_declared_models():
    metadata = MetaData()
    temp_table = Table(
        "temp_collect_table",
        metadata,
        Column("id", Integer, primary_key=True),
    )

    class TempModel(file_module.SQLModel):
        __table__ = temp_table

    tables = file_module._collect_sqlmodel_tables()

    assert (TempModel, temp_table) in tables


