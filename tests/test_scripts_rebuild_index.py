"""Tests for the rebuild_index script logic."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


# Provide lightweight stubs for optional dependencies imported during module load.
if "sqlmodel" not in sys.modules:
    sqlmodel_stub = ModuleType("sqlmodel")

    class Session:  # pragma: no cover - simple stand-in for import-time reference
        pass

    def _expression(*_args, **_kwargs):  # pragma: no cover - helper for import-time usage
        return None

    sqlmodel_stub.Session = Session
    sqlmodel_stub.delete = _expression
    sqlmodel_stub.select = _expression
    sys.modules["sqlmodel"] = sqlmodel_stub

if "prometheus_client" not in sys.modules:
    prometheus_stub = ModuleType("prometheus_client")

    class _Metric:  # pragma: no cover - helper used only for import-time metrics
        def labels(self, **_labels):
            return self

        def inc(self, *_args, **_kwargs):
            return None

        def observe(self, *_args, **_kwargs):
            return None

    def _factory(*_args, **_kwargs):
        return _Metric()

    prometheus_stub.Counter = _factory
    prometheus_stub.Histogram = _factory
    sys.modules["prometheus_client"] = prometheus_stub

if "scripts.ingest_path" not in sys.modules:
    ingest_stub = ModuleType("scripts.ingest_path")

    def _stub_ingest_path(_target: Path) -> int:  # pragma: no cover - replaced during tests
        raise AssertionError("ingest_path should be patched in tests")

    ingest_stub.ingest_path = _stub_ingest_path
    sys.modules["scripts.ingest_path"] = ingest_stub

import scripts.rebuild_index as rebuild_module


@pytest.fixture()
def mock_dependencies(monkeypatch):
    """Mock expensive dependencies used by the rebuild_index script."""

    settings = SimpleNamespace(data_dir="/default/data")
    vector_store_reset_calls: list[None] = []
    ingest_calls: list[Path] = []

    class DummyVectorStore:
        def reset_collection(self) -> None:
            vector_store_reset_calls.append(None)

    def fake_get_settings():
        return settings

    def fake_get_vector_store(provided_settings):
        assert provided_settings is settings
        return DummyVectorStore()

    def fake_ingest_path(target: Path) -> int:
        ingest_calls.append(target)
        return 123

    monkeypatch.setattr(rebuild_module, "get_settings", fake_get_settings)
    monkeypatch.setattr(rebuild_module, "get_vector_store", fake_get_vector_store)
    monkeypatch.setattr(rebuild_module, "ingest_path", fake_ingest_path)

    return settings, vector_store_reset_calls, ingest_calls


def test_rebuild_index_uses_explicit_path(mock_dependencies):
    _settings, reset_calls, ingest_calls = mock_dependencies
    explicit_path = Path("/custom/path")

    result = rebuild_module.rebuild_index(explicit_path)

    assert result == 123
    assert reset_calls == [None]
    assert ingest_calls == [explicit_path]


def test_rebuild_index_defaults_to_settings_path(mock_dependencies):
    settings, reset_calls, ingest_calls = mock_dependencies

    result = rebuild_module.rebuild_index()

    assert result == 123
    assert reset_calls == [None]
    assert ingest_calls == [Path(settings.data_dir)]
