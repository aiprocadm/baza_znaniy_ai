"""Tests for extended /api/kb/health fields."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_with_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import config as _cfg

    if hasattr(_cfg, "get_settings"):
        _cfg.get_settings.cache_clear()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    store.add_document("doc1", text="alpha " * 50)
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app, tmp_path


def test_health_includes_kb_stats(app_with_store):
    app, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/api/kb/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "kb_stats" in data
    stats = data["kb_stats"]
    assert stats["documents_count"] >= 1
    assert stats["chunks_count"] >= 1
    assert stats["db_size_bytes"] > 0
    assert "disk_free_bytes" in stats
    assert "last_indexed_at" in stats  # may be None on empty


def test_health_echoes_compliance_mode_when_unset(app_with_store, monkeypatch):
    monkeypatch.delenv("KB_COMPLIANCE_MODE", raising=False)
    app, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/api/kb/health")
    data = resp.json()
    assert "compliance_mode" in data
    assert data["compliance_mode"] is None
    assert data["compliance_implemented"] is False


def test_health_echoes_compliance_mode_when_set(app_with_store, monkeypatch):
    monkeypatch.setenv("KB_COMPLIANCE_MODE", "ru_strict")
    app, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/api/kb/health")
    data = resp.json()
    assert data["compliance_mode"] == "ru_strict"
    assert data["compliance_implemented"] is False
