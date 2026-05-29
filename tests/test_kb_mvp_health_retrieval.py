"""/api/kb/health surfaces retrieval degradation without breaking liveness."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.observability.retrieval_health as retrieval_health
import app.services.kb_embeddings as kb_embeddings
from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_with_hashing_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("KB_EMBEDDINGS_BACKEND", raising=False)
    kb_embeddings.reset_embedder()  # force re-resolution to the hashing default
    retrieval_health.reset()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    store.add_document("doc1", text="alpha " * 50)
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app


def test_health_reports_hashing_embedder_as_critical(app_with_hashing_store):
    client = TestClient(app_with_hashing_store)

    data = client.get("/api/kb/health").json()

    assert data["status"] == "ok"  # liveness probes must keep working
    assert data["degraded"] is True
    assert data["retrieval"]["severity"] == "critical"
    reasons = [r["reason"] for r in data["retrieval"]["reasons"]]
    assert "hashing_embedder" in reasons
