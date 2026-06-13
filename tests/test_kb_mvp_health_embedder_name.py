"""The admin embedder-warning banner reads health().embedder.name === 'hash'.
Lock that contract so a future refactor can't silently drop the field."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def hash_embedder_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("KB_API_KEY", raising=False)  # keep /health open
    # Force hashing backend explicitly so the test is env-independent.
    # Without this, sentence-transformers (installed in prod/dev) would be
    # auto-detected as the default and the embedder.name would be "st".
    monkeypatch.setenv("KB_EMBEDDINGS_BACKEND", "hash")

    import app.services.kb_embeddings as kb_embeddings

    kb_embeddings.reset_embedder()  # drop any cached embedder from prior tests

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    yield TestClient(fastapi_app)

    # Clean up: reset embedder cache so other tests start fresh
    kb_embeddings.reset_embedder()


def test_health_exposes_embedder_name(hash_embedder_client):
    resp = hash_embedder_client.get("/api/kb/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "embedder" in body
    assert "name" in body["embedder"]
    # When KB_EMBEDDINGS_BACKEND=hash is set the hashing fallback is active.
    assert body["embedder"]["name"] == "hash"
