"""Lock the health() contract that the admin degradation banner depends on.

Two fields are locked:
- ``health().embedder.name == "hash"`` — used by the legacy JS embedder-warning
  banner to detect the hashing backend.
- ``health().retrieval.degraded == True`` and a reason entry with
  ``reason == "hashing_embedder"`` — used by the new retrieval-degradation
  banner introduced in A3 so the admin console can surface the issue
  independently of per-query context.
"""
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


def test_health_retrieval_degraded_under_hash(hash_embedder_client):
    """Lock the retrieval contract the admin degradation banner depends on.

    When the hashing embedder is active the health response MUST include:
    - ``retrieval.degraded == True``
    - at least one entry in ``retrieval.reasons`` with ``reason == "hashing_embedder"``

    If either assertion fails a future backend refactor has broken the contract
    that the admin banner JS reads from ``GET /api/kb/health``.
    """
    resp = hash_embedder_client.get("/api/kb/health")
    assert resp.status_code == 200
    body = resp.json()

    assert "retrieval" in body, "health() must expose a 'retrieval' block"
    retrieval = body["retrieval"]

    assert retrieval.get("degraded") is True, (
        "retrieval.degraded must be True when the hashing embedder is active"
    )

    reason_keys = [r.get("reason") for r in retrieval.get("reasons", [])]
    assert "hashing_embedder" in reason_keys, (
        f"Expected 'hashing_embedder' in retrieval.reasons, got: {reason_keys}"
    )
