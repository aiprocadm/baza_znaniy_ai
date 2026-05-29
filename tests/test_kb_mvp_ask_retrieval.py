"""/api/kb/ask and /ask/stream carry the per-query retrieval degradation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.observability.retrieval_health as retrieval_health
from app.api.kb_mvp import router
from app.services import kb_llm
from app.services.kb_store import KnowledgeBaseStore


class _StubEmbedder:
    """A non-hashing embedder (name != 'hash') with a fixed dimension."""

    def __init__(self, dim: int = 8, name: str = "real") -> None:
        self.name = name
        self.dimension = dim
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.1] * self._dim


@pytest.fixture(autouse=True)
def _reset_retrieval_health():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def _client(store: KnowledgeBaseStore, monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/kb")
    app.state.kb_mvp_store = store
    monkeypatch.setattr(kb_llm, "select_provider", lambda: None)  # force extractive
    return TestClient(app)


def test_ask_reports_hashing_embedder_as_critical(tmp_path: Path, monkeypatch):
    # Default store -> hashing embedder (no KB_EMBEDDINGS_BACKEND) -> CRITICAL
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    data = client.post("/api/kb/ask", json={"question": "alpha"}).json()

    assert data["retrieval"] is not None
    assert data["retrieval"]["degraded"] is True
    assert data["retrieval"]["severity"] == "critical"
    reasons = [r["reason"] for r in data["retrieval"]["reasons"]]
    assert "hashing_embedder" in reasons


def test_ask_omits_retrieval_when_clean(tmp_path: Path, monkeypatch):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=_StubEmbedder())
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    data = client.post("/api/kb/ask", json={"question": "alpha"}).json()

    assert data["retrieval"] is None


def _read_meta_event(client: TestClient, question: str) -> dict:
    with client.stream("POST", "/api/kb/ask/stream", json={"question": question}) as resp:
        assert resp.status_code == 200
        text = "".join(chunk.decode("utf-8") for chunk in resp.iter_bytes())
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "event: meta" and i + 1 < len(lines):
            data_line = lines[i + 1]
            if data_line.startswith("data: "):
                return json.loads(data_line[len("data: ") :])
    raise AssertionError(f"meta event not found:\n{text}")


def test_ask_stream_meta_carries_retrieval_when_degraded(tmp_path: Path, monkeypatch):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")  # hashing default
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    meta = _read_meta_event(client, "alpha")

    assert meta["retrieval"] is not None
    assert meta["retrieval"]["severity"] == "critical"
    assert any(r["reason"] == "hashing_embedder" for r in meta["retrieval"]["reasons"])


def test_ask_stream_meta_retrieval_none_when_clean(tmp_path: Path, monkeypatch):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=_StubEmbedder())
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    meta = _read_meta_event(client, "alpha")

    assert meta["retrieval"] is None
