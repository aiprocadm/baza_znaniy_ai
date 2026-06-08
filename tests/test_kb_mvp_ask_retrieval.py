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
from app.services.kb_embeddings import HashingEmbedder
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
    # Explicit hashing embedder -> CRITICAL (ST is now the implicit default)
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=HashingEmbedder())
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
    store = KnowledgeBaseStore(
        tmp_path / "kb.sqlite", embedder=HashingEmbedder()
    )  # explicit hashing to test degradation path
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


# ---------------------------------------------------------------------------
# GgufEvalProvider.generate_stream — streaming path tests
# ---------------------------------------------------------------------------


class _FakeInner:
    """Stand-in for LlamaCppProvider that returns a fixed string."""

    def __init__(self, text: str = "Hello from GGUF") -> None:
        self._text = text

    def generate(self, prompt: str, *, context=None) -> str:
        return self._text


def _gguf_client(tmp_path: Path, monkeypatch, response_text: str = "Hello from GGUF") -> TestClient:
    """Build a TestClient whose select_provider returns a GgufEvalProvider backed by a fake inner."""
    from app.services.kb_llm import GgufEvalProvider

    inner = _FakeInner(response_text)
    # model_path only needs to pass the is_available() check if called; we bypass
    # select_provider entirely via monkeypatch so any string is fine.
    provider = GgufEvalProvider(model_path="fake/model.gguf", inner=inner)

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=_StubEmbedder())
    store.add_document("doc1", text="alpha beta gamma " * 20)

    app_inst = FastAPI()
    app_inst.include_router(router, prefix="/api/kb")
    app_inst.state.kb_mvp_store = store
    monkeypatch.setenv("KB_LLM_LOCAL_FALLBACK", "0")
    monkeypatch.setattr(kb_llm, "select_provider", lambda: provider)
    return TestClient(app_inst)


def test_ask_stream_gguf_provider_yields_text(tmp_path: Path, monkeypatch):
    """GgufEvalProvider.generate_stream must deliver text over /ask/stream."""
    client = _gguf_client(tmp_path, monkeypatch, response_text="Hello from GGUF")

    with client.stream("POST", "/api/kb/ask/stream", json={"question": "alpha"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk.decode("utf-8") for chunk in resp.iter_bytes())

    # There should be at least one token event carrying the generated text.
    assert "Hello from GGUF" in body
    assert "event: token" in body


def test_ask_stream_provider_without_generate_stream_degrades_gracefully(
    tmp_path: Path, monkeypatch
):
    """A provider that exposes only generate() (no generate_stream) must NOT crash /ask/stream.

    The guard in kb_mvp.py checks for generate_stream via getattr before calling
    it; if missing, the request falls through to the extractive path and returns
    status 200 with a valid SSE response.
    """

    class _SyncOnlyProvider:
        name = "sync_only"
        model = "fake-model"

        def generate(self, prompt: str, *, system=None, **kwargs) -> str:
            return "sync only text"

    # No generate_stream attribute on this provider.
    assert not hasattr(_SyncOnlyProvider(), "generate_stream")

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=_StubEmbedder())
    store.add_document("doc1", text="alpha beta gamma " * 20)

    app_inst = FastAPI()
    app_inst.include_router(router, prefix="/api/kb")
    app_inst.state.kb_mvp_store = store
    monkeypatch.setenv("KB_LLM_LOCAL_FALLBACK", "0")
    monkeypatch.setattr(kb_llm, "select_provider", lambda: _SyncOnlyProvider())

    client = TestClient(app_inst)
    with client.stream("POST", "/api/kb/ask/stream", json={"question": "alpha"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk.decode("utf-8") for chunk in resp.iter_bytes())

    # Should have at least a meta event and a done event (extractive fallback).
    assert "event: meta" in body
    assert "event: done" in body
    # Must not contain an unhandled-error SSE that signals a server crash.
    assert "internal streaming error" not in body
