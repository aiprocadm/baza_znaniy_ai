"""kb_store.search() raises the right retrieval-degradation reasons."""

from __future__ import annotations

import pytest

import app.observability.retrieval_health as retrieval_health
import app.services.kb_store as kb_store
from app.services.kb_store import KnowledgeBaseStore


class _StubEmbedder:
    def __init__(self, dim: int, name: str = "stub") -> None:
        self.name = name
        self.dimension = dim
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.1] * self._dim


@pytest.fixture(autouse=True)
def _reset():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def _store(tmp_path, embedder):
    return KnowledgeBaseStore(tmp_path / "kb.sqlite3", embedder=embedder)


def test_hashing_embedder_is_reported(tmp_path):
    store = _store(tmp_path, _StubEmbedder(dim=8, name="hash"))
    store.add_document("Doc", "alpha beta gamma", source="text")

    store.search("beta", top_k=3)

    rep = retrieval_health.current_report()
    assert retrieval_health.RetrievalReason.HASHING_EMBEDDER in rep.reasons


def test_dim_mismatch_is_reported(tmp_path):
    store = _store(tmp_path, _StubEmbedder(dim=8, name="real"))
    store.add_document("Doc", "alpha beta gamma", source="text")
    store._embedder = _StubEmbedder(dim=16, name="real")  # swap without reindex

    hits = store.search("beta", top_k=3)

    assert hits == []
    rep = retrieval_health.current_report()
    assert retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH in rep.reasons


def test_truncation_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_store, "_search_hard_limit", lambda: 2)
    store = _store(tmp_path, _StubEmbedder(dim=8, name="real"))
    store.add_document("D1", "alpha", source="text")
    store.add_document("D2", "beta", source="text")
    store.add_document("D3", "gamma", source="text")

    store.search("alpha", top_k=2)

    rep = retrieval_health.current_report()
    assert retrieval_health.RetrievalReason.SEARCH_TRUNCATED in rep.reasons


def test_clean_search_is_not_degraded(tmp_path):
    store = _store(tmp_path, _StubEmbedder(dim=8, name="real"))
    store.add_document("Doc", "alpha beta gamma", source="text")

    store.search("beta", top_k=3)

    rep = retrieval_health.current_report()
    assert rep is not None
    assert rep.source == "sqlite"
    assert rep.degraded is False
