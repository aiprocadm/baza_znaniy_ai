"""Tests for the vector store service helpers."""

from __future__ import annotations

from typing import Iterable, List

import pytest

import app.services.vectorstore as vectorstore


class DummyVectorStore:
    """A minimal stand-in for the real vector store implementation."""

    def __init__(self) -> None:
        self.ready_calls = 0
        self.upserted: List[List[dict[str, object]]] = []
        self.search_calls: List[tuple[str, int]] = []
        self.results: List[dict[str, object]] = []

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def upsert(self, items: Iterable[dict[str, object]]) -> None:
        self.upserted.append(list(items))

    def search(self, query: str, *, top_k: int) -> List[dict[str, object]]:
        self.search_calls.append((query, top_k))
        return self.results[:top_k]


class ExplodingVectorStore:
    """Vector store stub that always raises to trigger the fallback."""

    def __init__(self) -> None:
        self.ready_calls = 0

    def ensure_ready(self) -> None:
        self.ready_calls += 1
        raise RuntimeError("boom")

    def upsert(self, items: Iterable[dict[str, object]]) -> None:  # pragma: no cover - not called
        raise RuntimeError("boom")

    def search(self, query: str, *, top_k: int) -> List[dict[str, object]]:  # pragma: no cover - not called
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def clear_fallback_between_tests() -> None:
    """Ensure a clean fallback index for each test."""

    vectorstore.clear_fallback()
    yield
    vectorstore.clear_fallback()


def test_index_chunks_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyVectorStore()
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", dummy)

    chunks = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]

    stored = vectorstore.index_chunks(chunks)

    assert stored == len(chunks)
    assert dummy.ready_calls == 1
    assert dummy.upserted == [chunks]

    dummy.results = [{"text": "hit"}]
    hits = vectorstore.search("anything", top_k=5)

    assert hits == dummy.results
    assert dummy.search_calls == [("anything", 5)]


def test_index_chunks_fallback_and_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_store = ExplodingVectorStore()
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", failing_store)

    chunks = [
        {"id": 1, "text": "Alpha beta alpha"},
        {"id": 2, "text": "Beta beta"},
        {"id": 3, "text": "Gamma"},
    ]

    stored = vectorstore.index_chunks(chunks)

    assert stored == len(chunks)
    assert failing_store.ready_calls == 1

    results = vectorstore.search("beta", top_k=2)
    assert [chunk["id"] for chunk in results] == [2, 1]

    vectorstore.clear_fallback()
    assert vectorstore.search("beta", top_k=2) == []
