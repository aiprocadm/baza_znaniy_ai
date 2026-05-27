"""Tests for the vector store service helpers."""

from __future__ import annotations

from typing import Iterable, List

import pytest

import app.services.vectorstore as vectorstore
from app.retriever.vector_store import SearchFilters


class DummyVectorStore:
    """A minimal stand-in for the real vector store implementation.

    Mirrors :class:`app.retriever.vector_store.VectorStore` exactly:
    ``search(query, top_k, *, filters: SearchFilters)``. Filter content
    is captured into ``search_calls`` so tests can assert what the
    production ``vectorstore.search`` actually forwarded.
    """

    def __init__(self) -> None:
        self.ready_calls = 0
        self.upserted: List[List[dict[str, object]]] = []
        self.search_calls: List[tuple[str, int, SearchFilters]] = []
        self.results: List[dict[str, object]] = []

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def upsert(self, items: Iterable[dict[str, object]]) -> None:
        self.upserted.append(list(items))

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: SearchFilters,
    ) -> List[dict[str, object]]:
        self.search_calls.append((query, top_k, filters))
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

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: SearchFilters,
    ) -> List[dict[str, object]]:  # pragma: no cover - not called
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def clear_fallback_between_tests() -> None:
    """Ensure a clean fallback index for each test."""

    storage: list[dict[str, object]] = []
    vectorstore.set_fallback_storage(storage)
    vectorstore.clear_fallback()
    try:
        yield
    finally:
        vectorstore.set_fallback_storage([])
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
    hits = vectorstore.search("anything", top_k=5, tenant_id="t1")

    assert hits == dummy.results
    assert len(dummy.search_calls) == 1
    query, top_k, filters = dummy.search_calls[0]
    assert query == "anything"
    assert top_k == 5
    assert filters.tenant_id == "t1"
    assert filters.owner is None
    assert filters.tags == ()


def test_index_chunks_fallback_and_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_store = ExplodingVectorStore()
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", failing_store)

    # tenant_id is required end-to-end now; the fallback filter rejects
    # chunks that don't match the tenant. Tag every chunk with "t1" so
    # the ordering assertion below is not silently empty.
    chunks = [
        {"id": 1, "text": "Alpha beta alpha", "tenant_id": "t1"},
        {"id": 2, "text": "Beta beta", "tenant_id": "t1"},
        {"id": 3, "text": "Gamma", "tenant_id": "t1"},
    ]

    stored = vectorstore.index_chunks(chunks)

    assert stored == len(chunks)
    assert failing_store.ready_calls == 1

    results = vectorstore.search("beta", top_k=2, tenant_id="t1")
    assert [chunk["id"] for chunk in results] == [2, 1]

    vectorstore.clear_fallback()
    assert vectorstore.search("beta", top_k=2, tenant_id="t1") == []


def test_configurable_fallback_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_store = ExplodingVectorStore()
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", failing_store)

    shared_storage: list[dict[str, object]] = []
    vectorstore.set_fallback_storage(shared_storage)

    chunks = [{"id": 11, "text": "shared"}]
    stored = vectorstore.index_chunks(chunks)

    assert stored == 1
    assert shared_storage == chunks
    assert vectorstore.get_fallback_storage() is shared_storage


def test_fallback_search_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the in-memory fallback path explicitly. The original test
    # relied on the primary backend being unreachable in the test env;
    # injecting ExplodingVectorStore makes the fallback intent explicit
    # and survives changes to the default backend resolution.
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", ExplodingVectorStore())

    # All chunks live in tenant "t1" so the test exercises owner/tag
    # filtering in isolation, not the tenant gate.
    vectorstore.index_chunks(
        [
            {
                "id": 1,
                "text": "Replication setup",
                "tenant_id": "t1",
                "owner": "alice@kb.ai",
                "tags": ["prod", "runbook"],
            },
            {
                "id": 2,
                "text": "Replication setup",
                "tenant_id": "t1",
                "owner": "bob@kb.ai",
                "tags": ["dev"],
            },
            {
                "id": 3,
                "text": "Replication setup",
                "tenant_id": "t1",
                "owner": "alice@kb.ai",
                "tags": ["prod"],
            },
        ]
    )

    owner_hits = vectorstore.search("replication", top_k=10, tenant_id="t1", owner="alice@kb.ai")
    assert [item["id"] for item in owner_hits] == [1, 3]

    tag_hits = vectorstore.search("replication", top_k=10, tenant_id="t1", tags=["prod", "runbook"])
    assert [item["id"] for item in tag_hits] == [1]
