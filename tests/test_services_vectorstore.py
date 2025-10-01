"""Tests for the vectorstore service helpers."""

from __future__ import annotations

import sys
from typing import List

import pytest

from tests.test_scripts_export_import import (
    _install_config_stub,
    _install_retriever_stub,
)

# Ensure the vector store module can import without hitting production dependencies.
_install_config_stub()
_install_retriever_stub()


class _ImportVectorStore:
    def ensure_ready(self):  # pragma: no cover - import-time stub
        pass

    def upsert(self, items):  # pragma: no cover - import-time stub
        pass

    def search(self, query: str, *, top_k: int = 10):  # pragma: no cover - import-time stub
        return []


sys.modules["app.retriever"].get_vector_store = (  # type: ignore[attr-defined]
    lambda settings: _ImportVectorStore()
)

from app.services.vectorstore import clear_fallback, index_chunks, search


@pytest.fixture(autouse=True)
def reset_fallback():
    """Ensure the in-memory fallback is cleared between tests."""

    clear_fallback()
    yield
    clear_fallback()


class DummyVectorStore:
    """Simple dummy used to replace the real vector store in tests."""

    def __init__(self):
        self.ready_calls = 0
        self.upsert_calls: List[List[dict[str, object]]] = []
        self.search_calls: List[tuple[str, int]] = []
        self._results: List[dict[str, object]] = []

    def ensure_ready(self):
        self.ready_calls += 1

    def upsert(self, items):
        self.upsert_calls.append(list(items))

    def search(self, query: str, *, top_k: int):
        self.search_calls.append((query, top_k))
        return self._results[:top_k]


class FailingVectorStore:
    """Dummy vector store that raises whenever used."""

    def ensure_ready(self):
        raise RuntimeError("boom")

    def upsert(self, items):  # pragma: no cover - called only if ensure_ready passes
        raise RuntimeError("boom")

    def search(self, query: str, *, top_k: int):  # pragma: no cover - never reached
        raise AssertionError("search should not be called when falling back")


def test_index_chunks_and_search_with_vector_store(monkeypatch):
    dummy = DummyVectorStore()
    dummy._results = [{"id": "match"}]
    monkeypatch.setattr("app.services.vectorstore._VECTOR_STORE", dummy)

    chunks = [{"text": "chunk one"}, {"text": "chunk two"}]
    stored = index_chunks(chunks)

    assert stored == len(chunks)
    assert dummy.ready_calls == 1
    assert dummy.upsert_calls == [chunks]

    hits = search("anything", top_k=5)
    assert hits == dummy._results
    assert dummy.search_calls == [("anything", 5)]


def test_index_chunks_fallback_and_search_order(monkeypatch):
    monkeypatch.setattr("app.services.vectorstore._VECTOR_STORE", FailingVectorStore())

    chunks = [
        {"text": "foo foo foo"},
        {"text": "foo foo"},
        {"text": "foo"},
    ]

    indexed = index_chunks(chunks)
    assert indexed == len(chunks)

    # ensure the fallback was populated and search falls back with substring scoring
    hits = search("foo", top_k=3)
    assert [chunk["text"] for chunk in hits] == ["foo", "foo foo", "foo foo foo"]
