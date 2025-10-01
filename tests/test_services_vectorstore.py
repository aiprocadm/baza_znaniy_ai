"""Tests for the vector store service helpers."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Iterable, List

import pytest

import app
import app.core  # type: ignore  # ensure package is loaded for attribute injection


class _BootstrapVectorStore:
    """Minimal vector store used to satisfy imports during module loading."""

    def ensure_ready(self) -> None:  # pragma: no cover - import-time stub
        pass

    def upsert(self, items: Iterable[dict[str, object]]) -> None:  # pragma: no cover - import-time stub
        pass

    def search(self, query: str, *, top_k: int) -> List[dict[str, object]]:  # pragma: no cover - import-time stub
        return []


if "app.core.config" not in sys.modules:
    bootstrap_config = ModuleType("app.core.config")

    class _BootstrapSettings:  # pragma: no cover - import-time stub
        pass

    def _get_settings() -> _BootstrapSettings:  # pragma: no cover - import-time stub
        return _BootstrapSettings()

    bootstrap_config.get_settings = _get_settings  # type: ignore[attr-defined]
    sys.modules["app.core.config"] = bootstrap_config
    setattr(sys.modules["app.core"], "config", bootstrap_config)


if "app.retriever" not in sys.modules:
    bootstrap_module = ModuleType("app.retriever")
    _bootstrap_vector_store = _BootstrapVectorStore()

    def _get_vector_store(_settings: object) -> _BootstrapVectorStore:  # pragma: no cover - import-time stub
        return _bootstrap_vector_store

    bootstrap_module.get_vector_store = _get_vector_store  # type: ignore[attr-defined]
    sys.modules["app.retriever"] = bootstrap_module
    setattr(app, "retriever", bootstrap_module)


from app.services.vectorstore import clear_fallback, index_chunks, search
import app.services.vectorstore as vectorstore_module


class DummyVectorStore:
    """A minimal stand-in for the real vector store implementation."""

    def __init__(self) -> None:
        self.ready_calls = 0
        self.upserted: List[dict[str, object]] | None = None
        self.search_calls: list[tuple[str, int]] = []
        self.results: List[dict[str, object]] = [{"text": "dummy"}]

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def upsert(self, items: Iterable[dict[str, object]]) -> None:
        self.upserted = list(items)

    def search(self, query: str, *, top_k: int) -> List[dict[str, object]]:
        self.search_calls.append((query, top_k))
        return self.results


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
def _clear_fallback_between_tests() -> None:
    """Ensure a clean fallback index for each test."""

    clear_fallback()
    yield
    clear_fallback()


def test_index_chunks_success(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_store = DummyVectorStore()
    monkeypatch.setattr(vectorstore_module, "_VECTOR_STORE", dummy_store)

    chunks = [{"text": "first"}, {"text": "second"}]

    indexed = index_chunks(chunks)

    assert indexed == len(chunks)
    assert dummy_store.ready_calls == 1
    assert dummy_store.upserted == chunks

    dummy_store.results = [{"text": "hit"}]
    hits = search("irrelevant", top_k=5)

    assert hits == dummy_store.results
    assert dummy_store.search_calls == [("irrelevant", 5)]


def test_index_chunks_fallback_and_search_order(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_store = ExplodingVectorStore()
    monkeypatch.setattr(vectorstore_module, "_VECTOR_STORE", failing_store)

    chunks = [
        {"text": "Alpha Beta"},
        {"text": "Beta Beta"},
    ]

    indexed = index_chunks(chunks)
    assert indexed == len(chunks)
    assert failing_store.ready_calls == 1

    results = search("beta", top_k=10)
    assert results == [chunks[1], chunks[0]]

    clear_fallback()
    assert search("beta", top_k=10) == []
