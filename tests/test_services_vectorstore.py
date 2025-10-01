"""Tests for the vector store service helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Provide lightweight stubs so ``app.services.vectorstore`` can be imported in
# isolation from the rest of the (intentionally broken) application stack used
# in the kata fixtures.
# ---------------------------------------------------------------------------


class _StubVectorStore:
    """Minimal vector store used while importing the module under test."""

    def ensure_ready(self) -> None:  # pragma: no cover - import-time stub
        pass

    def upsert(self, chunks) -> None:  # pragma: no cover - import-time stub
        pass

    def search(self, query, *, top_k: int):  # pragma: no cover - import-time stub
        return []


def _stub_get_vector_store(_settings) -> _StubVectorStore:
    return _StubVectorStore()


stub_config = types.ModuleType("app.core.config")
stub_config.get_settings = lambda: object()

stub_retriever = types.ModuleType("app.retriever")
stub_retriever.get_vector_store = _stub_get_vector_store

sys.modules["app.core.config"] = stub_config
sys.modules["app.retriever"] = stub_retriever
services_pkg = types.ModuleType("app.services")
services_pkg.__path__ = []  # pragma: no cover - package placeholder
sys.modules["app.services"] = services_pkg
setattr(sys.modules.setdefault("app", types.ModuleType("app")), "services", services_pkg)


MODULE_PATH = Path(__file__).resolve().parent.parent / "app/services/vectorstore.py"
spec = importlib.util.spec_from_file_location("app.services.vectorstore", MODULE_PATH)
vectorstore_module = importlib.util.module_from_spec(spec)
sys.modules["app.services.vectorstore"] = vectorstore_module
spec.loader.exec_module(vectorstore_module)
setattr(services_pkg, "vectorstore", vectorstore_module)


clear_fallback = vectorstore_module.clear_fallback
index_chunks = vectorstore_module.index_chunks
search = vectorstore_module.search


@dataclass
class DummyVectorStore:
    """Simple dummy vector store used for tests."""

    fail: bool = False
    ensure_ready_calls: int = 0
    upsert_calls: List[list[dict[str, object]]] = field(default_factory=list)
    search_calls: List[tuple[str, int]] = field(default_factory=list)

    def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1
        if self.fail:
            raise RuntimeError("fail")

    def upsert(self, chunks) -> None:
        if self.fail:
            raise RuntimeError("fail")
        self.upsert_calls.append(list(chunks))

    def search(self, query: str, *, top_k: int):
        self.search_calls.append((query, top_k))
        if self.fail:
            raise RuntimeError("fail")
        return [
            {"query": query, "top_k": top_k, "source": "dummy"},
        ]


@pytest.fixture(autouse=True)
def reset_fallback():
    """Ensure the in-memory fallback is cleared before and after each test."""

    clear_fallback()
    yield
    clear_fallback()


def test_index_chunks_success(monkeypatch):
    dummy = DummyVectorStore()
    monkeypatch.setattr("app.services.vectorstore._VECTOR_STORE", dummy)

    chunks = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
    ]

    stored = index_chunks(chunks)

    assert stored == len(chunks)
    assert dummy.ensure_ready_calls == 1
    assert dummy.upsert_calls == [chunks]

    results = search("anything", top_k=5)

    assert results == [{"query": "anything", "top_k": 5, "source": "dummy"}]
    assert dummy.search_calls == [("anything", 5)]


def test_index_chunks_fallback(monkeypatch):
    dummy = DummyVectorStore(fail=True)
    monkeypatch.setattr("app.services.vectorstore._VECTOR_STORE", dummy)

    chunks = [
        {"id": 1, "text": "Alpha beta alpha"},
        {"id": 2, "text": "Beta beta"},
        {"id": 3, "text": "Gamma"},
    ]

    stored = index_chunks(chunks)

    assert stored == len(chunks)
    assert dummy.ensure_ready_calls == 1

    results = search("beta", top_k=2)

    assert [chunk["id"] for chunk in results] == [2, 1]

    clear_fallback()

    assert search("beta", top_k=2) == []
