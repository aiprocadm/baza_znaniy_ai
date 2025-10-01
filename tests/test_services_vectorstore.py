        codex/add-tests-for-vectorstore-module-mzy2bq

        codex/add-tests-for-vectorstore-module
        main
"""Tests for the vector store service helpers."""

from __future__ import annotations

        codex/add-tests-for-vectorstore-module-mzy2bq
import sys
from types import ModuleType
from typing import Iterable, List

import pytest

import app
import app.core  # type: ignore  # ensure package is loaded for attribute injection


class _BootstrapVectorStore:
    """Minimal vector store used to satisfy imports during module loading."""

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

"""Tests for the vectorstore service helpers."""

from __future__ import annotations

import sys
        main
from typing import List

import pytest

        codex/add-tests-for-vectorstore-module

# ---------------------------------------------------------------------------
# Provide lightweight stubs so ``app.services.vectorstore`` can be imported in
# isolation from the rest of the (intentionally broken) application stack used
# in the kata fixtures.
# ---------------------------------------------------------------------------


class _StubVectorStore:
    """Minimal vector store used while importing the module under test."""
        main

    def ensure_ready(self) -> None:  # pragma: no cover - import-time stub
        pass

        codex/add-tests-for-vectorstore-module-mzy2bq
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
        main

    clear_fallback()
    yield
    clear_fallback()


        codex/add-tests-for-vectorstore-module-mzy2bq
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
        main
    ]

    indexed = index_chunks(chunks)
    assert indexed == len(chunks)
        codex/add-tests-for-vectorstore-module-mzy2bq
    assert failing_store.ready_calls == 1

    results = search("beta", top_k=10)
    assert results == [chunks[1], chunks[0]]

    clear_fallback()
    assert search("beta", top_k=10) == []


    # ensure the fallback was populated and search falls back with substring scoring
    hits = search("foo", top_k=3)
    assert [chunk["text"] for chunk in hits] == ["foo", "foo foo", "foo foo foo"]
        main
        main
