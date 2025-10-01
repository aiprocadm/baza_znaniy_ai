"""Tests for the lightweight qdrant client compatibility layer."""

from __future__ import annotations

import importlib
import sys
from collections import deque
from types import SimpleNamespace
from typing import Iterable

import pytest


class VectorStoreStub:
    """A simple stub with hooks for various vector store operations."""

    def __init__(self, *, search_result=None):
        self.ensure_ready_called = False
        self.upsert_received = None
        self.search_calls = []
        self.search_result = search_result if search_result is not None else []

    def ensure_ready(self) -> None:  # pragma: no cover - trivial
        self.ensure_ready_called = True

    def upsert(self, chunks: Iterable[dict[str, object]]):  # pragma: no cover - trivial
        self.upsert_received = list(chunks)

    def search(self, query: str, top_k: int):  # pragma: no cover - trivial
        self.search_calls.append((query, top_k))
        return self.search_result


class ResettableVectorStoreStub(VectorStoreStub):
    def __init__(self):
        super().__init__()
        self.reset_called = False

    def reset_collection(self):  # pragma: no cover - trivial
        self.reset_called = True


class ExportImportVectorStoreStub(VectorStoreStub):
    def __init__(self, exported_batches):
        super().__init__()
        self.exported_batches = deque(exported_batches)
        self.export_called_with = []
        self.imported_payloads = []

    def export_payloads(self, *, batch_size):
        self.export_called_with.append(batch_size)
        while self.exported_batches:
            yield from self.exported_batches.popleft()

    def import_payloads(self, payloads):
        self.imported_payloads.extend(payloads)


@pytest.fixture
def qc_module(monkeypatch):
    """Import ``app.qdrant_client`` with light-weight dependency stubs."""

    settings = SimpleNamespace(
        qdrant_url="http://stub",  # pragma: no cover - attribute access only
        qdrant_api_key="key",
        qdrant_collection="collection",
        qdrant_path_resolved="/tmp/qdrant",
        vector_embed_model="model",
        vector_embed_dimension=7,
    )

    config_stub = SimpleNamespace(get_settings=lambda: settings)
    retriever_stub = SimpleNamespace(get_vector_store=lambda _settings: None)

    monkeypatch.setitem(sys.modules, "app.core.config", config_stub)
    monkeypatch.setitem(sys.modules, "app.retriever", retriever_stub)

    qc = importlib.import_module("app.qdrant_client")
    qc = importlib.reload(qc)
    return qc


def test_basic_operations_use_vector_store(qc_module, monkeypatch):
    """ensure_collection, upsert_chunks and search_chunks proxy to the vector store."""

    expected_search = [{"id": "chunk-1"}]
    stub = VectorStoreStub(search_result=expected_search)
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    qc_module.ensure_collection()
    assert stub.ensure_ready_called is True

    chunks = [{"id": "chunk-42"}]
    qc_module.upsert_chunks(chunks)
    assert stub.upsert_received == chunks

    result = qc_module.search_chunks("hello", top_k=3)
    assert result == expected_search
    assert stub.search_calls == [("hello", 3)]


def test_reset_collection_supported(qc_module, monkeypatch):
    stub = ResettableVectorStoreStub()
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    qc_module.reset_collection()
    assert stub.reset_called is True


def test_reset_collection_not_supported(qc_module, monkeypatch):
    stub = VectorStoreStub()
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    with pytest.raises(NotImplementedError):
        qc_module.reset_collection()


def test_export_payloads_supported(qc_module, monkeypatch):
    batches = [[{"id": 1}], [{"id": 2}]]
    stub = ExportImportVectorStoreStub(exported_batches=batches)
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    iterator = qc_module.export_payloads(batch_size=5)
    exported = list(iterator)
    assert exported == [{"id": 1}, {"id": 2}]
    assert stub.export_called_with == [5]


def test_export_payloads_not_supported(qc_module, monkeypatch):
    stub = VectorStoreStub()
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    with pytest.raises(NotImplementedError):
        qc_module.export_payloads()


def test_import_payloads_supported(qc_module, monkeypatch):
    stub = ExportImportVectorStoreStub(exported_batches=[])
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    payloads = [{"id": "one"}, {"id": "two"}]
    qc_module.import_payloads(payloads)
    assert stub.imported_payloads == payloads


def test_import_payloads_not_supported(qc_module, monkeypatch):
    stub = VectorStoreStub()
    monkeypatch.setattr(qc_module, "_vector_store", stub)

    with pytest.raises(NotImplementedError):
        qc_module.import_payloads([])


def test_export_import_roundtrip(qc_module, monkeypatch):
    """End-to-end check that export/import iterate and restore module state."""

    original_vector_store = qc_module._vector_store
    batches = [[{"id": "a"}, {"id": "b"}], [{"id": "c"}]]
    stub = ExportImportVectorStoreStub(exported_batches=batches)

    monkeypatch.setattr(qc_module, "_vector_store", stub)

    exported = list(qc_module.export_payloads(batch_size=2))
    assert exported == [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    qc_module.import_payloads(exported)
    assert stub.imported_payloads == exported

    # ensure monkeypatch will restore the original vector store when the test ends
    monkeypatch.setattr(qc_module, "_vector_store", original_vector_store)
