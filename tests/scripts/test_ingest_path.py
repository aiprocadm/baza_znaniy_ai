"""Tests for the ``scripts.ingest_path`` helper."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def ingest_module() -> types.ModuleType:
    module_name = "scripts.ingest_path"
    if module_name in sys.modules:
        del sys.modules[module_name]

    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in ("app.ingest", "app.ingest.service", "sqlmodel")
    }

    added_modules: list[str] = []

    # Provide lightweight stubs for the ingest package hierarchy to avoid importing heavy
    # optional dependencies during unit tests.  The real modules are restored afterwards.
    ingest_package = types.ModuleType("app.ingest")
    ingest_package.__path__ = []  # type: ignore[attr-defined]
    ingest_package.parse_and_chunk = lambda *args, **kwargs: []  # type: ignore[attr-defined]

    ingest_service = types.ModuleType("app.ingest.service")

    class _IngestService:  # pragma: no cover - behaviour not required for tests
        pass

    class _IngestWorker:  # pragma: no cover - behaviour not required for tests
        pass

    class _IngestJob:  # pragma: no cover - behaviour not required for tests
        pass

    ingest_service.IngestService = _IngestService  # type: ignore[attr-defined]
    ingest_service.IngestWorker = _IngestWorker  # type: ignore[attr-defined]
    ingest_service.IngestJob = _IngestJob  # type: ignore[attr-defined]

    for name, module in ("app.ingest", ingest_package), ("app.ingest.service", ingest_service):
        sys.modules[name] = module
        added_modules.append(name)

    try:  # pragma: no cover - real dependency available in CI environments
        importlib.import_module("sqlmodel")
    except ModuleNotFoundError:  # pragma: no cover - exercised in unit-test environments
        sqlmodel_stub = types.ModuleType("sqlmodel")

        class _Session:  # pragma: no cover - behaviour not needed for tests
            pass

        sqlmodel_stub.Session = _Session  # type: ignore[attr-defined]
        sys.modules["sqlmodel"] = sqlmodel_stub
        added_modules.append("sqlmodel")

    try:
        module = importlib.import_module(module_name)
    finally:
        for name in added_modules:
            sys.modules.pop(name, None)
        for name, original in original_modules.items():
            if original is not None:
                sys.modules[name] = original
    return module


class DummyVectorStore:
    """Simple in-memory implementation capturing method invocations."""

    def __init__(self) -> None:
        self.ensure_ready_calls = 0
        self.upsert_calls: list[list[Any]] = []

    def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1

    def upsert(self, chunks: list[Any]) -> None:
        # Store a shallow copy to ensure later mutations do not affect assertions.
        self.upsert_calls.append(list(chunks))


@pytest.fixture
def supported_extension(ingest_module: types.ModuleType) -> str:
    return sorted(ingest_module.SUPPORTED_EXTENSIONS)[0]


def test_iter_documents_filters_supported_extensions(
    ingest_module: types.ModuleType, tmp_path: Path, supported_extension: str
) -> None:
    supported_file = tmp_path / f"doc1{supported_extension}"
    supported_file.write_text("content")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    another_supported = nested_dir / f"doc2{supported_extension.upper()}"
    another_supported.write_text("other")

    unsupported_file = tmp_path / "ignore.bin"
    unsupported_file.write_text("nope")

    documents = list(ingest_module._iter_documents(tmp_path))

    assert documents == [supported_file, another_supported]


def test_iter_documents_accepts_single_file(
    ingest_module: types.ModuleType, tmp_path: Path, supported_extension: str
) -> None:
    document = tmp_path / f"single{supported_extension}"
    document.write_text("one")

    assert list(ingest_module._iter_documents(document)) == [document]


def test_ingest_path_indexes_only_supported_documents(
    ingest_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supported_extension: str,
) -> None:
    vector_store = DummyVectorStore()
    monkeypatch.setattr(ingest_module, "get_vector_store", lambda settings: vector_store)

    documents: dict[str, list[dict[str, str]]] = {
        f"alpha{supported_extension}": [{"id": "a1"}],
        f"beta{supported_extension}": [],
        f"gamma{supported_extension}": [{"id": "g1"}, {"id": "g2"}],
    }

    for name in documents:
        (tmp_path / name).write_text(f"payload for {name}")
    (tmp_path / "skipped.unsupported").write_text("skip me")

    parsed_documents: list[str] = []

    def fake_parse_and_chunk(name: str, payload: bytes) -> list[dict[str, str]]:
        parsed_documents.append(name)
        return documents[name]

    monkeypatch.setattr(ingest_module, "parse_and_chunk", fake_parse_and_chunk)

    total_chunks = ingest_module.ingest_path(tmp_path)

    assert vector_store.ensure_ready_calls == 1
    assert vector_store.upsert_calls == [
        documents[f"alpha{supported_extension}"],
        documents[f"gamma{supported_extension}"],
    ]
    assert total_chunks == 3
    assert sorted(parsed_documents) == sorted(documents.keys())
