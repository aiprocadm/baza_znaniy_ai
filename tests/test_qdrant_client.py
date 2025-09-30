"""Tests for the Qdrant helper utilities."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Iterable
from unittest.mock import MagicMock

import numpy as np
import pytest


def _install_qdrant_stubs() -> None:
    """Install stub modules for qdrant_client so tests can run without dependency."""

    if "qdrant_client" in sys.modules:
        return

    qdrant_module = ModuleType("qdrant_client")

    class _StubQdrantClient:  # pragma: no cover - simple stand-in
        def __init__(self, *_: object, **__: object) -> None:  # noqa: D401
            pass

    qdrant_module.QdrantClient = _StubQdrantClient

    http_module = ModuleType("qdrant_client.http")
    models_module = ModuleType("qdrant_client.http.models")
    exceptions_module = ModuleType("qdrant_client.http.exceptions")

    class VectorParams:
        def __init__(self, size: int, distance: object) -> None:
            self.size = size
            self.distance = distance

    class HnswConfigDiff:
        def __init__(self, **kwargs: object) -> None:
            self.params = kwargs

    class SearchParams:
        def __init__(self, **kwargs: object) -> None:
            self.params = kwargs

    class Distance:
        COSINE = "cosine"

    class PayloadSchemaType:
        KEYWORD = "keyword"
        INTEGER = "integer"

    class PointStruct:
        def __init__(self, id: str, vector: list[float], payload: dict[str, object]) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    models_module.VectorParams = VectorParams
    models_module.HnswConfigDiff = HnswConfigDiff
    models_module.SearchParams = SearchParams
    models_module.Distance = Distance
    models_module.PayloadSchemaType = PayloadSchemaType
    models_module.PointStruct = PointStruct

    class UnexpectedResponse(Exception):
        pass

    exceptions_module.UnexpectedResponse = UnexpectedResponse

    qdrant_module.http = http_module

    sys.modules["qdrant_client"] = qdrant_module
    sys.modules["qdrant_client.http"] = http_module
    sys.modules["qdrant_client.http.models"] = models_module
    sys.modules["qdrant_client.http.exceptions"] = exceptions_module


_install_qdrant_stubs()

from app import qdrant_client as qc
from qdrant_client.http import models as qmodels


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    """Ensure module level singletons are reset between tests."""

    monkeypatch.setattr(qc, "_model", None)
    monkeypatch.setattr(qc, "_qdrant", None)
    yield
    monkeypatch.setattr(qc, "_model", None)
    monkeypatch.setattr(qc, "_qdrant", None)


@pytest.fixture
def mocked_qdrant(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a mocked Qdrant client and patch the constructor."""

    client = MagicMock(spec=[
        "get_collection",
        "recreate_collection",
        "create_payload_index",
        "upsert",
        "search",
    ])

    def factory(**_: object) -> qc.QdrantClient:
        return client

    monkeypatch.setattr(qc, "QdrantClient", factory)
    return client


def _collection_info(size: int) -> object:
    """Build a minimal collection information structure."""

    vectors = SimpleNamespace(size=size)
    params = SimpleNamespace(vectors=vectors)
    config = SimpleNamespace(params=params)
    return SimpleNamespace(config=config)


def _multi_collection_info(*sizes: int) -> object:
    vectors = {f"vec{i}": SimpleNamespace(size=size) for i, size in enumerate(sizes)}
    params = SimpleNamespace(vectors=vectors)
    config = SimpleNamespace(params=params)
    return SimpleNamespace(config=config)


def test_embedder_caches_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[object] = []

    class DummySentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            instances.append(self)

        def get_sentence_embedding_dimension(self) -> int:
            return qc.EMBED_DIMENSION

    monkeypatch.setattr(qc, "SentenceTransformer", DummySentenceTransformer)

    first = qc._embedder()
    second = qc._embedder()

    assert first is second
    assert instances == [first]
    assert first.model_name == qc.EMBED_MODEL


def test_embedder_raises_for_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySentenceTransformer:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return qc.EMBED_DIMENSION + 1

    monkeypatch.setattr(qc, "SentenceTransformer", DummySentenceTransformer)

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        qc._embedder()


def test_qdrant_client_forwards_url_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[dict[str, object]] = []
    sentinel = object()

    def factory(**kwargs: object) -> object:
        created_clients.append(kwargs)
        return sentinel

    monkeypatch.setattr(qc, "QdrantClient", factory)
    monkeypatch.setattr(qc, "QDRANT_URL", "http://custom")
    monkeypatch.setattr(qc, "QDRANT_API_KEY", "secret")

    client1 = qc._qdrant_client()
    client2 = qc._qdrant_client()

    assert client1 is sentinel
    assert client2 is sentinel
    assert created_clients == [{"url": "http://custom", "api_key": "secret"}]


def test_qdrant_client_skips_api_key_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[dict[str, object]] = []
    sentinel = object()

    def factory(**kwargs: object) -> object:
        created_clients.append(kwargs)
        return sentinel

    monkeypatch.setattr(qc, "QdrantClient", factory)
    monkeypatch.setattr(qc, "QDRANT_URL", "http://example")
    monkeypatch.setattr(qc, "QDRANT_API_KEY", "")

    client1 = qc._qdrant_client()
    client2 = qc._qdrant_client()

    assert client1 is sentinel
    assert client2 is sentinel
    assert created_clients == [{"url": "http://example"}]


def test_ensure_collection_recreates_when_missing(mocked_qdrant: MagicMock) -> None:
    mocked_qdrant.get_collection.return_value = None

    qc.ensure_collection()

    mocked_qdrant.recreate_collection.assert_called_once()
    payload_calls = mocked_qdrant.create_payload_index.call_args_list
    assert {c.kwargs["field_name"] for c in payload_calls} == {"file", "page", "sha256"}


def test_ensure_collection_skips_recreate_for_matching_size(mocked_qdrant: MagicMock) -> None:
    mocked_qdrant.get_collection.return_value = _collection_info(qc.EMBED_DIMENSION)

    qc.ensure_collection()

    mocked_qdrant.recreate_collection.assert_not_called()
    mocked_qdrant.create_payload_index.assert_not_called()


def test_ensure_collection_recreates_for_mismatched_size(mocked_qdrant: MagicMock) -> None:
    mocked_qdrant.get_collection.return_value = _collection_info(qc.EMBED_DIMENSION + 1)

    qc.ensure_collection()

    mocked_qdrant.recreate_collection.assert_called_once()


def test_ensure_collection_recreates_for_partial_multi_vector(mocked_qdrant: MagicMock) -> None:
    mocked_qdrant.get_collection.return_value = _multi_collection_info(qc.EMBED_DIMENSION, qc.EMBED_DIMENSION + 2)

    qc.ensure_collection()

    mocked_qdrant.recreate_collection.assert_called_once()


def test_upsert_chunks_deduplicates_by_sha256(monkeypatch: pytest.MonkeyPatch, mocked_qdrant: MagicMock) -> None:
    monkeypatch.setattr(qc, "ensure_collection", MagicMock())

    embeddings = np.stack([
        np.full(qc.EMBED_DIMENSION, 1.0, dtype=np.float32),
        np.full(qc.EMBED_DIMENSION, 2.0, dtype=np.float32),
    ])
    encode_mock = MagicMock(return_value=embeddings)
    monkeypatch.setattr(qc, "_encode_texts", encode_mock)

    chunks = [
        {"sha256": "a", "text": "first", "file": "doc1", "page": 1},
        {"sha256": "b", "text": "second", "file": "doc2", "page": 2},
        {"sha256": "a", "text": "updated", "file": "doc1", "page": 3},
    ]

    qc.upsert_chunks(chunks)

    encode_mock.assert_called_once_with(["updated", "second"])
    mocked_qdrant.upsert.assert_called_once()
    points = mocked_qdrant.upsert.call_args.kwargs["points"]
    assert {point.id for point in points} == {"a", "b"}
    for point in points:
        if point.id == "a":
            assert point.payload["text"] == "updated"
            assert point.payload["page"] == 3


def test_upsert_chunks_raises_without_sha(monkeypatch: pytest.MonkeyPatch, mocked_qdrant: MagicMock) -> None:
    monkeypatch.setattr(qc, "ensure_collection", MagicMock())
    encode_mock = MagicMock()
    monkeypatch.setattr(qc, "_encode_texts", encode_mock)

    with pytest.raises(ValueError, match="sha256"):
        qc.upsert_chunks([{"text": "missing"}])

    encode_mock.assert_not_called()
    mocked_qdrant.upsert.assert_not_called()


def test_search_chunks_returns_payload_with_float_scores(monkeypatch: pytest.MonkeyPatch, mocked_qdrant: MagicMock) -> None:
    ensure_mock = MagicMock()
    monkeypatch.setattr(qc, "ensure_collection", ensure_mock)
    query_vector = np.full((1, qc.EMBED_DIMENSION), 1 / np.sqrt(qc.EMBED_DIMENSION), dtype=np.float32)
    monkeypatch.setattr(qc, "_encode_texts", MagicMock(return_value=query_vector))

    point1 = qmodels.PointStruct(id="1", vector=[0.0], payload={
        "file": "doc1",
        "page": 1,
        "sha256": "sha1",
        "text": "first",
    })
    point1.score = np.float32(0.5)

    point2 = qmodels.PointStruct(id="2", vector=[0.0], payload={
        "file": "doc2",
        "page": 2,
        "sha256": "sha2",
        "text": "second",
    })
    point2.score = np.float32(0.4)

    point3 = qmodels.PointStruct(id="3", vector=[0.0], payload={
        "file": "doc3",
        "page": 3,
        "sha256": "sha3",
        "text": "third",
    })
    point3.score = np.float32(0.3)

    mocked_qdrant.search.return_value = [point1, point2, point3]

    results = qc.search_chunks("question", top_k=2)

    ensure_mock.assert_called_once()
    mocked_qdrant.search.assert_called_once()
    assert len(results) == 2
    first, second = results
    assert first == {
        "file": "doc1",
        "page": 1,
        "sha256": "sha1",
        "text": "first",
        "score": pytest.approx(0.5),
    }
    assert second == {
        "file": "doc2",
        "page": 2,
        "sha256": "sha2",
        "text": "second",
        "score": pytest.approx(0.4),
    }


def test_search_chunks_skips_search_for_empty_query(monkeypatch: pytest.MonkeyPatch, mocked_qdrant: MagicMock) -> None:
    ensure_mock = MagicMock()
    monkeypatch.setattr(qc, "ensure_collection", ensure_mock)
    monkeypatch.setattr(
        qc,
        "_encode_texts",
        MagicMock(return_value=np.zeros((0, qc.EMBED_DIMENSION), dtype=np.float32)),
    )

    results = qc.search_chunks("", top_k=3)

    ensure_mock.assert_called_once()
    mocked_qdrant.search.assert_not_called()
    assert results == []


def test_encode_texts_returns_zeros_for_empty_input() -> None:
    embeddings = qc._encode_texts([])

    assert embeddings.shape == (0, qc.EMBED_DIMENSION)
    assert embeddings.dtype == np.float32
    assert not embeddings.size or np.all(embeddings == 0)


def test_encode_texts_normalises_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySentenceTransformer:
        def __init__(self, *_: object, **__: object) -> None:
            self.calls = 0

        def get_sentence_embedding_dimension(self) -> int:
            return qc.EMBED_DIMENSION

        def encode(self, texts: Iterable[str], convert_to_numpy: bool = True) -> np.ndarray:
            self.calls += 1
            base = np.arange(1, qc.EMBED_DIMENSION + 1, dtype=np.float32)
            return np.stack([base * (i + 1) for i, _ in enumerate(texts)], axis=0)

    monkeypatch.setattr(qc, "SentenceTransformer", DummySentenceTransformer)

    embeddings = qc._encode_texts(["foo", "bar"])

    assert embeddings.shape == (2, qc.EMBED_DIMENSION)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

