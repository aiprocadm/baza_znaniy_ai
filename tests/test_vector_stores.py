"""Unit tests for the vector store backends."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pytest

if "qdrant_client" not in sys.modules:
    qdrant_module = types.ModuleType("qdrant_client")

    class _StubQdrantClient:
        def __init__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - stub init
            self.args = args
            self.kwargs = kwargs

    qdrant_module.QdrantClient = _StubQdrantClient

    http_module = types.ModuleType("qdrant_client.http")
    models_module = types.ModuleType("qdrant_client.http.models")
    exceptions_module = types.ModuleType("qdrant_client.http.exceptions")

    class UnexpectedResponse(Exception):
        pass

    class VectorParams:
        def __init__(self, size: int, distance: object):
            self.size = size
            self.distance = distance

    class HnswConfigDiff:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class PayloadSchemaType:
        KEYWORD = "keyword"
        INTEGER = "integer"

    class PointStruct:
        def __init__(self, id: str, vector: Sequence[float], payload: dict[str, object]):
            self.id = id
            self.vector = list(vector)
            self.payload = payload

    class Distance:
        COSINE = "cosine"

    models_module.VectorParams = VectorParams
    models_module.HnswConfigDiff = HnswConfigDiff
    models_module.PayloadSchemaType = PayloadSchemaType
    models_module.PointStruct = PointStruct
    models_module.Distance = Distance
    exceptions_module.UnexpectedResponse = UnexpectedResponse

    qdrant_module.http = http_module
    http_module.models = models_module  # type: ignore[attr-defined]
    http_module.exceptions = exceptions_module  # type: ignore[attr-defined]

    sys.modules["qdrant_client"] = qdrant_module
    sys.modules["qdrant_client.http"] = http_module
    sys.modules["qdrant_client.http.models"] = models_module
    sys.modules["qdrant_client.http.exceptions"] = exceptions_module

from app.core.config import Settings
from app.retriever import FaissVectorStore, QdrantVectorStore
from app.retriever import vector_store as vs
from app.retriever import qdrant as qmodule


@dataclass
class _StubRecord:
    payload: dict[str, object]
    score: float
    id: str | None = None


class _StubClient:
    def __init__(self, *_: object, **__: object) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.upserts: list[list[dict[str, object]]] = []
        self.search_queries: list[dict[str, object]] = []

    def get_collection(self, name: str) -> object:
        if name not in self.collections:
            raise qmodule.UnexpectedResponse("missing collection")
        return self.collections[name]

    def recreate_collection(self, **kwargs: object) -> None:
        self.collections[kwargs["collection_name"]] = {
            "config": type("cfg", (), {"params": type("params", (), {"vectors": type("vec", (), {"size": kwargs["vectors_config"].size})()})()})()
        }

    def create_payload_index(self, **_: object) -> None:
        pass

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs["points"])

    def search(self, **kwargs: object) -> list[_StubRecord]:
        self.search_queries.append(kwargs)
        return [
            _StubRecord({"text": "stub", "file": "f"}, score=0.42, id="abc123"),
        ]


class _StubEmbedder:
    def __init__(self, model_name: str) -> None:  # pragma: no cover - simple init
        self.model_name = model_name
        self.calls: list[Sequence[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray:
        assert convert_to_numpy is True
        self.calls.append(tuple(texts))
        return np.ones((len(texts), 3), dtype=np.float32)


def _make_settings(tmp_path: Path, backend: str, **overrides: object) -> Settings:
    params: dict[str, object] = {
        "data_dir": tmp_path,
        "vector_backend": backend,
        "vector_embed_model": "stub",
        "vector_embed_dimension": 3,
        "embed_batch_size": 2,
    }
    params.update(overrides)
    return Settings(**params)


def test_get_vector_store_selects_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubFaiss:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

    class _StubQdrant:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

    def _stub_factory(settings: Settings) -> object:
        if settings.vector_backend == "faiss":
            return _StubFaiss(settings)
        return _StubQdrant(settings)

    monkeypatch.setattr(vs, "_build_backend", _stub_factory)

    settings = _make_settings(tmp_path, backend="faiss")
    vs.get_vector_store.cache_clear()
    store = vs.get_vector_store(settings)
    assert isinstance(store, _StubFaiss)

    settings_q = _make_settings(tmp_path, backend="qdrant")
    vs.get_vector_store.cache_clear()
    store_q = vs.get_vector_store(settings_q)
    assert isinstance(store_q, _StubQdrant)
    vs.get_vector_store.cache_clear()


def test_qdrant_upsert_batches_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, backend="qdrant")

    embedder = _StubEmbedder("stub")
    settings.embed_batch_size = 2

    # Provide lightweight stand-ins for qdrant models
    class _PointStruct:
        def __init__(self, id: str, vector: Sequence[float], payload: dict[str, object]):
            self.id = id
            self.vector = list(vector)
            self.payload = payload

    class _VectorParams:
        def __init__(self, size: int, distance: object):
            self.size = size
            self.distance = distance

    class _HnswConfigDiff:
        def __init__(self, **_: object) -> None:
            pass

    class _PayloadSchemaType:
        KEYWORD = "keyword"
        INTEGER = "integer"

    monkeypatch.setattr(qmodule.qmodels, "PointStruct", _PointStruct)
    monkeypatch.setattr(qmodule.qmodels, "VectorParams", _VectorParams)
    monkeypatch.setattr(qmodule.qmodels, "HnswConfigDiff", _HnswConfigDiff)
    monkeypatch.setattr(qmodule.qmodels, "PayloadSchemaType", _PayloadSchemaType)

    store = QdrantVectorStore(
        settings=settings,
        embedder_factory=lambda _: embedder,
        client_factory=lambda **_: _StubClient(),
    )

    chunks = [
        {"sha256": "1", "text": "a", "file": "f", "page": 1},
        {"sha256": "2", "text": "b", "file": "f", "page": 2},
        {"sha256": "3", "text": "c", "file": "f", "page": 3},
    ]

    store.upsert(chunks)

    # Two batches: [a, b] and [c]
    assert [list(call) for call in embedder.calls] == [["a", "b"], ["c"]]


def test_qdrant_upsert_stream_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, backend="qdrant")
    object.__setattr__(settings, "qdrant_upsert_batch", 2)

    embedder = _StubEmbedder("stub")

    class _PointStruct:
        def __init__(self, id: str, vector: Sequence[float], payload: dict[str, object]):
            self.id = id
            self.vector = list(vector)
            self.payload = payload

    monkeypatch.setattr(qmodule.qmodels, "PointStruct", _PointStruct)

    client = _StubClient()
    store = QdrantVectorStore(
        settings=settings,
        embedder_factory=lambda _: embedder,
        client_factory=lambda **_: client,
    )

    chunks = (
        {"sha256": str(index), "text": f"chunk-{index}", "file": "f", "page": index}
        for index in range(5)
    )

    store.upsert(chunks)

    assert len(client.upserts) == 3  # 5 items batched by 2 => 3 network calls
    assert sum(len(batch) for batch in client.upserts) == 5


def test_qdrant_initialises_embedded_client_when_url_missing(tmp_path: Path) -> None:
    storage_dir = tmp_path / "embedded"
    settings = _make_settings(
        tmp_path,
        backend="qdrant",
        qdrant_url="",
        qdrant_path=storage_dir,
    )

    called: dict[str, object] = {}

    def _client_factory(**kwargs: object) -> _StubClient:
        called.update(kwargs)
        return _StubClient()

    store = QdrantVectorStore(
        settings=settings,
        embedder_factory=lambda _: _StubEmbedder("stub"),
        client_factory=_client_factory,
    )

    store._client_instance()

    assert called == {"path": str(settings.qdrant_path_resolved)}
    assert settings.qdrant_path_resolved.exists()


def test_faiss_search_returns_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, backend="faiss")

    class _Embedder(_StubEmbedder):
        def encode(self, texts: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray:
            base = super().encode(texts, convert_to_numpy)
            # Produce slightly different vectors for deterministic ordering
            factors = np.arange(1, len(texts) + 1, dtype=np.float32).reshape(-1, 1)
            return base * factors

    store = FaissVectorStore(settings=settings, embedder_factory=_Embedder)
    chunks: Iterable[dict[str, object]] = [
        {"sha256": "a", "text": "alpha"},
        {"sha256": "b", "text": "beta"},
    ]

    store.upsert(chunks)
    hits = store.search("query", top_k=1)

    assert hits
    assert hits[0]["sha256"] in {"a", "b"}
    assert "score" in hits[0]

