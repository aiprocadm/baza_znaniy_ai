"""Unit tests for the vector store backends."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
        BOOL = "bool"

    class PointStruct:
        def __init__(self, id: str, vector: Sequence[float], payload: dict[str, object]):
            self.id = id
            self.vector = list(vector)
            self.payload = payload

    class Distance:
        COSINE = "cosine"

    class MatchValue:
        def __init__(self, value: object):
            self.value = value

    class MatchText:
        def __init__(self, text: str):
            self.text = text

    class FieldCondition:
        def __init__(self, key: str, match: object):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must: object = None, should: object = None, must_not: object = None):
            self.must = must
            self.should = should
            self.must_not = must_not

    models_module.VectorParams = VectorParams
    models_module.HnswConfigDiff = HnswConfigDiff
    models_module.PayloadSchemaType = PayloadSchemaType
    models_module.PointStruct = PointStruct
    models_module.Distance = Distance
    models_module.MatchValue = MatchValue
    models_module.MatchText = MatchText
    models_module.FieldCondition = FieldCondition
    models_module.Filter = Filter
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
        # info objects are SimpleNamespace so production's `info.config.params.vectors.size`
        # attribute chain (see app/retriever/qdrant.py:154) resolves on the second
        # ensure_ready call (search-after-upsert).
        self.collections: dict[str, SimpleNamespace] = {}
        self.upserts: list[list[dict[str, object]]] = []
        self.search_queries: list[dict[str, object]] = []

    def get_collection(self, name: str) -> object:
        if name not in self.collections:
            raise qmodule.UnexpectedResponse("missing collection")
        return self.collections[name]

    def recreate_collection(self, **kwargs: object) -> None:
        size = kwargs["vectors_config"].size
        self.collections[kwargs["collection_name"]] = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=size)))
        )

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
    # embed_batch_size=2 is set via _make_settings; QdrantVectorStore reads it on upsert.
    settings = _make_settings(tmp_path, backend="qdrant")

    embedder = _StubEmbedder("stub")

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
    # FAISS post-filters by tenant_id (faiss.py:247-251); chunks must carry it
    # (or owner, which upsert falls back to) for hits to survive filtering.
    chunks: Iterable[dict[str, object]] = [
        {"sha256": "a", "text": "alpha", "tenant_id": "test-tenant"},
        {"sha256": "b", "text": "beta", "tenant_id": "test-tenant"},
    ]

    store.upsert(chunks)
    hits = store.search(
        "query",
        top_k=1,
        filters=vs.SearchFilters.from_input(tenant_id="test-tenant"),
    )

    assert hits
    assert hits[0]["sha256"] in {"a", "b"}
    assert "score" in hits[0]
    assert hits[0]["tenant_id"] == "test-tenant"


def test_qdrant_search_builds_filter_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path, backend="qdrant")

    class _FieldCondition:
        def __init__(self, key: str, match: object):
            self.key = key
            self.match = match

    class _MatchValue:
        def __init__(self, value: object):
            self.value = value

    class _MatchText:
        def __init__(self, text: str):
            self.text = text

    class _Filter:
        def __init__(self, must: list[object]):
            self.must = must

    monkeypatch.setattr(qmodule.qmodels, "FieldCondition", _FieldCondition)
    monkeypatch.setattr(qmodule.qmodels, "MatchValue", _MatchValue)
    monkeypatch.setattr(qmodule.qmodels, "MatchText", _MatchText)
    monkeypatch.setattr(qmodule.qmodels, "Filter", _Filter)

    client = _StubClient()
    store = QdrantVectorStore(
        settings=settings,
        embedder_factory=lambda _: _StubEmbedder("stub"),
        client_factory=lambda **_: client,
    )

    store.search(
        "query",
        3,
        filters=vs.SearchFilters.from_input(
            tenant_id="tenant-a",
            owner="tenant-a",
            tags=["a", "b"],
            act_type="law",
            issuer="минюст",
            reg_number="123",
            is_active=False,
            revision_mode="historical",
        ),
    )

    sent_filter = client.search_queries[-1]["query_filter"]
    keys = [condition.key for condition in sent_filter.must]
    assert keys == [
        "tenant_id",
        "owner",
        "tags",
        "tags",
        "act_type",
        "issuer",
        "reg_number",
        "is_active",
        "is_active",
    ]
    # act_type (index 4) is exact-match; issuer (index 5) is full-text/substring.
    assert isinstance(sent_filter.must[4].match, _MatchValue)
    assert isinstance(sent_filter.must[5].match, _MatchText)
    # is_active=False must serialise as a MatchValue with literal False, not None
    # (the `is not None` branch in _to_qdrant_filter).
    assert isinstance(sent_filter.must[7].match, _MatchValue)
    assert sent_filter.must[7].match.value is False
    # revision_mode="historical" appends a second is_active=False condition.
    assert sent_filter.must[8].match.value is False
    # tenant_id and tags carry MatchValue, not MatchText (exact-match semantics).
    assert isinstance(sent_filter.must[0].match, _MatchValue)
    assert sent_filter.must[0].match.value == "tenant-a"
    assert {sent_filter.must[2].match.value, sent_filter.must[3].match.value} == {"a", "b"}


def test_qdrant_and_faiss_apply_same_filters(tmp_path: Path) -> None:
    chunks = [
        {
            "sha256": "1",
            "text": "alpha law",
            "file": "f",
            "page": 1,
            "owner": "tenant-a",
            "tenant_id": "tenant-a",
            "tags": ["prod"],
            "meta": {
                "act_type": "law",
                "issuer": "MinJust",
                "reg_number": "123",
                "is_active": True,
            },
        },
        {
            "sha256": "2",
            "text": "alpha law old",
            "file": "f",
            "page": 2,
            "owner": "tenant-a",
            "tenant_id": "tenant-a",
            "tags": ["prod"],
            "meta": {
                "act_type": "law",
                "issuer": "MinJust",
                "reg_number": "123",
                "is_active": False,
            },
        },
        {
            "sha256": "3",
            "text": "alpha other",
            "file": "f",
            "page": 3,
            "owner": "tenant-b",
            "tenant_id": "tenant-b",
            "tags": ["prod"],
            "meta": {
                "act_type": "law",
                "issuer": "MinJust",
                "reg_number": "123",
                "is_active": True,
            },
        },
    ]
    filters = vs.SearchFilters.from_input(
        tenant_id="tenant-a",
        tags=["prod"],
        act_type="law",
        issuer="min",
        reg_number="123",
        revision_mode="current",
    )

    fsettings = _make_settings(tmp_path / "fa", backend="faiss")
    fstore = FaissVectorStore(settings=fsettings, embedder_factory=lambda _: _StubEmbedder("stub"))
    fstore.upsert(chunks)
    fhits = fstore.search("alpha", top_k=5, filters=filters)

    client = _StubClient()
    qsettings = _make_settings(tmp_path / "qd", backend="qdrant")
    qstore = QdrantVectorStore(
        settings=qsettings,
        embedder_factory=lambda _: _StubEmbedder("stub"),
        client_factory=lambda **_: client,
    )
    qstore.upsert(chunks)
    qstore.search("alpha", top_k=5, filters=filters)

    qfilter = client.search_queries[-1]["query_filter"]
    must_keys = [c.key for c in qfilter.must]
    assert "tenant_id" in must_keys and "tags" in must_keys and "reg_number" in must_keys
    assert [h["sha256"] for h in fhits] == ["1"]
