"""Tests for the qdrant service helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, List, Sequence

import numpy as np
import pytest


@dataclass
class StubSettings:
    embed_model: str = "test-model"
    embed_dimension: int = 3
    embed_batch_size: int = 32
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = "secret"
    qdrant_collection: str = "unit-test"
    retrieve_topk: int = 4
    rerank_topk: int = 2
    rerank_enabled: bool = True
    llm_model_name: str = "stub-model"
    ollama_model: str | None = None
    max_context_tokens: int = 3000


class SentenceTransformerStub:
    """Stub for SentenceTransformer that exposes hooks for tests."""

    dimension: int = StubSettings.embed_dimension
    encode_callback: Callable[[Sequence[str]], np.ndarray] | None = None
    init_calls: List[str] = []

    def __init__(self, model_name: str) -> None:  # pragma: no cover - simple init
        self.model_name = model_name
        SentenceTransformerStub.init_calls.append(model_name)

    def get_sentence_embedding_dimension(self) -> int:
        return self.__class__.dimension

    def encode(self, texts: Sequence[str], convert_to_numpy: bool = True) -> np.ndarray:
        if not convert_to_numpy:  # pragma: no cover - defensive
            raise AssertionError("encode should be called with convert_to_numpy=True")
        if self.__class__.encode_callback is None:
            return np.ones((len(texts), self.__class__.dimension), dtype=np.float64)
        return self.__class__.encode_callback(texts)


class PointStruct:
    def __init__(self, id: str, vector: Sequence[float], payload: dict[str, object]):
        self.id = id
        self.vector = list(vector)
        self.payload = payload


class SearchResult:
    def __init__(self, payload: dict[str, object], score: object):
        self.payload = payload
        self.score = score


class QdrantClientStub:
    """Stub Qdrant client capturing calls for assertions."""

    last_instance: "QdrantClientStub | None" = None

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = dict(kwargs)
        self.collection_info: object | None = None
        self.recreate_calls: list[dict[str, object]] = []
        self.create_index_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []
        self.search_results: list[SearchResult] = []
        QdrantClientStub.last_instance = self

    def get_collection(self, name: str) -> object:
        if self.collection_info is None:
            raise UnexpectedResponse("missing collection")
        return self.collection_info

    def recreate_collection(self, **kwargs: object) -> None:
        self.recreate_calls.append(dict(kwargs))

    def create_payload_index(self, **kwargs: object) -> None:
        self.create_index_calls.append(dict(kwargs))

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(dict(kwargs))

    def search(self, **kwargs: object) -> list[SearchResult]:
        self.search_calls.append(dict(kwargs))
        return list(self.search_results)


class VectorParams:
    def __init__(self, size: int, distance: object):
        self.size = size
        self.distance = distance


class Distance:
    COSINE = "cosine"


class HnswConfigDiff:
    def __init__(self, **kwargs: object) -> None:  # pragma: no cover - container
        self.kwargs = dict(kwargs)


class PayloadSchemaType:
    KEYWORD = "keyword"
    INTEGER = "integer"


class SearchParams:
    def __init__(self, **kwargs: object) -> None:  # pragma: no cover - container
        self.kwargs = dict(kwargs)


class UnexpectedResponse(Exception):
    pass


def _install_stub_modules(monkeypatch: pytest.MonkeyPatch, settings: StubSettings) -> None:
    sentence_module = types.ModuleType("sentence_transformers")
    sentence_module.SentenceTransformer = SentenceTransformerStub
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_module)

    qdrant_module = types.ModuleType("qdrant_client")
    qdrant_module.__path__ = []  # type: ignore[attr-defined]
    qdrant_module.QdrantClient = QdrantClientStub

    http_module = types.ModuleType("qdrant_client.http")
    http_module.__path__ = []  # type: ignore[attr-defined]

    models_module = types.ModuleType("qdrant_client.http.models")
    models_module.VectorParams = VectorParams
    models_module.Distance = Distance
    models_module.HnswConfigDiff = HnswConfigDiff
    models_module.PayloadSchemaType = PayloadSchemaType
    models_module.PointStruct = PointStruct
    models_module.SearchParams = SearchParams

    exceptions_module = types.ModuleType("qdrant_client.http.exceptions")
    exceptions_module.UnexpectedResponse = UnexpectedResponse

    http_module.models = models_module  # type: ignore[attr-defined]
    http_module.exceptions = exceptions_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models_module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.exceptions", exceptions_module)

    config_module = types.ModuleType("srv.projects.kb.app.config")

    def get_settings() -> StubSettings:
        return settings

    config_module.get_settings = get_settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "srv.projects.kb.app.config", config_module)


@pytest.fixture
def qdrant_context(monkeypatch: pytest.MonkeyPatch):
    settings = StubSettings()
    SentenceTransformerStub.dimension = settings.embed_dimension
    SentenceTransformerStub.encode_callback = None
    SentenceTransformerStub.init_calls = []
    QdrantClientStub.last_instance = None

    _install_stub_modules(monkeypatch, settings)

    module_name = f"srv.projects.kb.app.qdrant_client_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name, "srv/projects/kb/app/qdrant_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    yield types.SimpleNamespace(
        module=module,
        settings=settings,
        sentence_cls=SentenceTransformerStub,
        qdrant_cls=QdrantClientStub,
        qmodels=sys.modules["qdrant_client.http.models"],
    )

    sys.modules.pop(module_name, None)


def _make_collection_info(vectors: object) -> object:
    params = types.SimpleNamespace(vectors=vectors)
    config = types.SimpleNamespace(params=params)
    return types.SimpleNamespace(config=config)


def test_embedder_caches_instances(qdrant_context) -> None:
    module = qdrant_context.module
    sentence_cls = qdrant_context.sentence_cls

    embedder_a = module._embedder()
    embedder_b = module._embedder()

    assert embedder_a is embedder_b
    assert sentence_cls.init_calls == [qdrant_context.settings.embed_model]


def test_embedder_dimension_mismatch_raises(qdrant_context) -> None:
    module = qdrant_context.module
    sentence_cls = qdrant_context.sentence_cls
    sentence_cls.dimension = qdrant_context.settings.embed_dimension + 1
    module._model = None

    with pytest.raises(RuntimeError):
        module._embedder()


def test_qdrant_client_api_key_handling(qdrant_context) -> None:
    module = qdrant_context.module
    settings = qdrant_context.settings
    qdrant_cls = qdrant_context.qdrant_cls

    module._qdrant = None
    settings.qdrant_api_key = "super-secret"
    client = module._qdrant_client()
    assert isinstance(client, qdrant_cls)
    assert client.init_kwargs["url"] == settings.qdrant_url
    assert client.init_kwargs["api_key"] == "super-secret"

    module._qdrant = None
    settings.qdrant_api_key = ""
    client = module._qdrant_client()
    assert "api_key" not in client.init_kwargs


def test_ensure_collection_recreates_for_dimension_mismatch(qdrant_context) -> None:
    module = qdrant_context.module
    qdrant_cls = qdrant_context.qdrant_cls
    settings = qdrant_context.settings

    module._qdrant = None
    client = module._qdrant_client()
    client.collection_info = _make_collection_info(
        types.SimpleNamespace(size=settings.embed_dimension + 2)
    )

    module.ensure_collection()

    assert client.recreate_calls
    assert len(client.create_index_calls) == 3

    module._qdrant = None
    client = module._qdrant_client()
    client.collection_info = _make_collection_info(
        {
            "a": types.SimpleNamespace(size=settings.embed_dimension + 1),
            "b": types.SimpleNamespace(size=settings.embed_dimension),
        }
    )

    module.ensure_collection()

    assert client.recreate_calls


def test_ensure_collection_skips_when_matching(qdrant_context) -> None:
    module = qdrant_context.module

    module._qdrant = None
    client = module._qdrant_client()
    client.collection_info = _make_collection_info(
        types.SimpleNamespace(size=qdrant_context.settings.embed_dimension)
    )

    module.ensure_collection()

    assert client.recreate_calls == []
    assert client.create_index_calls == []


def test_upsert_chunks_deduplicates_and_normalises(qdrant_context) -> None:
    module = qdrant_context.module
    qmodels = qdrant_context.qmodels
    sentence_cls = qdrant_context.sentence_cls
    settings = qdrant_context.settings

    module._qdrant = None
    client = module._qdrant_client()
    client.collection_info = _make_collection_info(
        types.SimpleNamespace(size=settings.embed_dimension)
    )
    client.recreate_calls.clear()

    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64)
    sentence_cls.encode_callback = lambda texts: vectors[: len(texts)]

    chunks = [
        {"sha256": "dup", "text": "first", "file": "a", "page": 1},
        {"sha256": "dup", "text": "second", "file": "b", "page": 2},
        {"sha256": "uniq", "text": "third", "file": "c", "page": 3},
    ]

    module.upsert_chunks(chunks)

    assert client.upsert_calls
    points = client.upsert_calls[-1]["points"]
    assert len(points) == 2
    assert all(isinstance(point, qmodels.PointStruct) for point in points)
    first_payload = points[0].payload
    assert first_payload["text"] == "second"

    vectors_out = np.array([point.vector for point in points], dtype=np.float64)
    expected = np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(vectors_out, expected, atol=1e-6)


def test_upsert_chunks_requires_sha(qdrant_context) -> None:
    module = qdrant_context.module

    with pytest.raises(ValueError):
        module.upsert_chunks([{"text": "missing"}])


def test_search_chunks_respects_limit_and_formats_results(qdrant_context) -> None:
    module = qdrant_context.module
    sentence_cls = qdrant_context.sentence_cls
    settings = qdrant_context.settings

    module._qdrant = None
    client = module._qdrant_client()
    client.collection_info = _make_collection_info(
        types.SimpleNamespace(size=settings.embed_dimension)
    )
    client.search_results = [
        SearchResult({"file": "a", "page": 1, "sha256": "one", "text": "hello"}, Decimal("0.5")),
        SearchResult({"file": "b", "page": 2, "sha256": "two", "text": "world"}, 0.25),
    ]

    sentence_cls.encode_callback = lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    results = module.search_chunks("query", top_k=2)

    assert client.search_calls
    call = client.search_calls[-1]
    assert call["limit"] == 2
    assert np.isclose(call["query_vector"][0], 1.0)
    assert len(results) == 2
    assert all(isinstance(item["score"], float) for item in results)


def test_search_chunks_returns_empty_when_encoding_missing(qdrant_context) -> None:
    module = qdrant_context.module
    sentence_cls = qdrant_context.sentence_cls

    module._qdrant = None
    client = module._qdrant_client()
    client.collection_info = _make_collection_info(
        types.SimpleNamespace(size=qdrant_context.settings.embed_dimension)
    )

    sentence_cls.encode_callback = lambda texts: np.zeros(
        (0, qdrant_context.settings.embed_dimension), dtype=np.float32
    )

    results = module.search_chunks("query")

    assert results == []
    assert client.search_calls == []


def test_encode_texts_handles_empty_input(qdrant_context) -> None:
    module = qdrant_context.module
    sentence_cls = qdrant_context.sentence_cls

    embeddings = module._encode_texts([])

    assert embeddings.shape == (0, qdrant_context.settings.embed_dimension)
    assert embeddings.dtype == np.float32
    assert not sentence_cls.init_calls


def test_encode_texts_normalises_embeddings(qdrant_context) -> None:
    module = qdrant_context.module
    sentence_cls = qdrant_context.sentence_cls

    sentence_cls.encode_callback = lambda texts: np.array(
        [[3.0, 4.0, 0.0] for _ in texts], dtype=np.float64
    )

    embeddings = module._encode_texts(["example"])

    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings, np.array([[0.6, 0.8, 0.0]], dtype=np.float32))

