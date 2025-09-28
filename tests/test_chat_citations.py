"""Tests for chat API citation handling."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

from fastapi.testclient import TestClient

# Provide lightweight stubs for optional heavy dependencies before importing the app.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("CHAT_DB_PATH", os.path.join(tempfile.gettempdir(), "chat_store_test.sqlite"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "qdrant_client" not in sys.modules:
    fake_qdrant = types.ModuleType("qdrant_client")
    fake_http = types.ModuleType("qdrant_client.http")
    fake_http_models = types.ModuleType("qdrant_client.http.models")
    fake_http_exceptions = types.ModuleType("qdrant_client.http.exceptions")

    class FakeUnexpectedResponse(Exception):
        pass

    class FakeVectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class FakeDistance:
        COSINE = "cosine"

    class FakePayloadSchemaType:
        KEYWORD = "keyword"
        INTEGER = "integer"

    class FakePointStruct:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_http_models.VectorParams = FakeVectorParams
    fake_http_models.Distance = FakeDistance
    fake_http_models.PayloadSchemaType = FakePayloadSchemaType
    fake_http_models.PointStruct = FakePointStruct
    fake_http_exceptions.UnexpectedResponse = FakeUnexpectedResponse

    class FakeQdrantClient:
        def __init__(self, **_kwargs):
            pass

        def get_collection(self, *_args, **_kwargs):
            raise FakeUnexpectedResponse

        def recreate_collection(self, **_kwargs):
            return None

        def create_payload_index(self, **_kwargs):
            return None

        def upsert(self, **_kwargs):  # pragma: no cover - not used in tests
            return None

        def search(self, **_kwargs):  # pragma: no cover - not used in tests
            return []

    fake_qdrant.QdrantClient = FakeQdrantClient
    fake_qdrant.http = fake_http
    fake_http.models = fake_http_models
    fake_http.exceptions = fake_http_exceptions

    sys.modules["qdrant_client"] = fake_qdrant
    sys.modules["qdrant_client.http"] = fake_http
    sys.modules["qdrant_client.http.models"] = fake_http_models
    sys.modules["qdrant_client.http.exceptions"] = fake_http_exceptions

if "sentence_transformers" not in sys.modules:
    fake_st = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, _name):
            pass

        def encode(self, texts, convert_to_numpy=True):
            import numpy as np

            return np.ones((len(texts), 3), dtype=float)

        def get_sentence_embedding_dimension(self):
            return 3

    fake_st.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_st

from app.main import app
from app.memory.store import MemoryStore


def test_chat_returns_unique_citations_and_shortage_flag(tmp_path, monkeypatch):
    client = TestClient(app)

    memory_path = tmp_path / "memory.sqlite"
    app.mem = MemoryStore(
        db_path=str(memory_path),
        ttl_days=90,
        summary_trigger=10,
        max_tokens=2000,
    )

    monkeypatch.setattr("app.main.ensure_model", lambda: None)
    monkeypatch.setattr("app.main.ensure_collection", lambda: None)
    monkeypatch.setattr("app.main.generate", lambda _prompt: "Ответ")

    hits = [
        {"file": "doc1.pdf", "page": 1, "score": 0.9},
        {"file": "doc1.pdf", "page": 1, "score": 0.8},
        {"file": "doc2.pdf", "page": 2, "score": 0.7},
    ]
    def fake_search_chunks(_msg, top_k=10, **_kwargs):
        return hits

    monkeypatch.setattr("app.main.search_chunks", fake_search_chunks)

    response = client.post(
        "/api/chat",
        json={"user_id": "tester", "message": "Привет", "conversation_id": "conv"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["conversation_id"]
    assert [c["file"] for c in payload["citations"]] == ["doc1.pdf", "doc2.pdf"]
    assert payload["citations_insufficient"] is True
    assert len(payload["citations"]) == 2

