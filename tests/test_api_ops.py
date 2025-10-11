from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import ops


def _build_client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(ops.router)
    return TestClient(app)


def test_health_endpoint_returns_version(monkeypatch):
    monkeypatch.setattr(ops, "get_version_info", lambda: {"revision": "abc"})
    client = _build_client(monkeypatch)

    response = client.get("/ops/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": {"revision": "abc"}}


def test_warmup_endpoint_ensures_vector_store(monkeypatch):
    class _VectorStore:
        def __init__(self) -> None:
            self.ready_calls = 0

        def ensure_ready(self) -> None:
            self.ready_calls += 1

    store = _VectorStore()
    monkeypatch.setattr(ops, "get_vector_store", lambda: store)
    monkeypatch.setattr(ops, "get_version_info", lambda: {"revision": "abc"})

    client = _build_client(monkeypatch)
    response = client.post("/ops/warmup")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == {"revision": "abc"}
    assert isinstance(payload["vectorstore_ready_ms"], int)
    assert payload["vectorstore_ready_ms"] >= 0
    assert store.ready_calls == 1
