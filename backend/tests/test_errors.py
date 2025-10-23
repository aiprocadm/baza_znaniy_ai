from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from backend.app.db.utils import init_db
from backend.app.main import create_app


def _create_client(*, raise_server_exceptions: bool = True) -> TestClient:
    init_db()
    app = create_app()
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_validation_error_uses_standard_format() -> None:
    client = _create_client()

    response = client.post("/api/v1/documents/generate", json={})

    assert response.status_code == 422
    payload = response.json()
    assert set(payload) == {"code", "message", "details", "trace_id"}
    assert payload["code"] == "validation_error"
    assert payload["message"] == "Request validation failed"
    assert isinstance(payload["details"], list)
    assert payload["details"]  # not empty
    assert isinstance(payload["trace_id"], str) and payload["trace_id"]


def test_http_exception_uses_standard_format() -> None:
    client = _create_client()

    response = client.post(
        "/api/v1/documents/generate",
        json={"template_id": "missing", "document_name": "name", "context": {}},
    )

    assert response.status_code == 404
    payload = response.json()
    assert set(payload) == {"code", "message", "details", "trace_id"}
    assert payload["code"] == "http_404"
    assert payload["message"] == "Template not found"
    assert payload["details"] is None
    assert isinstance(payload["trace_id"], str) and payload["trace_id"]


def test_unhandled_exception_uses_standard_format() -> None:
    client = _create_client(raise_server_exceptions=False)

    router = APIRouter()

    @router.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    client.app.include_router(router)

    response = client.get("/boom")

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "internal_server_error"
    assert payload["message"] == "Internal server error"
    assert payload["details"] is None
    assert isinstance(payload["trace_id"], str) and payload["trace_id"]
    assert response.headers["X-Trace-Id"] == payload["trace_id"]
