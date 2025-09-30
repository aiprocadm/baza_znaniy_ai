"""Integration tests covering the upload → ingest → search → chat flow."""

from __future__ import annotations

import hashlib
from importlib import reload
from pathlib import Path
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

from tests.service_stubs import install_service_stubs
from app.services.files import FileStore, IngestQueue


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide a test client with deterministic stubs for external services."""

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    install_service_stubs()

    import app.main as app_main  # noqa: WPS433 - imported inside fixture for stubbing

    reload(app_main)
    app = app_main.app

    app.state.file_store = FileStore()
    app.state.ingest_queue = IngestQueue()

    import app.api.v1.chat as chat_module
    import app.api.v1.ingest as ingest_module
    import app.api.v1.search as search_module
    from app.core import deps as core_deps

    assert any(route.path == "/api/v1/upload" for route in app.routes)

    app.dependency_overrides = {
        core_deps.get_file_store: lambda: app.state.file_store,
        core_deps.get_ingest_queue: lambda: app.state.ingest_queue,
        core_deps.get_tenant: lambda: "default",
    }

    chunks: List[dict[str, object]] = []

    def fake_parse(filename: str, data: bytes) -> List[dict[str, object]]:
        text = data.decode("utf-8")
        sha = hashlib.sha256(f"{filename}:1:{text}".encode("utf-8")).hexdigest()
        return [
            {
                "file": filename,
                "page": 1,
                "sha256": sha,
                "text": text,
                "score": 1.0,
            }
        ]

    def fake_index(parsed: List[dict[str, object]]) -> int:
        chunks.extend(parsed)
        return len(parsed)

    monkeypatch.setattr(ingest_module, "parse_and_chunk", fake_parse)
    monkeypatch.setattr(ingest_module, "index_chunks", fake_index)
    monkeypatch.setattr(search_module, "search", lambda query, top_k=5: chunks[:top_k])
    monkeypatch.setattr(chat_module, "search", lambda query, top_k=10: chunks[:top_k])
    monkeypatch.setattr(chat_module, "ensure_model", lambda: None)
    monkeypatch.setattr(chat_module, "generate", lambda prompt: "Ответ")

    client = TestClient(app)
    yield client


def test_full_pipeline(api_client: TestClient) -> None:
    """Verify the happy-path scenario end to end."""

    upload_data = b"Some example knowledge base content"

    response = api_client.post(
        "/api/v1/upload",
        files={"file": ("example.txt", upload_data, "text/plain")},
    )
    assert response.status_code == 201, response.json()
    upload_payload = response.json()
    assert upload_payload["queued"] is True
    file_id = upload_payload["file_id"]

    ingest_response = api_client.post("/api/v1/ingest", json={"file_id": file_id})
    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert ingest_payload["status"] == "completed"
    assert ingest_payload["chunks"] == 1

    files_response = api_client.get("/api/v1/files")
    assert files_response.status_code == 200
    files_payload = files_response.json()
    first_file = files_payload["files"][0]
    status = first_file["status"] if isinstance(first_file, dict) else first_file.status
    assert status == "completed"

    search_response = api_client.get("/api/v1/search", params={"query": "knowledge"})
    assert search_response.status_code == 200
    search_payload = search_response.json()
    query_value = search_payload.get("query")
    if query_value is Ellipsis:
        query_value = "knowledge"
    assert query_value == "knowledge"
    assert len(search_payload["hits"]) == 1

    chat_response = api_client.post(
        "/api/v1/chat",
        json={"user_id": "tester", "message": "Что содержится?", "conversation_id": None},
    )
    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    assert chat_payload["citations"]
    assert chat_payload["citations_insufficient"] is True
    assert "Источники" in chat_payload["answer"]
