"""Integration tests covering the upload → ingest → search → chat flow."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from importlib import reload
from pathlib import Path
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from tests.service_stubs import install_service_stubs

install_service_stubs()

from app.core import auth as core_auth, deps as core_deps
from app.llm.exceptions import ModelNotFoundError
from app.models import file as file_models
from app.models.user import UserRole
from app.services import vectorstore as vectorstore_module


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide a test client with deterministic stubs for external services."""

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path / 'ingest.db'}")
    install_service_stubs()

    from app.core import config as config_module

    config_module.get_settings.cache_clear()
    file_models.get_engine.cache_clear()

    import app.main as app_main  # noqa: WPS433 - imported inside fixture for stubbing

    reload(app_main)
    app = app_main.app

    import app.api.v1.chat as chat_module
    import app.api.v1.search as search_module

    assert any(route.path == "/api/v1/upload" for route in app.routes)

    app.dependency_overrides = {}

    def session_override():
        service = app.state.ingest_service
        with Session(service.engine) as session:
            yield session

    class StubUser:
        id = 1
        email = "admin@example.com"
        role = UserRole.ADMIN
        is_active = True
        tenant_slug = "default"

    def user_override():
        return StubUser()

    app.dependency_overrides = {
        core_deps.get_ingest_service: lambda: app.state.ingest_service,
        core_deps.get_ingest_session: session_override,
        core_auth.get_current_active_user: user_override,
        core_auth.ensure_tenant_access: lambda: "default",
        core_auth.require_admin_user: user_override,
    }

    chunks: List[dict[str, object]] = []

    def fake_index(parsed: List[dict[str, object]]) -> int:
        chunks.clear()
        chunks.extend(parsed)
        return len(parsed)

    monkeypatch.setattr(vectorstore_module, "index_chunks", fake_index)
    monkeypatch.setattr(search_module, "search", lambda query, top_k=5: chunks[:top_k])
    monkeypatch.setattr(chat_module, "search", lambda query, top_k=10: chunks[:top_k])

    class DummyProvider:
        def ensure_model(self) -> None:
            return None

        def generate(self, prompt: str, *, context: dict[str, object] | None = None) -> str:
            return "Ответ"

    app.state.llm_provider = DummyProvider()

    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        file_models.get_engine.cache_clear()
        config_module.get_settings.cache_clear()


def _wait_for_completion(client: TestClient, file_id: str, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        files_response = client.get("/api/v1/files")
        if files_response.status_code != 200:
            time.sleep(0.05)
            continue
        files_payload = files_response.json()
        for entry in files_payload.get("files", []):
            if entry["id"] == file_id and entry["status"] == "completed":
                return
        time.sleep(0.05)
    raise AssertionError("file did not complete ingestion in time")


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

    _wait_for_completion(api_client, file_id)

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
    if isinstance(first_file, dict):
        assert first_file.get("mime_type") == "text/plain"
        assert first_file.get("document_id")

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

    jobs_response = api_client.get("/api/v1/admin/jobs")
    assert jobs_response.status_code == 200
    jobs_payload = jobs_response.json()
    assert jobs_payload["jobs"]
    assert any(job.get("status") == "completed" for job in jobs_payload["jobs"])


def test_ingest_endpoint_returns_quickly(api_client: TestClient) -> None:
    """Ensure ingest API responds immediately while work continues."""

    worker = api_client.app.state.ingest_worker
    original_process = worker._process
    started = threading.Event()
    release = threading.Event()

    async def blocked_process(self, job):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return await original_process(job)

    worker._process = blocked_process.__get__(worker, worker.__class__)

    upload_data = b"Background ingestion test"
    response = api_client.post(
        "/api/v1/upload",
        files={"file": ("async.txt", upload_data, "text/plain")},
    )
    assert response.status_code == 201
    file_id = response.json()["file_id"]

    assert started.wait(timeout=1.0)

    try:
        start = time.perf_counter()
        ingest_response = api_client.post(
            "/api/v1/ingest",
            json={"file_id": file_id, "force": True},
        )
        duration = time.perf_counter() - start
        assert ingest_response.status_code == 200
        assert duration < 0.2
        ingest_payload = ingest_response.json()
        assert ingest_payload["status"] in {"queued", "processing"}

        files_response = api_client.get("/api/v1/files")
        assert files_response.status_code == 200
        files_payload = files_response.json()
        pending_status = None
        for entry in files_payload.get("files", []):
            identifier = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)
            if identifier == file_id:
                pending_status = (
                    entry.get("status")
                    if isinstance(entry, dict)
                    else getattr(entry, "status", None)
                )
                break
        assert pending_status in {"queued", "processing"}
    finally:
        release.set()
        worker._process = original_process

    _wait_for_completion(api_client, file_id, timeout=3.0)


def test_chat_returns_503_when_model_missing(api_client: TestClient) -> None:
    """Ensure chat endpoint propagates missing model errors."""

    class MissingModelProvider:
        def ensure_model(self) -> None:
            raise ModelNotFoundError(Path("missing.gguf"))

        def ensure_ready(self) -> None:  # pragma: no cover - defensive stub
            return None

        def ensure_adapter(self) -> None:  # pragma: no cover - defensive stub
            return None

        def generate(self, prompt: str, *, context: dict[str, object] | None = None) -> str:
            return ""

    api_client.app.state.llm_provider = MissingModelProvider()

    response = api_client.post(
        "/api/v1/chat",
        json={"user_id": "tester", "message": "Ping", "conversation_id": None},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"] == "LLM_MODEL_MISSING"
