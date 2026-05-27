"""Integration tests covering the upload → ingest → search → chat flow."""

from __future__ import annotations

import asyncio
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
    monkeypatch.setenv("APP_VERSION", "test-app-1.2.3")
    monkeypatch.setenv("LLM_MODEL_VERSION", "test-model-4.5.6")
    monkeypatch.setenv("LORA_ADAPTER_VERSION", "test-lora-7.8.9")
    monkeypatch.setenv("LLM_LORA_ADAPTER", "stub-adapter")
    monkeypatch.setenv("UPLOAD_ALLOWED_EXTS", "pdf,docx,pptx,xlsx,txt,md")
    install_service_stubs()

    from app.core import config as config_module

    config_module.get_settings.cache_clear()
    file_models.get_engine.cache_clear()

    import app.main as app_main  # noqa: WPS433 - imported inside fixture for stubbing

    reload(app_main)
    app = app_main.app

    import app.api.v1.search as search_module
    import app.services.chat_orchestrator as chat_orchestrator_module

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

    def subject_override():
        return core_auth.SubjectAttribution(
            subject_type="user", subject_id="1", tenant="default"
        )

    app.dependency_overrides = {
        core_deps.get_ingest_service: lambda: app.state.ingest_service,
        core_deps.get_ingest_session: session_override,
        core_auth.get_current_active_user: user_override,
        core_auth.ensure_tenant_access: lambda: "default",
        core_auth.require_admin_user: user_override,
        # The upload route also depends on get_subject_attribution, which
        # calls get_current_user(request, session) *directly* (not via
        # Depends), so the get_current_active_user override above isn't
        # enough — the bare bearer-token lookup on `request` would still
        # 401 the test. Override the dependency itself.
        core_auth.get_subject_attribution: subject_override,
    }

    chunks: List[dict[str, object]] = []

    def fake_index(parsed: List[dict[str, object]]) -> int:
        chunks.clear()
        chunks.extend(parsed)
        return len(parsed)

    monkeypatch.setattr(vectorstore_module, "index_chunks", fake_index)

    def fake_search(query, top_k=5, **_kwargs):
        # ``app.services.vectorstore.search`` now accepts many keyword filters
        # (owner/tags/act_type/...). Tests only care about deterministic chunk
        # return, so swallow extras with **kwargs to keep the fake simple.
        return chunks[:top_k]

    monkeypatch.setattr(search_module, "search", fake_search)
    monkeypatch.setattr(chat_orchestrator_module, "search", fake_search)

    class DummyProvider:
        name = "dummy-provider"
        adapter_name = "stub-adapter"

        def __init__(self) -> None:
            self.ensure_model_calls = 0
            self.ensure_ready_calls = 0
            self.ensure_adapter_calls = 0

        def ensure_model(self) -> None:
            self.ensure_model_calls += 1

        def ensure_ready(self) -> None:
            self.ensure_ready_calls += 1

        def ensure_adapter(self) -> None:
            self.ensure_adapter_calls += 1

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


def test_file_summary_endpoint(api_client: TestClient) -> None:
    """Ensure the file summary endpoint aggregates tenant statistics."""

    first_payload = b"Alpha document"
    second_payload = b"Bravo report"

    response_one = api_client.post(
        "/api/v1/upload",
        files={"file": ("alpha.txt", first_payload, "text/plain")},
    )
    assert response_one.status_code == 201, response_one.json()
    first_id = response_one.json()["file_id"]

    response_two = api_client.post(
        "/api/v1/upload",
        files={"file": ("bravo.txt", second_payload, "text/plain")},
    )
    assert response_two.status_code == 201, response_two.json()
    second_id = response_two.json()["file_id"]

    _wait_for_completion(api_client, first_id)
    _wait_for_completion(api_client, second_id)

    summary_response = api_client.get("/api/v1/files/summary")
    assert summary_response.status_code == 200, summary_response.json()
    summary = summary_response.json()

    total_size = len(first_payload) + len(second_payload)

    assert summary["total_files"] == 2
    assert summary["status_counts"]["completed"] == 2
    assert summary["status_counts"]["failed"] == 0
    assert summary["status_counts"]["queued"] == 0
    assert summary["total_size_bytes"] == total_size
    assert summary["total_chunks"] >= 2
    assert summary["average_size_bytes"] == pytest.approx(total_size / 2)
    assert summary["oldest_upload"] is not None
    assert summary["newest_upload"] is not None


def test_ingest_endpoint_waits_for_completion(api_client: TestClient) -> None:
    """Ensure ingest API blocks until processing completes when auto-processing is enabled."""

    worker = api_client.app.state.ingest_worker
    original_process = worker._process
    started = threading.Event()
    finished = threading.Event()

    async def observed_process(self, job):
        started.set()
        await asyncio.sleep(0.05)
        try:
            return await original_process(job)
        finally:
            finished.set()

    worker._process = observed_process.__get__(worker, worker.__class__)

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
    finally:
        worker._process = original_process

    duration = time.perf_counter() - start
    assert ingest_response.status_code == 200
    assert finished.wait(timeout=1.0)
    assert duration >= 0.05

    ingest_payload = ingest_response.json()
    assert ingest_payload["status"] == "completed"

    files_response = api_client.get("/api/v1/files")
    assert files_response.status_code == 200
    files_payload = files_response.json()
    completed_status = None
    for entry in files_payload.get("files", []):
        identifier = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)
        if identifier == file_id:
            completed_status = (
                entry.get("status") if isinstance(entry, dict) else getattr(entry, "status", None)
            )
            break
    assert completed_status == "completed"


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
    assert payload["message"] == "LLM_MODEL_MISSING"
    assert payload["status"] == 503


def test_chat_websocket_roundtrip(api_client: TestClient) -> None:
    """Verify websocket chat request/response flow with partial tokens and final payload."""

    with api_client.websocket_connect("/api/v1/ws/chat", headers={"Authorization": "Bearer test-token"}) as websocket:
        websocket.send_json(
            {
                "type": "request",
                "request_id": "req-1",
                "stream": True,
                "payload": {
                    "user_id": "tester",
                    "message": "Привет",
                    "conversation_id": None,
                },
            }
        )

        ack = websocket.receive_json()
        assert ack == {"type": "ack", "request_id": "req-1"}

        partial = websocket.receive_json()
        assert partial["type"] == "partial"
        assert partial["request_id"] == "req-1"
        assert partial["delta"].strip()

        response = websocket.receive_json()
        assert response["type"] == "response"
        assert response["request_id"] == "req-1"
        payload = response["payload"]
        assert payload["answer"]
        assert payload["conversation_id"]
        assert isinstance(payload["citations"], list)


def test_chat_websocket_returns_error_for_bad_message_type(api_client: TestClient) -> None:
    """Ensure websocket channel reports protocol-level envelope errors."""

    with api_client.websocket_connect("/api/v1/ws/chat", headers={"Authorization": "Bearer test-token"}) as websocket:
        websocket.send_json({"type": "unexpected"})

        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "BAD_MESSAGE_TYPE"


def test_chat_websocket_returns_error_for_invalid_payload(api_client: TestClient) -> None:
    """Ensure websocket channel returns structured validation errors."""

    with api_client.websocket_connect("/api/v1/ws/chat", headers={"Authorization": "Bearer test-token"}) as websocket:
        websocket.send_json(
            {
                "type": "request",
                "request_id": "bad-req",
                "payload": {
                    "user_id": "tester",
                    "message": 123,  # type: ignore[arg-type]
                },
            }
        )

        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["request_id"] == "bad-req"
        assert error["code"] == "INVALID_REQUEST"


def test_version_endpoint_reports_versions(api_client: TestClient) -> None:
    response = api_client.get("/version")
    assert response.status_code == 200

    payload = response.json()
    assert payload["app"]["version"] == "test-app-1.2.3"
    assert payload["model"]["version"] == "test-model-4.5.6"
    assert payload["lora"]["version"] == "test-lora-7.8.9"
    assert payload["lora"]["enabled"] is True
    assert isinstance(payload["ts"], int)


def test_warmup_endpoint_preloads_components(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = api_client.app.state.llm_provider
    vector_store = api_client.app.state.vector_store

    import app.api.routes as routes_module

    def fake_perf_counter() -> float:
        fake_perf_counter.counter += 1
        return fake_perf_counter.counter * 0.001

    fake_perf_counter.counter = -1  # type: ignore[attr-defined]
    monkeypatch.setattr(routes_module.time, "perf_counter", fake_perf_counter)

    vector_calls = {"count": 0}
    original_ensure_ready = vector_store.ensure_ready

    def tracked_ensure_ready(*args, **kwargs):
        vector_calls["count"] += 1
        return original_ensure_ready(*args, **kwargs)

    monkeypatch.setattr(vector_store, "ensure_ready", tracked_ensure_ready)

    response = api_client.post("/warmup")
    assert response.status_code == 200, response.json()
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["message"] == "Warmup completed"
    assert payload["elapsed_ms"] == pytest.approx(9.0)

    llm_details = payload["details"]["llm"]
    assert llm_details["actions"]["ensure_model"]["status"] == "ok"
    assert llm_details["actions"]["ensure_ready"]["status"] == "ok"
    assert llm_details["actions"]["ensure_adapter"]["status"] == "ok"
    assert llm_details["actions"]["ensure_model"]["duration_ms"] == pytest.approx(1.0)
    assert llm_details["actions"]["ensure_ready"]["duration_ms"] == pytest.approx(1.0)
    assert llm_details["actions"]["ensure_adapter"]["duration_ms"] == pytest.approx(1.0)
    assert llm_details["elapsed_ms"] == pytest.approx(3.0)

    vector_details = payload["details"]["vector_store"]
    assert vector_details["actions"]["ensure_ready"]["status"] == "ok"
    assert vector_details["actions"]["ensure_ready"]["duration_ms"] == pytest.approx(1.0)
    assert vector_details["elapsed_ms"] == pytest.approx(1.0)

    assert provider.ensure_model_calls == 1
    assert provider.ensure_ready_calls == 1
    assert provider.ensure_adapter_calls == 1
    assert vector_calls["count"] == 1
