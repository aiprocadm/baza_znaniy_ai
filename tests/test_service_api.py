"""Integration tests for the knowledge base service API."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import logging
import os
import shutil
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, status
from fastapi.testclient import TestClient

from app.core.datetime_utils import utc_now
from app.services.files import FileRecord as UploadFileRecord, IngestStatus
from tests.demo_assets import ensure_demo_assets

from tests.service_stubs import install_service_stubs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb" / "app"
INDEX_HTML = PROJECT_ROOT / "srv" / "projects" / "kb" / "data" / "www" / "index.html"


def _load_service_app(tmp_path: Path) -> Any:
    package_name = "kb_service_app"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    install_service_stubs()

    if "app.api.v1" not in sys.modules:
        v1_module = types.ModuleType("app.api.v1")
        v1_module.router = APIRouter()
        sys.modules["app.api.v1"] = v1_module

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    if "qdrant_client" not in sys.modules:
        qdrant_package = types.ModuleType("qdrant_client")
        qdrant_package.__path__ = []  # type: ignore[attr-defined]

        class _DummyQdrantClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        http_module = types.ModuleType("qdrant_client.http")
        http_module.__path__ = []  # type: ignore[attr-defined]

        models_module = types.ModuleType("qdrant_client.http.models")

        class _VectorParams:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class _HnswConfigDiff:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class _SearchParams:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class _PointStruct:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class _Distance:
            COSINE = "cosine"

        class _PayloadSchemaType:
            KEYWORD = "keyword"
            INTEGER = "integer"

        models_module.VectorParams = _VectorParams
        models_module.Distance = _Distance
        models_module.HnswConfigDiff = _HnswConfigDiff
        models_module.PayloadSchemaType = _PayloadSchemaType
        models_module.PointStruct = _PointStruct
        models_module.SearchParams = _SearchParams

        exceptions_module = types.ModuleType("qdrant_client.http.exceptions")

        class _UnexpectedResponse(Exception):
            pass

        exceptions_module.UnexpectedResponse = _UnexpectedResponse

        qdrant_package.QdrantClient = _DummyQdrantClient
        http_module.models = models_module
        http_module.exceptions = exceptions_module

        sys.modules["qdrant_client"] = qdrant_package
        sys.modules["qdrant_client.http"] = http_module
        sys.modules["qdrant_client.http.models"] = models_module
        sys.modules["qdrant_client.http.exceptions"] = exceptions_module

    config_spec = importlib.util.spec_from_file_location(
        f"{package_name}.config", SERVICE_ROOT / "config.py"
    )
    assert config_spec and config_spec.loader
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)
    config_module.get_settings.cache_clear()

    os.environ.setdefault("DATA_DIR", str(tmp_path))

    main_spec = importlib.util.spec_from_file_location(
        f"{package_name}.main", SERVICE_ROOT / "main.py"
    )
    assert main_spec and main_spec.loader
    main_module = importlib.util.module_from_spec(main_spec)
    sys.modules[main_spec.name] = main_module
    main_spec.loader.exec_module(main_module)
    return main_module


@pytest.fixture()
def service_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "0")

    app_module = _load_service_app(tmp_path)

    settings = app_module.get_settings()
    settings.data_dir = Path(settings.data_dir)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)

    monkeypatch.setattr(app_module, "ensure_collection", lambda: None)
    monkeypatch.setattr(app_module, "ensure_model", lambda: None)
    monkeypatch.setattr(app_module, "upsert_chunks", lambda chunks: None)
    monkeypatch.setattr(app_module, "generate", lambda prompt: "Ответ")

    def fake_search_chunks(_query: str, top_k: int = 10):
        return [
            {"file": "doc1.pdf", "page": 1, "score": 0.9},
            {"file": "doc1.pdf", "page": 1, "score": 0.8},
            {"file": "doc2.pdf", "page": 2, "score": 0.7},
        ][:top_k]

    monkeypatch.setattr(app_module, "search_chunks", fake_search_chunks)

    return app_module


def test_health_get_returns_status_and_timestamp(service_app: Any):
    with TestClient(service_app.app) as client:
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "ok"
        assert isinstance(data["ts"], int)
        assert data["ts"] > 0


def test_health_head_returns_status_and_timestamp(service_app: Any):
    with TestClient(service_app.app) as client:
        response = client._request("HEAD", "/health")
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "ok"
        assert isinstance(data["ts"], int)
        assert data["ts"] > 0


def test_upload_rejects_invalid_extension(service_app: Any):
    with TestClient(service_app.app) as client:
        response = client.post(
            "/api/docs/upload",
            data={"user_id": "tester"},
            files={"files": ("image.png", b"binary", "image/png")},
        )

        expected = getattr(status, "HTTP_415_UNSUPPORTED_MEDIA_TYPE", 415)
        assert response.status_code == expected
        assert response.json()["detail"] == "UPLOAD_INVALID_EXT"


def test_chat_returns_citations(service_app: Any):
    with TestClient(service_app.app) as client:
        payload = {"user_id": "tester", "message": "Привет", "conversation_id": "conv"}
        first_response = client.post("/api/chat", json=payload)
        assert first_response.status_code == 200

        second_response = client.post("/api/chat", json=payload)
        data = second_response.json()

        assert second_response.status_code == 200
        assert data["citations"]
        assert len(data["citations"]) == 2
        assert data["citations_insufficient"] is True


def test_ingest_metrics_endpoint_reports_queue_state(
    service_app: Any, tmp_path: Path
) -> None:
    app = service_app.app
    file_store = app.state.file_store
    ingest_queue = app.state.ingest_queue

    payload_path = tmp_path / "payload.txt"
    payload_path.write_text("payload")

    now = utc_now()
    tenant = "default"

    pending = UploadFileRecord(
        id="p-1",
        filename="pending.txt",
        tenant=tenant,
        path=payload_path,
        size=payload_path.stat().st_size,
        uploaded_at=now - timedelta(minutes=5),
        status=IngestStatus.PENDING,
    )
    failed = UploadFileRecord(
        id="f-1",
        filename="failed.txt",
        tenant=tenant,
        path=payload_path,
        size=payload_path.stat().st_size,
        uploaded_at=now - timedelta(minutes=1),
        status=IngestStatus.FAILED,
    )
    failed.error = "conversion failed"

    for record in (pending, failed):
        file_store.add(record)
        ingest_queue.enqueue(record.id)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/ingest/metrics",
            headers={"x-tenant": tenant},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_files"] == 2
    assert payload["queue_depth"] == 2
    assert payload["status_counts"]["pending"] == 1
    assert payload["status_counts"]["failed"] == 1
    assert payload["status_counts"]["processing"] == 0
    assert payload["recent_failures"]
    assert payload["recent_failures"][0]["file_id"] == failed.id
    assert payload["recent_failures"][0]["error"] == "conversion failed"
    failure_uploaded = payload["recent_failures"][0]["uploaded_at"].replace("Z", "+00:00")
    last_activity = payload["last_activity_at"].replace("Z", "+00:00")
    assert datetime.fromisoformat(failure_uploaded).tzinfo is not None
    assert datetime.fromisoformat(last_activity).tzinfo is not None
    assert payload["oldest_pending_age_seconds"] > 0


def test_upload_returns_expected_response(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sample_dir = tmp_path / "samples"
    ensure_demo_assets(sample_dir)
    sample_path = sample_dir / "demo_notes.txt"
    sample_bytes = sample_path.read_bytes()

    captured_chunks: list[list[dict[str, Any]]] = []
    call_counts = {"parse": 0, "upsert": 0}

    def fake_parse_and_chunk(filename: str, data: bytes):
        call_counts["parse"] += 1
        assert data == sample_bytes
        return [
            {"file": filename, "page": 1, "content": "chunk-1"},
            {"file": filename, "page": 2, "content": "chunk-2"},
        ]

    def fake_upsert_chunks(chunks: list[dict[str, Any]]):
        call_counts["upsert"] += 1
        captured_chunks.append(list(chunks))

    monkeypatch.setattr(service_app, "parse_and_chunk", fake_parse_and_chunk)
    monkeypatch.setattr(service_app, "upsert_chunks", fake_upsert_chunks)

    settings = service_app.get_settings()
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    with TestClient(service_app.app) as client:
        response = client.post(
            "/api/docs/upload",
            data={"user_id": "tester"},
            files={"files": (sample_path.name, sample_bytes, "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload == {
        "ok": True,
        "files": [sample_path.name],
        "chunks": 2,
    }

    assert call_counts == {"parse": 1, "upsert": 1}
    assert captured_chunks == [
        [
            {"file": sample_path.name, "page": 1, "content": "chunk-1"},
            {"file": sample_path.name, "page": 2, "content": "chunk-2"},
        ]
    ]


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        (
            "slides.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        (
            "spreadsheet.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        ("notes.md", "text/markdown"),
    ],
)
def test_upload_accepts_configured_extensions(
    service_app: Any, monkeypatch: pytest.MonkeyPatch, filename: str, content_type: str
) -> None:
    call_counts = {"parse": 0, "index": 0}
    extension = filename.rsplit(".", 1)[-1]

    def fake_parse(target: str, data: bytes):
        call_counts["parse"] += 1
        assert target == filename
        assert data == b"payload"
        return [{"file": target, "page": 1, "content": "chunk"}]

    def fake_index(request: Any, chunks: Any) -> int:
        call_counts["index"] += 1
        items = list(chunks)
        assert items and items[0]["file"] == filename
        limits = getattr(request.app.state, "upload_limits", None)
        assert limits is not None
        allowed = getattr(limits, "allowed_extensions", set())
        assert extension in allowed
        return len(items)

    monkeypatch.setattr(service_app, "parse_and_chunk", fake_parse)
    monkeypatch.setattr("app.api.routes.parse_and_chunk", fake_parse)
    monkeypatch.setattr("app.api.routes._index_chunks", fake_index)

    with TestClient(service_app.app) as client:
        response = client.post(
            "/api/docs/upload",
            data={"user_id": "tester"},
            files={"files": (filename, b"payload", content_type)},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["ok"] is True
    assert payload["chunks"] == 1
    assert call_counts == {"parse": 1, "index": 1}


def test_upload_returns_no_text_found_when_chunks_missing(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sample_dir = tmp_path / "samples"
    ensure_demo_assets(sample_dir)
    sample_path = sample_dir / "demo_notes.txt"
    sample_bytes = sample_path.read_bytes()

    call_counts = {"parse": 0, "save": 0, "upsert": 0}

    def fake_parse_and_chunk(filename: str, data: bytes):
        call_counts["parse"] += 1
        assert data == sample_bytes
        return []

    def fake_save_file(path: Path, data: bytes):
        call_counts["save"] += 1

    def fake_upsert_chunks(chunks: list[dict[str, Any]]):
        call_counts["upsert"] += 1

    monkeypatch.setattr(service_app, "parse_and_chunk", fake_parse_and_chunk)
    monkeypatch.setattr(service_app, "_save_file", fake_save_file)
    monkeypatch.setattr(service_app, "upsert_chunks", fake_upsert_chunks)

    settings = service_app.get_settings()
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    with TestClient(service_app.app) as client:
        response = client.post(
            "/api/docs/upload",
            data={"user_id": "tester"},
            files={"files": (sample_path.name, sample_bytes, "text/plain")},
        )

    assert response.status_code == 400
    payload = response.json()

    assert payload["detail"] == "NO_TEXT_FOUND"
    assert payload.get("ok") is not True

    assert call_counts == {"parse": 1, "save": 0, "upsert": 0}


@pytest.mark.parametrize(
    "asset_specs",
    [
        [
            (
                "demo_contract.pdf",
                "application/pdf",
                [
                    {"file": "demo_contract.pdf", "page": 1, "content": "contract-chunk-1"},
                    {"file": "demo_contract.pdf", "page": 2, "content": "contract-chunk-2"},
                ],
            ),
            (
                "demo_overview.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                [
                    {"file": "demo_overview.docx", "page": 1, "content": "overview-chunk-1"},
                    {"file": "demo_overview.docx", "page": 2, "content": "overview-chunk-2"},
                ],
            ),
        ]
    ],
)
def test_docs_upload_indexes_each_demo_asset(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset_specs: list[tuple[str, str, list[dict[str, Any]]]]
):
    sample_dir = tmp_path / "samples"
    ensure_demo_assets(sample_dir)

    files_payload: list[tuple[str, tuple[str, bytes, str]]] = []
    asset_bytes: dict[str, bytes] = {}
    fake_chunks_map: dict[str, list[dict[str, Any]]] = {}

    for asset_name, content_type, fake_chunks in asset_specs:
        file_path = sample_dir / asset_name
        data = file_path.read_bytes()
        files_payload.append(("files", (asset_name, data, content_type)))
        asset_bytes[asset_name] = data
        fake_chunks_map[asset_name] = fake_chunks

    parse_calls: list[str] = []
    upsert_call_counts: dict[str, int] = {}
    captured_chunks: dict[str, list[dict[str, Any]]] = {}

    def fake_parse_and_chunk(filename: str, data: bytes):
        parse_calls.append(filename)
        assert data == asset_bytes[filename]
        return fake_chunks_map[filename]

    def fake_upsert_chunks(chunks: list[dict[str, Any]]):
        assert chunks
        files_in_chunk = {chunk["file"] for chunk in chunks}
        assert len(files_in_chunk) == 1
        file_name = next(iter(files_in_chunk))
        upsert_call_counts[file_name] = upsert_call_counts.get(file_name, 0) + 1
        captured_chunks[file_name] = list(chunks)

    monkeypatch.setattr(service_app, "parse_and_chunk", fake_parse_and_chunk)
    monkeypatch.setattr(service_app, "upsert_chunks", fake_upsert_chunks)

    settings = service_app.get_settings()
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    with TestClient(service_app.app) as client:
        response = client.post(
            "/api/docs/upload",
            data={"user_id": "tester"},
            files=files_payload,
        )

    assert response.status_code == 200
    payload = response.json()

    expected_files = [asset_name for asset_name, *_ in asset_specs]
    assert payload["files"] == expected_files
    assert payload["chunks"] > 0

    assert sorted(parse_calls) == sorted(expected_files)
    for asset_name, *_ in asset_specs:
        assert upsert_call_counts.get(asset_name) == 1
        assert captured_chunks[asset_name] == fake_chunks_map[asset_name]


def test_root_serves_index_html(service_app: Any):
    expected = INDEX_HTML.read_text(encoding="utf-8")

    with TestClient(service_app.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text == expected
    assert "/api/docs/upload" in response.text
    assert "/api/chat" in response.text
    assert "/health" in response.text


def _clone_settings(settings: Any) -> Any:
    return copy.deepcopy(settings)


def test_init_memory_store_toggle(service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _clone_settings(service_app.get_settings())
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "0")
    store = service_app._init_memory_store(settings)
    assert store is None

    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "true")
    monkeypatch.setenv("CHAT_MEMORY_DB_PATH", str(tmp_path / "memory" / "store.sqlite"))
    monkeypatch.setenv("CHAT_MEMORY_TTL_DAYS", "5")
    monkeypatch.setenv("CHAT_SUMMARY_TRIGGER", "7")
    monkeypatch.setenv("CHAT_MEMORY_MAXTOK", "4321")

    enabled_store = service_app._init_memory_store(settings)
    assert isinstance(enabled_store, service_app.MemoryStore)
    assert enabled_store.ttl == 5 * 86400
    assert enabled_store.trigger == 7
    assert enabled_store.max_tokens == 4321


def test_chat_records_memory_when_enabled(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _clone_settings(service_app.get_settings())
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)
    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "1")

    created_instances: list[Any] = []
    record_calls: list[tuple[str, str, str, str]] = []

    class StubMemoryStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            created_instances.append(self)

        def load_context(self, user_id: str, conversation_id: str) -> str:
            return "context"

        def record(
            self, user_id: str, conversation_id: str, message: str, answer: str
        ) -> None:
            record_calls.append((user_id, conversation_id, message, answer))

    monkeypatch.setattr(service_app, "MemoryStore", StubMemoryStore)

    payload = {"user_id": "mem-user", "message": "Привет", "conversation_id": "conv"}

    with TestClient(service_app.app) as client:
        response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    assert created_instances, "Memory store should have been initialised"
    assert record_calls == [
        (payload["user_id"], payload["conversation_id"], payload["message"], "Ответ")
    ]


def test_load_index_html_missing_file_uses_fallback(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(service_app, "WEB_ROOT", tmp_path / "missing")

    fallback = service_app._load_index_html()

    assert fallback == "<h1>Knowledge Base</h1>"


def test_init_memory_store_logs_on_failure(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    settings = _clone_settings(service_app.get_settings())
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)
    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "1")

    class BrokenMemoryStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(service_app, "MemoryStore", BrokenMemoryStore)

    with caplog.at_level(logging.ERROR):
        store = service_app._init_memory_store(settings)

    assert store is None
    assert any("Failed to initialise memory store" in message for message in caplog.messages)


def test_bootstrap_initialises_state(service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _clone_settings(service_app.get_settings())
    settings.log_level = "DEBUG"
    settings.data_dir = tmp_path / "bootstrap-data"
    settings.rate_limit = "42r/m"
    settings.rate_burst = 13
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "1")
    if settings.data_dir.exists():
        shutil.rmtree(settings.data_dir)

    service_app.bootstrap()

    assert logging.getLogger().level == logging.DEBUG
    assert (settings.data_dir / "files").exists()
    assert (settings.data_dir / "db").exists()
    assert hasattr(service_app.app.state, "chat_store")
    assert hasattr(service_app.app.state, "summarizer")
    assert isinstance(service_app.app.state.memory_store, service_app.MemoryStore)
    assert service_app.app.extra["public_host"] == settings.app_host
    assert service_app.app.extra["rate_limit"] == settings.rate_limit
    assert service_app.app.extra["rate_burst"] == settings.rate_burst


def test_normalise_extension_edge_cases(service_app: Any):
    assert service_app._normalise_extension("") == ""
    assert service_app._normalise_extension("no_extension") == ""
    assert service_app._normalise_extension("  Report.PDF  ") == "pdf"


def test_index_chunks(monkeypatch: pytest.MonkeyPatch, service_app: Any):
    calls: list[list[dict[str, Any]]] = []

    def fake_upsert(chunks: list[dict[str, Any]]):
        calls.append(list(chunks))

    monkeypatch.setattr(service_app, "upsert_chunks", fake_upsert)

    empty_result = service_app._index_chunks([])
    assert empty_result == 0
    assert calls == []

    chunks = [{"file": "doc", "content": "chunk"}]
    count = service_app._index_chunks(chunks)
    assert count == 1
    assert calls == [chunks]


def test_upload_document_skips_empty_reads(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _clone_settings(service_app.get_settings())
    settings.data_dir = tmp_path
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    parse_called = False

    def fake_parse(filename: str, data: bytes):
        nonlocal parse_called
        parse_called = True
        return [{"file": filename, "content": "chunk"}]

    monkeypatch.setattr(service_app, "ensure_collection", lambda: None)
    monkeypatch.setattr(service_app, "parse_and_chunk", fake_parse)
    monkeypatch.setattr(service_app, "_save_file", lambda path, data: None)
    monkeypatch.setattr(service_app, "upsert_chunks", lambda chunks: None)

    class EmptyFile:
        filename = "empty.txt"

        async def read(self) -> bytes:  # pragma: no cover - trivial coroutine
            return b""

    async def invoke() -> None:
        await service_app.upload_document(files=[EmptyFile()], user_id="u", conversation_id=None)

    with pytest.raises(service_app.HTTPException) as exc:
        asyncio.run(invoke())

    assert exc.value.status_code == service_app.status.HTTP_400_BAD_REQUEST
    assert exc.value.detail == "NO_TEXT_FOUND"
    assert parse_called is False


def test_upload_document_appends_timestamp_for_existing_files(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _clone_settings(service_app.get_settings())
    settings.data_dir = tmp_path
    settings.retrieve_topk = 3
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    target_dir = settings.files_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_path = target_dir / "report.txt"
    existing_path.write_bytes(b"old")

    monkeypatch.setattr(service_app, "ensure_collection", lambda: None)

    parsed_chunks = [{"file": "report.txt", "page": 1, "content": "chunk"}]
    monkeypatch.setattr(service_app, "parse_and_chunk", lambda name, data: parsed_chunks)

    upsert_calls: list[list[dict[str, Any]]] = []

    def fake_upsert(chunks: list[dict[str, Any]]):
        upsert_calls.append(list(chunks))

    monkeypatch.setattr(service_app, "upsert_chunks", fake_upsert)

    saved_paths: list[Path] = []

    def fake_save(path: Path, data: bytes):
        saved_paths.append(path)

    monkeypatch.setattr(service_app, "_save_file", fake_save)
    monkeypatch.setattr(service_app.time, "time", lambda: 1700000000)

    class NonEmptyFile:
        filename = "report.txt"

        async def read(self) -> bytes:  # pragma: no cover - trivial coroutine
            return b"data"

    async def invoke() -> service_app.UploadResponse:
        return await service_app.upload_document(
            files=[NonEmptyFile()],
            user_id="u",
            conversation_id=None,
        )

    response = asyncio.run(invoke())

    assert response.ok is True
    assert response.files == ["report.txt"]
    assert response.chunks == 1
    assert saved_paths[0].name == "report.txt.1700000000"
    assert saved_paths[0].parent == target_dir
    assert upsert_calls == [parsed_chunks]


def test_format_answer_with_and_without_pages(service_app: Any):
    answer = "Ответ"
    citations = [
        {"file": "doc1.pdf", "page": 2},
        {"file": "doc2.txt"},
    ]
    formatted = service_app._format_answer(answer, citations)
    assert "страница 2" in formatted
    assert "doc2.txt" in formatted

    no_citations = service_app._format_answer(answer, [])
    assert no_citations == "Ответ"


def test_chat_truncates_hits_and_formats_response(
    service_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _clone_settings(service_app.get_settings())
    settings.data_dir = tmp_path / "chat-data"
    settings.retrieve_topk = 5
    settings.rerank_topk = 2
    settings.chat_min_citations = 1
    settings.chat_max_citations = 3
    settings.chat_summary_trigger = 2
    settings.chat_history_limit = 5
    monkeypatch.setattr(service_app, "get_settings", lambda: settings)

    service_app.bootstrap()

    monkeypatch.setattr(service_app, "ensure_model", lambda: None)
    monkeypatch.setattr(service_app, "ensure_collection", lambda: None)

    hits = [
        {"file": f"doc{i}.pdf", "page": i, "score": 1 - i * 0.1}
        for i in range(1, 5)
    ]

    def fake_search(_query: str, top_k: int = 10):
        assert top_k == settings.retrieve_topk
        return list(hits)

    monkeypatch.setattr(service_app, "search_chunks", fake_search)

    recorded_hits: list[dict[str, Any]] = []

    def fake_select(hits_input: list[dict[str, Any]], minimum: int, maximum: int):
        recorded_hits.extend(hits_input)
        return hits_input[:minimum], True

    monkeypatch.setattr(service_app, "select_citations", fake_select)
    monkeypatch.setattr(service_app, "build_context", lambda h, token_limit=3000: "context")
    monkeypatch.setattr(service_app, "generate", lambda prompt: "Готовый ответ")

    payload = service_app.ChatRequest(user_id="u", message="Привет", conversation_id=None)
    response = service_app.chat(payload)

    assert recorded_hits == hits[: settings.rerank_topk]
    assert response.answer.startswith("Готовый ответ")
    assert response.citations == hits[: settings.chat_min_citations]
    assert response.citations_insufficient is False
