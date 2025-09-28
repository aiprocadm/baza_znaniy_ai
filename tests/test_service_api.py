"""Integration tests for the knowledge base service API."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

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

        assert response.status_code == 400
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

def test_root_serves_index_html(service_app: Any):
    expected = INDEX_HTML.read_text(encoding="utf-8")

    with TestClient(service_app.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.text == expected
    assert "/api/docs/upload" in response.text
    assert "/api/chat" in response.text
    assert "/health" in response.text
