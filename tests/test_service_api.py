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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb" / "app"


def _load_service_app(tmp_path: Path) -> Any:
    package_name = "kb_service_app"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

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
