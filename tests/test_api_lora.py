"""Tests for the LoRA management API."""

from __future__ import annotations

from importlib import reload
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.service_stubs import install_service_stubs


@pytest.fixture()
def lora_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide a test client with llama.cpp stubs configured."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    base_model = tmp_path / "base.gguf"
    base_model.write_bytes(b"model")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path / 'ingest.db'}")
    monkeypatch.setenv("LLM_MODEL_NAME", str(base_model))
    monkeypatch.setenv("LLM_MODEL_PATH", str(base_model))
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")

    install_service_stubs()

    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    import app.main as app_main

    reload(app_main)
    app = app_main.app
    app.dependency_overrides = {}

    client = TestClient(app)
    try:
        yield client
    finally:
        close = getattr(client, 'close', None)
        if callable(close):
            close()
        config_module.get_settings.cache_clear()


def _create_adapter(tmp_path: Path, name: str = "adapter.gguf") -> Path:
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(exist_ok=True)
    adapter_path = adapter_dir / name
    adapter_path.write_bytes(b"adapter")
    return adapter_path


def test_load_and_unload_adapter_updates_ready(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path)

    load_response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 0.75},
    )
    assert load_response.status_code == 200, load_response.json()
    load_payload = load_response.json()
    assert load_payload["loaded"] is True
    assert load_payload["path"] == str(adapter_path.resolve())
    assert load_payload["scaling"] == pytest.approx(0.75)

    ready_response = lora_client.get("/ready")
    assert ready_response.status_code == 200
    ready_payload = ready_response.json()
    lora_details = ready_payload["details"]["lora"]
    assert lora_details["status"] == "ok"
    assert lora_details["detail"]["loaded"] is True

    unload_response = lora_client.post(
        "/api/v1/lora/unload",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert unload_response.status_code == 200
    unload_payload = unload_response.json()
    assert unload_payload["loaded"] is False

    ready_after = lora_client.get("/ready")
    assert ready_after.status_code == 200
    ready_after_payload = ready_after.json()
    assert (
        ready_after_payload["details"]["lora"]["detail"]["loaded"] is False
    )


def test_load_missing_adapter_returns_not_found(lora_client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "missing.gguf"
    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(missing), "scaling": 1.0},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "ADAPTER_NOT_FOUND"


def test_repeat_load_returns_conflict(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "repeat.gguf")

    first = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert first.status_code == 200

    second = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert second.status_code == 409
    assert second.json()["detail"] == "ADAPTER_ALREADY_LOADED"


def test_unload_without_adapter_returns_conflict(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "ghost.gguf")
    response = lora_client.post(
        "/api/v1/lora/unload",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "ADAPTER_NOT_LOADED"


def test_scaling_validation_rejects_non_positive(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "invalid.gguf")
    try:
        response = lora_client.post(
            "/api/v1/lora/load",
            json={"path": str(adapter_path), "scaling": -0.5},
        )
    except ValidationError:
        return
    assert response.status_code == 422


def test_scaling_validation_rejects_non_finite(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "nan.gguf")
    payload = {"path": str(adapter_path), "scaling": float("nan")}
    try:
        response = lora_client.post(
            "/api/v1/lora/load",
            json=payload,
        )
    except ValidationError:
        return
    assert response.status_code == 422
