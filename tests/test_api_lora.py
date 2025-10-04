"""Tests for the LoRA management API."""

from __future__ import annotations

import asyncio
from importlib import reload
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.v1.lora import load_lora_adapter

from tests.service_stubs import install_service_stubs

install_service_stubs()

from app.api.v1 import lora as lora_module
from app.models.lora import LoraLoadRequest


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
    assert ready_response.status_code == 200, ready_response.json()
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


@pytest.mark.parametrize("invalid_scaling", [-0.5, 0.0, 10.5, float("nan")])
def test_scaling_validation_rejects_non_positive(
    tmp_path: Path, invalid_scaling: float
) -> None:
    from app.api.v1.lora import HTTP_UNPROCESSABLE_ENTITY, load_lora_adapter

    adapter_path = _create_adapter(tmp_path, "invalid.gguf")

    class DummyManager:
        async def load_adapter(self, *_: object, **__: object) -> None:  # pragma: no cover
            raise AssertionError("load_adapter should not be invoked for invalid scaling")

    payload = SimpleNamespace(path=adapter_path, scaling=invalid_scaling)

    async def invoke() -> None:
        await load_lora_adapter(payload, DummyManager())

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(invoke())

    assert excinfo.value.status_code == HTTP_UNPROCESSABLE_ENTITY
    assert excinfo.value.detail == "INVALID_SCALING"


def test_scaling_validation_rejects_non_numeric(tmp_path: Path) -> None:
    from app.api.v1.lora import HTTP_UNPROCESSABLE_ENTITY, load_lora_adapter

    adapter_path = _create_adapter(tmp_path, "invalid_str.gguf")

    class DummyManager:
        async def load_adapter(self, *_: object, **__: object) -> None:  # pragma: no cover
            raise AssertionError("load_adapter should not be invoked for invalid scaling")

    payload = SimpleNamespace(path=adapter_path, scaling="not-a-number")

    async def invoke() -> None:
        await load_lora_adapter(payload, DummyManager())

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(invoke())

    assert excinfo.value.status_code == HTTP_UNPROCESSABLE_ENTITY
    assert excinfo.value.detail == "INVALID_SCALING"



class _BypassManager:
    async def load_adapter(self, path: Path, scaling: float) -> None:  # pragma: no cover - should not run
        raise AssertionError("load_adapter should not be called when scaling is invalid")


def _construct_lora_payload(path: Path, scaling: float) -> LoraLoadRequest:
    """Instantiate ``LoraLoadRequest`` without running validators."""

    constructor = getattr(LoraLoadRequest, "model_construct", None)
    if callable(constructor):  # pragma: no cover - exercised when real pydantic v2 installed
        return constructor(path=path, scaling=scaling)

    payload = object.__new__(LoraLoadRequest)
    payload.__dict__ = {"path": path, "scaling": scaling}
    return payload


@pytest.mark.parametrize(
    "bad_scaling",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(0.0, id="zero"),
        pytest.param(math.inf, id="infinite"),
        pytest.param(math.nan, id="nan"),
    ],
)
def test_runtime_scaling_guard_rejects_invalid_values(bad_scaling: float) -> None:
    payload = _construct_lora_payload(Path("/tmp/ghost.gguf"), bad_scaling)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(lora_module.load_lora_adapter(payload, manager=_BypassManager()))

    expected_status = getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)
    assert excinfo.value.status_code == expected_status


@pytest.mark.parametrize(
    "invalid_scaling",
    (-0.1, 0.0, float("inf"), float("nan")),
)
def test_load_endpoint_rejects_invalid_scaling_when_bypassed(
    invalid_scaling: float, tmp_path: Path
) -> None:
    from app.api.v1 import lora as lora_module

    dummy_manager = SimpleNamespace(load_adapter=AsyncMock())
    payload = SimpleNamespace(path=tmp_path / "adapter.gguf", scaling=invalid_scaling)

    async def _invoke() -> None:
        await lora_module.load_lora_adapter(payload, dummy_manager)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_invoke())

    assert exc.value.status_code == 422
    dummy_manager.load_adapter.assert_not_awaited()

    assert response.json()["detail"] == "INVALID_SCALING"


def test_load_adapter_accepts_valid_scaling(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "valid.gguf")

    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 0.25},
    )

    assert response.status_code == 200, response.json()



@pytest.mark.parametrize(
    "scaling",
    [float("nan"), float("inf"), 0.0, -1.0, 10.00001, "invalid"],
)
def test_scaling_validation_rejects_non_positive(scaling: object) -> None:
    payload = SimpleNamespace(path="/tmp/adapter.gguf", scaling=scaling)
    manager = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            load_lora_adapter(payload=payload, manager=manager)  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "INVALID_SCALING"
    manager.load_adapter.assert_not_called()
