"""Tests for the LoRA management API."""

from __future__ import annotations

import asyncio
import math
from decimal import Decimal
from importlib import reload
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

try:  # pragma: no cover - optional dependency for wider validation coverage
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional in test environments
    np = None  # type: ignore[assignment]

from tests.service_stubs import install_service_stubs

install_service_stubs()

from app.api.status_codes import HTTP_UNPROCESSABLE_CONTENT
from app.api.v1 import lora as lora_module
from app.api.v1.lora import load_lora_adapter
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
    monkeypatch.setenv("AUTH_DISABLED_FOR_TESTS", "1")

    install_service_stubs()

    import importlib

    import app.api.v1.lora as lora_module
    import app.core.deps as deps_module
    import app.llm.manager as manager_module
    import app.llm.llama_cpp_provider as provider_module
    importlib.reload(provider_module)
    importlib.reload(manager_module)
    importlib.reload(deps_module)
    importlib.reload(lora_module)

    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    import app.main as app_main

    reload(app_main)
    app = app_main.app
    app.dependency_overrides = {}
    client = TestClient(app)
    stub_llm = SimpleNamespace(
        name="stub-llm",
        ensure_ready=lambda: None,
        ensure_model=lambda: None,
        ensure_adapter=lambda: None,
    )
    app.state.llm_provider = stub_llm
    app.state.llm_client = stub_llm
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
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
    assert ready_after_payload["details"]["lora"]["detail"]["loaded"] is False


def test_load_missing_adapter_returns_not_found(lora_client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "missing.gguf"
    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(missing), "scaling": 1.0},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "ADAPTER_NOT_FOUND"


def test_repeat_load_returns_conflict(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "repeat.gguf")

    first = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert first.status_code == status.HTTP_200_OK

    second = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert second.status_code == status.HTTP_409_CONFLICT
    assert second.json()["detail"] == "ADAPTER_ALREADY_LOADED"


def test_unload_without_adapter_returns_conflict(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "ghost.gguf")
    response = lora_client.post(
        "/api/v1/lora/unload",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["detail"] == "ADAPTER_NOT_LOADED"


def test_load_adapter_accepts_valid_scaling(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "valid.gguf")

    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 0.25},
    )

    assert response.status_code == status.HTTP_200_OK, response.json()


class _BypassManager:
    async def load_adapter(self, *_: object, **__: object) -> None:  # pragma: no cover
        raise AssertionError("load_adapter should not be called when scaling is invalid")


def _construct_lora_payload(path: Path, scaling: object) -> LoraLoadRequest:
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

    assert excinfo.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert excinfo.value.detail == "INVALID_SCALING"


@pytest.mark.parametrize(
    "raw_value",
    [
        pytest.param(-0.5, id="negative-number"),
        pytest.param(0, id="zero-int"),
        pytest.param("nan", id="nan-string"),
        pytest.param("inf", id="inf-string"),
        pytest.param({"kind": "non-numeric"}, id="non-numeric-object"),
    ],
)
def test_load_endpoint_rejects_invalid_scaling_when_bypassed(
    lora_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_value: object,
) -> None:
    adapter_path = _create_adapter(tmp_path, "invalid_guard.gguf")

    def _passthrough_model_validate(
        cls: type[LoraLoadRequest], data: dict[str, object], /, *_: object, **__: object
    ) -> object:
        scaling_raw = data.get("scaling")
        if isinstance(scaling_raw, str):
            if scaling_raw.lower() == "nan":
                scaling_value: object = float("nan")
            elif scaling_raw.lower() == "inf":
                scaling_value = float("inf")
            else:
                scaling_value = float(scaling_raw)
        else:
            scaling_value = scaling_raw
        return SimpleNamespace(path=Path(data["path"]), scaling=scaling_value)

    monkeypatch.setattr(
        lora_module.LoraLoadRequest,
        "model_validate",
        classmethod(_passthrough_model_validate),
    )

    def _identity_scaling(value: object) -> object:
        return value

    monkeypatch.setattr(lora_module, "ensure_valid_scaling", _identity_scaling)

    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": raw_value},
    )

    assert response.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert response.json()["detail"] == "INVALID_SCALING"


def test_load_endpoint_negative_scaling_returns_422_without_manager_call(
    lora_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path = _create_adapter(tmp_path, "negative_integration.gguf")

    async_manager = AsyncMock()
    async_manager.load_adapter = AsyncMock()

    def _construct_negative(
        cls: type[LoraLoadRequest], data: dict[str, object], /, *_: object, **__: object
    ) -> object:
        return _construct_lora_payload(Path(data["path"]), data.get("scaling", -0.5))

    monkeypatch.setattr(
        lora_module.LoraLoadRequest,
        "model_validate",
        classmethod(_construct_negative),
    )

    app = lora_client.app
    dependency = lora_module.get_lora_manager
    app.dependency_overrides[dependency] = lambda: async_manager

    try:
        response = lora_client.post(
            "/api/v1/lora/load",
            json={"path": str(adapter_path), "scaling": -0.5},
        )
    finally:
        app.dependency_overrides.pop(dependency, None)

    assert response.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert response.json()["detail"] == "INVALID_SCALING"
    async_manager.load_adapter.assert_not_called()


def _coerce_scaling_candidate(candidate: object) -> object:
    """Return a concrete scaling value for parametrised test inputs."""

    if isinstance(candidate, tuple) and candidate and candidate[0] == "numpy":
        if np is None:  # pragma: no cover - dependency is optional
            pytest.skip("numpy not available")
        _, value, dtype_name = candidate
        dtype = getattr(np, dtype_name, None)
        if callable(dtype):
            try:
                return dtype(value)
            except TypeError:  # pragma: no cover - fall back to array constructor
                pass
        array_ctor = getattr(np, "array", None)
        if callable(array_ctor):
            try:
                dtype_obj = getattr(np, dtype_name, None)
                array = array_ctor(value, dtype=dtype_obj)
                return array.item() if hasattr(array, "item") else array
            except Exception:  # pragma: no cover - fall back to skip
                pass
        pytest.skip("numpy scalar construction unavailable")

    return candidate


@pytest.mark.parametrize(
    "scaling",
    [
        pytest.param(-1.0, id="negative-float"),
        pytest.param(0.0, id="zero"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(Decimal("-0.5"), id="decimal-negative"),
        pytest.param(Decimal("0"), id="decimal-zero"),
        pytest.param(SimpleNamespace(value=-0.25), id="namespace-value-negative"),
        pytest.param(SimpleNamespace(scaling=0.0), id="namespace-scaling-zero"),
        pytest.param(10.00001, id="exceeds-maximum"),
        pytest.param("invalid", id="non-numeric"),
        pytest.param(("numpy", 0.0, "float32"), id="numpy-zero"),
        pytest.param(("numpy", -0.1, "float64"), id="numpy-negative"),
        pytest.param(
            SimpleNamespace(value=SimpleNamespace(scaling=-1.0)),
            id="nested-namespace",
        ),
    ],
)
def test_scaling_validation_rejects_non_positive(scaling: object) -> None:
    payload = SimpleNamespace(
        path=Path("/tmp/adapter.gguf"), scaling=_coerce_scaling_candidate(scaling)
    )
    manager = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            load_lora_adapter(payload=payload, manager=manager)  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert exc_info.value.detail == "INVALID_SCALING"
    manager.load_adapter.assert_not_called()


def test_scaling_validation_rejects_non_numeric(tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "invalid_str.gguf")

    class DummyManager:
        async def load_adapter(self, *_: object, **__: object) -> None:  # pragma: no cover
            raise AssertionError("load_adapter should not be invoked for invalid scaling")

    payload = SimpleNamespace(path=adapter_path, scaling="not-a-number")

    async def invoke() -> None:
        await load_lora_adapter(payload, DummyManager())

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(invoke())

    assert excinfo.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert excinfo.value.detail == "INVALID_SCALING"


def test_load_endpoint_preserves_original_request_model(tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "immutable.gguf")

    payload = LoraLoadRequest.model_construct(path=adapter_path, scaling=Decimal("0.75"))

    status_payload = SimpleNamespace(
        loaded=True,
        path=adapter_path,
        scaling=0.75,
        adapter_name="immutable",
    )

    manager = SimpleNamespace(load_adapter=AsyncMock(return_value=status_payload))

    response = asyncio.run(load_lora_adapter(payload, manager=manager))

    assert isinstance(response, lora_module.LoraStatusResponse)
    manager.load_adapter.assert_awaited_once_with(adapter_path, 0.75)
    assert payload.scaling == pytest.approx(0.75)
    assert isinstance(payload.scaling, float)
