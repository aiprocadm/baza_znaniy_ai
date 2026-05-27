from __future__ import annotations

import asyncio
import json
import math
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock

try:  # Optional dependency used in parametrised tests
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional
    np = None  # type: ignore[assignment]

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from app.api.status_codes import HTTP_UNPROCESSABLE_CONTENT
from app.api.v1 import lora as lora_module
import app.api.routes_lora as routes_lora_module
from app.core import config as config_module
from app.llm import cache as cache_module
from app.llm import lora_runtime
import app.services.lora_manager as lora_manager_service
from app.llm.lora_runtime import AdapterInfo
from app.models.lora import (
    LoraAdapterNamePayload,
    LoraLoadRequest,
    LoraStatusResponse,
    LoraUnloadRequest,
)
from pydantic import ValidationError


class StubProvider:
    def __init__(self) -> None:
        self.loaded: Path | None = None
        self.scaling: float | None = None
        self.unloaded = False
        self.model_checked = False
        self.ready_checked = False

    def ensure_model(self) -> None:
        self.model_checked = True

    def ensure_ready(self) -> None:
        self.ready_checked = True

    def load_lora(self, path: Path, *, scaling: float | None = None) -> None:
        self.loaded = Path(path)
        if scaling is not None:
            self.scaling = float(scaling)

    def unload_lora(self) -> None:
        self.unloaded = True
        self.loaded = None

    def generate(self, prompt: str, *, context: dict | None = None) -> str:
        return "ok"


@pytest.fixture()
def lora_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    registry = tmp_path / "registry"
    adapter_dir = registry / "demo"
    adapter_dir.mkdir(parents=True)
    (tmp_path / "model.gguf").write_bytes(b"gguf")
    adapter_path = adapter_dir / "adapter.gguf"
    adapter_path.write_bytes(b"adapter")
    manifest = {
        "name": "demo",
        "base": "meta-llama/Llama-3-8b-Instruct",
        "type": "gguf",
        "seq_len": 4096,
        "created_at": "2024-01-01T00:00:00Z",
    }
    (adapter_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    provider = StubProvider()

    monkeypatch.setenv("LORA_REGISTRY_DIR", str(registry))
    monkeypatch.setenv("LORA_DEFAULT_ADAPTER", "none")
    monkeypatch.setenv("USE_LORA", "1")
    monkeypatch.setenv("LLM_MODEL_NAME", manifest["base"])
    monkeypatch.setenv("LLM_MODEL_PATH", str(tmp_path / "model.gguf"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path / 'db.sqlite'}")
    monkeypatch.setenv("AUTH_DISABLED_FOR_TESTS", "1")
    monkeypatch.setenv("RERANK_ENABLED", "0")

    monkeypatch.setattr(cache_module, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_manager_service, "get_cached_provider", lambda settings=None: provider)

    config_module.get_settings.cache_clear()

    from importlib import reload

    import app.main as app_main

    reload(app_main)
    client = TestClient(app_main.app)
    try:
        yield client
    finally:
        client.close()
        lora_runtime.set_active_adapter(None)
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
    assert load_response.status_code == status.HTTP_200_OK, load_response.text
    load_payload = load_response.json()
    assert load_payload["loaded"] is True
    assert load_payload["adapter"]["path"] == str(adapter_path.resolve())
    assert load_payload["adapter"]["scaling"] == pytest.approx(0.75)

    ready_response = lora_client.get("/ready")
    assert ready_response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    ready_payload = ready_response.json()
    lora_details = ready_payload["details"]["lora"]
    assert lora_details["status"] == "ok"
    assert lora_details["loaded"] is True
    assert lora_details["detail"]["adapter"]["path"] == str(adapter_path.resolve())

    unload_response = lora_client.post(
        "/api/v1/lora/unload",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert unload_response.status_code == status.HTTP_200_OK
    unload_payload = unload_response.json()
    assert unload_payload["loaded"] is False

    ready_after = lora_client.get("/ready")
    assert ready_after.status_code in {
        status.HTTP_200_OK,
        status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    ready_after_payload = ready_after.json()
    assert ready_after_payload["details"]["lora"]["loaded"] is False


def test_load_missing_adapter_returns_not_found(lora_client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "missing.gguf"
    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(missing), "scaling": 1.0},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    payload = response.json()
    assert payload["message"] == "ADAPTER_NOT_FOUND"
    assert payload["status"] == status.HTTP_404_NOT_FOUND


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
    payload = second.json()
    assert payload["message"] == "ADAPTER_ALREADY_LOADED"
    assert payload["status"] == status.HTTP_409_CONFLICT


def test_unload_without_adapter_returns_conflict(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "ghost.gguf")
    response = lora_client.post(
        "/api/v1/lora/unload",
        json={"path": str(adapter_path), "scaling": 1.0},
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    payload = response.json()
    assert payload["message"] == "ADAPTER_NOT_LOADED"
    assert payload["status"] == status.HTTP_409_CONFLICT


def test_load_adapter_accepts_valid_scaling(lora_client: TestClient, tmp_path: Path) -> None:
    adapter_path = _create_adapter(tmp_path, "valid.gguf")

    response = lora_client.post(
        "/api/v1/lora/load",
        json={"path": str(adapter_path), "scaling": 0.25},
    )

    assert response.status_code == status.HTTP_200_OK, response.text


class _BypassManager:
    async def load_adapter(self, *_: object, **__: object) -> None:  # pragma: no cover
        raise AssertionError("load_adapter should not be called when scaling is invalid")


def _construct_lora_payload(path: Path, scaling: object) -> LoraLoadRequest:
    payload = object.__new__(LoraLoadRequest)
    payload.__dict__ = {"path": path, "scaling": scaling}
    payload.__pydantic_fields_set__ = {"path", "scaling"}
    return payload


@pytest.mark.parametrize(
    "bad_scaling",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(0.0, id="zero"),
        pytest.param(math.inf, id="infinite"),
        pytest.param(math.nan, id="nan"),
        pytest.param("nan", id="nan-string"),
        pytest.param("inf", id="inf-string"),
        pytest.param({"kind": "invalid"}, id="non-numeric"),
    ],
)
def test_runtime_scaling_guard_rejects_invalid_values(bad_scaling: object) -> None:
    payload = _construct_lora_payload(Path("/tmp/ghost.gguf"), bad_scaling)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(lora_module.load_lora_adapter(payload, manager=_BypassManager()))

    assert excinfo.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert excinfo.value.detail == "INVALID_SCALING"


def _coerce_scaling_candidate(candidate: object) -> object:
    if isinstance(candidate, tuple) and candidate and candidate[0] == "numpy":
        if np is None:
            pytest.skip("numpy not available")
        _, value, dtype_name = candidate
        dtype = getattr(np, dtype_name, None)
        if callable(dtype):
            try:
                return dtype(value)
            except TypeError:
                pass
        array_ctor = getattr(np, "array", None)
        if callable(array_ctor):
            try:
                dtype_obj = getattr(np, dtype_name, None)
                array = array_ctor(value, dtype=dtype_obj)
                return array.item() if hasattr(array, "item") else array
            except Exception:
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
            lora_module.load_lora_adapter(payload=payload, manager=manager)  # type: ignore[arg-type]
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
        await lora_module.load_lora_adapter(payload, DummyManager())

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(invoke())

    assert excinfo.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert excinfo.value.detail == "INVALID_SCALING"


def test_load_endpoint_maps_manager_errors(tmp_path: Path) -> None:
    payload = _construct_lora_payload(tmp_path / "adapter.gguf", 1.0)

    class RaisingManager:
        def __init__(self, exc: Exception) -> None:
            self._exc = exc

        async def load_adapter(self, *_: object, **__: object) -> None:
            raise self._exc

    with pytest.raises(HTTPException) as invalid_exc:
        asyncio.run(
            lora_module.load_lora_adapter(
                payload, manager=RaisingManager(lora_module.InvalidScalingError("bad"))
            )
        )
    assert invalid_exc.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert invalid_exc.value.detail == "INVALID_SCALING"

    with pytest.raises(HTTPException) as unsupported_exc:
        asyncio.run(
            lora_module.load_lora_adapter(
                payload,
                manager=RaisingManager(lora_module.UnsupportedAdapterFormatError("unsupported")),
            )
        )
    assert unsupported_exc.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert unsupported_exc.value.detail == "unsupported"

    with pytest.raises(HTTPException) as compatibility_exc:
        asyncio.run(
            lora_module.load_lora_adapter(
                payload,
                manager=RaisingManager(lora_runtime.AdapterCompatibilityError("no support")),
            )
        )
    assert compatibility_exc.value.status_code == HTTP_UNPROCESSABLE_CONTENT
    assert compatibility_exc.value.detail == "no support"


def test_load_and_unload_cycle(lora_client: TestClient) -> None:
    load = lora_client.post("/admin/lora/load", json={"name": "demo"})
    assert load.status_code == status.HTTP_200_OK, load.text
    payload = load.json()
    assert payload["loaded"] is True
    assert payload["adapter"]["name"] == "demo"
    assert payload["adapter"]["scaling"] == pytest.approx(1.0)

    ready = lora_client.get("/ready")
    ready_payload = ready.json()["details"]["lora"]
    assert ready_payload["status"] == "ok"
    assert ready_payload["detail"]["loaded"] is True

    unload = lora_client.post("/admin/lora/unload", json={"name": "demo"})
    assert unload.status_code == status.HTTP_200_OK
    assert unload.json()["loaded"] is False


def test_admin_routes_error_mapping(
    monkeypatch: pytest.MonkeyPatch, lora_client: TestClient
) -> None:
    def _fake_list() -> list[AdapterInfo]:
        return [
            AdapterInfo.from_path(
                Path("/tmp/demo.gguf"),
                base="meta-llama/Llama-3-8b-Instruct",
                scaling=0.5,
            )
        ]

    monkeypatch.setattr(routes_lora_module, "list_adapters", _fake_list)
    response = lora_client.get("/admin/lora/list")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload[0]["scaling"] == pytest.approx(0.5)

    monkeypatch.setattr(routes_lora_module, "active_adapter", lambda: None)
    status_response = lora_client.get("/admin/lora/status")
    assert status_response.status_code == status.HTTP_200_OK
    assert status_response.json()["loaded"] is False

    monkeypatch.setattr(
        routes_lora_module,
        "load_adapter",
        lambda name: (_ for _ in ()).throw(lora_runtime.RegistryError("boom")),
    )
    error_response = lora_client.post("/admin/lora/load", json={"name": "missing"})
    assert error_response.status_code == status.HTTP_404_NOT_FOUND
    payload = error_response.json()
    assert payload["message"] == "boom"
    assert payload["status"] == status.HTTP_404_NOT_FOUND

    monkeypatch.setattr(
        routes_lora_module,
        "load_adapter",
        lambda name: (_ for _ in ()).throw(lora_runtime.AdapterCompatibilityError("bad")),
    )
    conflict = lora_client.post("/admin/lora/load", json={"name": "bad"})
    assert conflict.status_code == HTTP_UNPROCESSABLE_CONTENT
    payload = conflict.json()
    assert payload["message"] == "bad"
    assert payload["status"] == HTTP_UNPROCESSABLE_CONTENT

    monkeypatch.setattr(
        routes_lora_module,
        "load_adapter",
        lambda name: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    generic = lora_client.post("/admin/lora/load", json={"name": "broken"})
    assert generic.status_code == status.HTTP_400_BAD_REQUEST
    payload = generic.json()
    assert payload["message"] == "broken"
    assert payload["status"] == status.HTTP_400_BAD_REQUEST

    monkeypatch.setattr(
        routes_lora_module,
        "unload_adapter",
        lambda name=None: (_ for _ in ()).throw(lora_runtime.RegistryError("oops")),
    )
    unload_error = lora_client.post("/admin/lora/unload", json={"name": "demo"})
    assert unload_error.status_code == status.HTTP_400_BAD_REQUEST
    payload = unload_error.json()
    assert payload["message"] == "oops"
    assert payload["status"] == status.HTTP_400_BAD_REQUEST


def test_lora_runtime_manager_handles_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = StubProvider()
    monkeypatch.setattr(cache_module, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_manager_service, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Llama-3-8b-Instruct")
    config_module.get_settings.cache_clear()
    manager = lora_module.get_lora_manager()
    assert manager.settings.llm_model_name == "meta-llama/Llama-3-8b-Instruct"

    adapter_path = _create_adapter(tmp_path, "manager.gguf").resolve()

    snapshot = asyncio.run(manager.load_adapter(adapter_path, 0.5))
    assert snapshot.info is not None
    assert snapshot.info.payload == adapter_path
    assert snapshot.info.scaling == pytest.approx(0.5)
    assert provider.loaded == adapter_path
    assert provider.scaling == pytest.approx(0.5)

    status_payload = LoraStatusResponse.from_runtime(snapshot.info)
    assert status_payload.adapter is not None
    assert status_payload.adapter.scaling == pytest.approx(0.5)

    asyncio.run(manager.unload_adapter(adapter_path))
    assert provider.unloaded is True
    assert lora_runtime.active_adapter() is None
    config_module.get_settings.cache_clear()


def test_lora_runtime_manager_rejects_invalid_scaling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = StubProvider()
    monkeypatch.setattr(cache_module, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_manager_service, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Llama-3-8b-Instruct")
    config_module.get_settings.cache_clear()
    manager = lora_module.get_lora_manager()
    adapter_path = _create_adapter(tmp_path, "invalid.gguf").resolve()

    with pytest.raises(lora_module.InvalidScalingError):
        asyncio.run(manager.load_adapter(adapter_path, -0.1))

    with pytest.raises(lora_module.InvalidScalingError):
        asyncio.run(manager.load_adapter(adapter_path, float("inf")))
    config_module.get_settings.cache_clear()


def test_lora_runtime_manager_handles_peft_and_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class PeftProvider(StubProvider):
        def __init__(self) -> None:
            super().__init__()
            self.peft_loaded: tuple[Path, float] | None = None

        def load_peft_adapter(self, path: Path, *, scaling: float) -> None:
            self.peft_loaded = (Path(path), float(scaling))

    provider = PeftProvider()
    monkeypatch.setattr(lora_manager_service, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Llama-3-8b-Instruct")
    config_module.get_settings.cache_clear()
    manager = lora_module.get_lora_manager()

    peft_path = _create_adapter(tmp_path, "adapter.safetensors").resolve()
    snapshot = asyncio.run(manager.load_adapter(peft_path, 0.8))
    assert provider.peft_loaded == (peft_path, pytest.approx(0.8))
    assert snapshot.info is not None and snapshot.info.scaling == pytest.approx(0.8)

    with pytest.raises(lora_manager_service.UnsupportedAdapterFormatError):
        asyncio.run(manager.load_adapter(_create_adapter(tmp_path, "adapter.txt"), 0.5))

    asyncio.run(manager.unload_adapter(peft_path))
    config_module.get_settings.cache_clear()


def test_lora_runtime_manager_compatibility_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class NoLoraProvider:
        def ensure_model(self) -> None:  # pragma: no cover - simple stub
            return None

        def ensure_ready(self) -> None:  # pragma: no cover - simple stub
            return None

    class NoPeftProvider(StubProvider):
        def load_lora(
            self, path: Path, *, scaling: float | None = None
        ) -> None:  # pragma: no cover
            raise AssertionError("Should not be called")

    monkeypatch.setattr(
        lora_manager_service, "get_cached_provider", lambda settings=None: NoLoraProvider()
    )
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: NoLoraProvider())
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Llama-3-8b-Instruct")
    config_module.get_settings.cache_clear()
    manager = lora_module.get_lora_manager()
    gguf_path = _create_adapter(tmp_path, "missing.gguf").resolve()
    with pytest.raises(lora_runtime.AdapterCompatibilityError):
        asyncio.run(manager.load_adapter(gguf_path, 0.5))

    monkeypatch.setattr(
        lora_manager_service, "get_cached_provider", lambda settings=None: NoPeftProvider()
    )
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: NoPeftProvider())
    config_module.get_settings.cache_clear()
    manager = lora_module.get_lora_manager()
    peft_path = _create_adapter(tmp_path, "adapter.safetensors").resolve()
    with pytest.raises(lora_runtime.AdapterCompatibilityError):
        asyncio.run(manager.load_adapter(peft_path, 0.5))
    config_module.get_settings.cache_clear()


def test_lora_runtime_manager_unload_validations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = StubProvider()
    monkeypatch.setattr(lora_manager_service, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setenv("LLM_MODEL_NAME", "meta-llama/Llama-3-8b-Instruct")
    config_module.get_settings.cache_clear()
    manager = lora_module.get_lora_manager()

    adapter_path = _create_adapter(tmp_path, "active.gguf").resolve()
    asyncio.run(manager.load_adapter(adapter_path, 0.6))

    wrong_path = _create_adapter(tmp_path, "other.gguf").resolve()
    with pytest.raises(lora_manager_service.AdapterNotLoadedError):
        asyncio.run(manager.unload_adapter(wrong_path))

    asyncio.run(manager.unload_adapter(adapter_path))
    config_module.get_settings.cache_clear()


def test_lora_models_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        LoraLoadRequest(path=None, scaling=1.0)

    with pytest.raises(ValidationError):
        LoraLoadRequest(path=tmp_path / "adapter.gguf", scaling="nan")

    with pytest.raises(ValidationError):
        LoraLoadRequest(path=tmp_path / "adapter.gguf", scaling=0.0)

    with pytest.raises(ValidationError):
        LoraLoadRequest(path=tmp_path / "adapter.gguf", scaling=11.0)

    payload = LoraAdapterNamePayload(name="demo")
    assert payload.name == "demo"

    unload_request = LoraUnloadRequest(path=None, extra="ignored")
    assert unload_request.path is None

    with pytest.raises(ValueError):
        LoraLoadRequest._validate_path(None)

    for invalid_scaling in ("nan", 0.0, 11.0, {"bad": "value"}):
        with pytest.raises(ValueError):
            LoraLoadRequest._validate_scaling(invalid_scaling)  # type: ignore[arg-type]
