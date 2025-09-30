"""Unit tests for the Ollama client used by the service app."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb" / "app"


def _load_ollama_client_module():
    """Load the service Ollama client module under a temporary package name."""

    package_name = "kb_service_app"

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    config_module = types.ModuleType(f"{package_name}.config")

    def _stub_get_settings():
        raise RuntimeError("get_settings stub must be patched in tests")

    config_module.get_settings = _stub_get_settings  # type: ignore[attr-defined]
    sys.modules[config_module.__name__] = config_module

    module_spec = importlib.util.spec_from_file_location(
        f"{package_name}.ollama_client", SERVICE_ROOT / "ollama_client.py"
    )
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ollama_client_module():
    module = _load_ollama_client_module()
    try:
        yield module
    finally:
        package_name = "kb_service_app"
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)


class DummyResponse:
    def __init__(self, data: dict | None = None):
        self._data = data or {}

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        return None


def _patch_settings(monkeypatch: pytest.MonkeyPatch, module, base_url: str, model: str):
    settings = SimpleNamespace(
        ollama_base_url=base_url,
        gen_model=model,
        llm_model_name=model,
        llm_model=model,
        ollama_model=model,
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


def test_ensure_model_caches_after_success(monkeypatch: pytest.MonkeyPatch, ollama_client_module):
    module = ollama_client_module
    module._ENSURED_MODEL = False
    settings = _patch_settings(monkeypatch, module, "http://ollama.local", "test-model")

    calls: list[tuple] = []

    class DummyClient:
        def __init__(self, *, timeout):
            self.timeout = timeout
            calls.append(("init", timeout))

        def __enter__(self):
            calls.append(("enter", self.timeout))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", self.timeout))
            return False

        def get(self, url: str):
            calls.append(("get", url))
            return DummyResponse({"models": [{"name": settings.gen_model}]})

        def post(self, url: str, json: dict):
            calls.append(("post", url, json))
            return DummyResponse()

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    module.ensure_model()

    assert module._ENSURED_MODEL is True
    assert any(call[0] == "get" for call in calls)
    assert not any(call[0] == "post" for call in calls)

    snapshot = list(calls)
    module.ensure_model()
    assert calls == snapshot


def test_ensure_model_pulls_when_missing(monkeypatch: pytest.MonkeyPatch, ollama_client_module):
    module = ollama_client_module
    module._ENSURED_MODEL = False
    settings = _patch_settings(monkeypatch, module, "http://ollama.internal/", "missing-model")

    clients: list = []

    class DummyClient:
        def __init__(self, *, timeout):
            self.timeout = timeout
            self.requests: list[tuple] = []
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str):
            self.requests.append(("get", url))
            return DummyResponse({"models": []})

        def post(self, url: str, json: dict):
            self.requests.append(("post", url, json))
            return DummyResponse()

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    module.ensure_model()

    assert module._ENSURED_MODEL is True
    assert len(clients) == 2

    base_url = settings.ollama_base_url.rstrip("/")
    tags_client, pull_client = clients

    assert tags_client.timeout == 60
    assert tags_client.requests == [("get", f"{base_url}/api/tags")]

    assert pull_client.timeout is None
    assert pull_client.requests == [
        ("post", f"{base_url}/api/pull", {"name": settings.gen_model})
    ]


def test_generate_sends_prompt_and_returns_response(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    settings = _patch_settings(monkeypatch, module, "http://ollama.service", "gen-model")

    payloads: list[tuple[str, dict]] = []

    class DummyClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, json: dict):
            payloads.append((url, json))
            return DummyResponse({"response": "text"})

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    result = module.generate("Привет")

    assert result == "text"
    assert payloads == [
        (
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            {"model": settings.gen_model, "prompt": "Привет", "stream": False},
        )
    ]


def test_generate_propagates_httpx_errors(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    _patch_settings(monkeypatch, module, "http://ollama", "broken")

    class ErrorClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, json: dict):
            request = httpx.Request("POST", url)
            raise httpx.RequestError("boom", request=request)

    monkeypatch.setattr(module.httpx, "Client", ErrorClient)

    with pytest.raises(httpx.RequestError):
        module.generate("fail")
