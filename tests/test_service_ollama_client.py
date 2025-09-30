"""Tests for the service Ollama client module."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb" / "app"
PACKAGE_NAME = "kb_service_app"
MODULE_NAME = f"{PACKAGE_NAME}.ollama_client"


class DummyResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def _load_ollama_client_module():
    for name in list(sys.modules):
        if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
            sys.modules.pop(name, None)

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = package

    config_module = types.ModuleType(f"{PACKAGE_NAME}.config")

    def _unpatched_settings():
        raise RuntimeError("tests must patch get_settings")

    config_module.get_settings = _unpatched_settings  # type: ignore[attr-defined]
    sys.modules[config_module.__name__] = config_module

    spec = importlib.util.spec_from_file_location(MODULE_NAME, SERVICE_ROOT / "ollama_client.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ollama_client_module():
    module = _load_ollama_client_module()
    try:
        yield module
    finally:
        for name in list(sys.modules):
            if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
                sys.modules.pop(name, None)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, module, base_url: str, model: str):
    settings = SimpleNamespace(ollama_base_url=base_url, gen_model=model)
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


def test_ensure_model_returns_early_when_cached(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    module._ENSURED_MODEL = True
    _patch_settings(monkeypatch, module, "http://ollama", "model")

    def forbidden_client(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("httpx.Client must not be constructed when model already ensured")

    monkeypatch.setattr(module.httpx, "Client", forbidden_client)

    module.ensure_model()


def test_ensure_model_reads_tags_only_when_model_exists(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    module._ENSURED_MODEL = False
    settings = _patch_settings(monkeypatch, module, "http://ollama.local", "ready-model")
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

        def post(self, url: str, json: dict):  # pragma: no cover - should not be hit
            raise AssertionError("pull should not be triggered when model exists")

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    module.ensure_model()

    assert module._ENSURED_MODEL is True
    base_url = settings.ollama_base_url.rstrip("/")
    assert calls == [
        ("init", 60),
        ("enter", 60),
        ("get", f"{base_url}/api/tags"),
        ("exit", 60),
    ]


def test_ensure_model_does_not_recreate_client_once_cached(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    module._ENSURED_MODEL = False
    settings = _patch_settings(monkeypatch, module, "http://ollama", "ready-model")
    instantiations = 0

    class DummyClient:
        def __init__(self, *, timeout):
            nonlocal instantiations
            instantiations += 1
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str):
            return DummyResponse({"models": [{"name": settings.gen_model}]})

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    module.ensure_model()

    assert module._ENSURED_MODEL is True
    assert instantiations == 1

    module.ensure_model()

    assert instantiations == 1


def test_ensure_model_pulls_when_missing(monkeypatch: pytest.MonkeyPatch, ollama_client_module):
    module = ollama_client_module
    module._ENSURED_MODEL = False
    settings = _patch_settings(monkeypatch, module, "http://ollama.internal/", "missing-model")
    clients: list["DummyClient"] = []

    class DummyClient:
        def __init__(self, *, timeout):
            self.timeout = timeout
            self.calls: list[tuple] = []
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str):
            self.calls.append(("get", url))
            return DummyResponse({"models": []})

        def post(self, url: str, json: dict):
            self.calls.append(("post", url, json))
            return DummyResponse()

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    module.ensure_model()

    assert module._ENSURED_MODEL is True
    assert len(clients) == 2
    base_url = settings.ollama_base_url.rstrip("/")

    first, second = clients
    assert first.timeout == 60
    assert first.calls == [("get", f"{base_url}/api/tags")]
    assert second.timeout is None
    assert second.calls == [
        ("post", f"{base_url}/api/pull", {"name": settings.gen_model})
    ]


def test_generate_returns_text_and_bubbles_errors(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    _patch_settings(monkeypatch, module, "http://ollama.service", "gen-model")
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
            if json["prompt"] == "boom":
                raise ValueError("boom")
            return DummyResponse({"response": "text"})

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    result = module.generate("Привет")

    assert result == "text"
    assert payloads[0] == (
        "http://ollama.service/api/generate",
        {"model": "gen-model", "prompt": "Привет", "stream": False},
    )

    with pytest.raises(ValueError, match="boom"):
        module.generate("boom")


def test_generate_returns_default_when_missing_response(
    monkeypatch: pytest.MonkeyPatch, ollama_client_module
):
    module = ollama_client_module
    module._ENSURED_MODEL = False
    _patch_settings(monkeypatch, module, "http://ollama.service", "gen-model")

    class DummyClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, json: dict):
            return DummyResponse({})

    monkeypatch.setattr(module.httpx, "Client", DummyClient)

    assert module.generate("hello") == ""
