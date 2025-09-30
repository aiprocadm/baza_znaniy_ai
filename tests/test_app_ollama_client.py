"""Tests for :class:`app.llm.providers.OllamaProvider`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from app.core.config import Settings
from app.llm.providers import OllamaProvider


class DummyResponse:
    """Minimal httpx response stub."""

    def __init__(self, payload: dict[str, Any] | None = None):
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - matches httpx API
        return None


class DummyClient:
    """Context manager mimicking :class:`httpx.Client`."""

    def __init__(self, *, timeout: Any, responses: Iterator[DummyResponse], calls: list):
        self.timeout = timeout
        self._responses = responses
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str) -> DummyResponse:
        self.calls.append(("get", self.timeout, url))
        return next(self._responses)

    def post(self, url: str, json: dict[str, Any]) -> DummyResponse:
        self.calls.append(("post", self.timeout, url, json))
        return next(self._responses)


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        ollama_base_url="http://ollama:11434",
        llm_model_name="test-model",
        max_context_tokens=1024,
        max_generation_tokens=256,
    )


def test_ensure_model_skips_pull_when_model_present(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls: list = []

    def client_stub(*, timeout):
        responses = iter([DummyResponse({"models": [{"name": settings.llm_model_name}]})])
        client = DummyClient(timeout=timeout, responses=responses, calls=calls)
        return client

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_stub)
    provider = OllamaProvider(settings)

    provider.ensure_model()

    assert provider._model_ensured is True
    assert calls == [("get", 60, f"{provider.base_url}/api/tags")]


def test_ensure_model_pulls_when_model_missing(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls_first: list = []
    calls_second: list = []

    responses_first = iter([DummyResponse({"models": []})])
    responses_second = iter([DummyResponse({})])

    clients = []

    def client_stub(*, timeout):
        if timeout == 60:
            client = DummyClient(timeout=timeout, responses=responses_first, calls=calls_first)
        else:
            client = DummyClient(timeout=timeout, responses=responses_second, calls=calls_second)
        clients.append(client)
        return client

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_stub)
    provider = OllamaProvider(settings)

    provider.ensure_model()

    assert provider._model_ensured is True
    assert len(clients) == 2
    assert calls_first == [("get", 60, f"{provider.base_url}/api/tags")]
    assert calls_second == [
        (
            "post",
            None,
            f"{provider.base_url}/api/pull",
            {"name": provider.model_name},
        )
    ]


def test_generate_posts_prompt_and_returns_response(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls: list = []
    response_payload = {"response": "generated text"}

    def client_stub(*, timeout):
        responses = iter([DummyResponse(response_payload)])
        return DummyClient(timeout=timeout, responses=responses, calls=calls)

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_stub)
    provider = OllamaProvider(settings)

    result = provider.generate("hello world")

    assert result == "generated text"
    assert calls == [
        (
            "post",
            None,
            f"{provider.base_url}/api/generate",
            {
                "model": provider.model_name,
                "prompt": "hello world",
                "stream": False,
                "options": {
                    "num_ctx": settings.max_context_tokens,
                    "num_predict": settings.max_generation_tokens,
                },
            },
        )
    ]


def test_generate_merges_context_payload(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    calls: list = []

    def client_stub(*, timeout):
        responses = iter([DummyResponse({"response": "ok"})])
        return DummyClient(timeout=timeout, responses=responses, calls=calls)

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_stub)
    provider = OllamaProvider(settings)

    context = {"system": "test", "prompt": "ignored"}
    provider.generate("prompt", context=context)

    assert calls[0][3]["system"] == "test"
    assert calls[0][3]["prompt"] == "prompt"
