"""Tests for the Ollama-backed LLM provider wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from app.llm.providers import OllamaProvider


class DummyResponse:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - mimic httpx API
        return None


class DummyClient:
    def __init__(self, *, timeout: Any, responses: Iterator[DummyResponse], calls: list) -> None:
        self.timeout = timeout
        self._responses = responses
        self.calls = calls

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - interface stub
        return False

    def get(self, url: str) -> DummyResponse:
        self.calls.append(("get", self.timeout, url))
        return next(self._responses)

    def post(self, url: str, json: dict[str, Any]) -> DummyResponse:
        self.calls.append(("post", self.timeout, url, json))
        return next(self._responses)


def _provider(base_url: str = "http://ollama", model: str = "model") -> OllamaProvider:
    settings = SimpleNamespace(
        ollama_base_url=base_url,
        llm_model_name=model,
        max_context_tokens=2048,
    )
    return OllamaProvider(settings=settings)  # type: ignore[arg-type]


def test_ensure_model_skips_pull_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    provider = _provider()

    def client_factory(*, timeout: Any) -> DummyClient:
        responses = iter([DummyResponse({"models": [{"name": provider.model_name}]})])
        return DummyClient(timeout=timeout, responses=responses, calls=calls)

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_factory)

    provider.ensure_model()

    assert calls == [("get", 60, f"{provider.base_url}/api/tags")]


def test_ensure_model_pulls_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    get_calls: list = []
    post_calls: list = []
    provider = _provider("http://ollama.local", "missing-model")

    responses_first = iter([DummyResponse({"models": []})])
    responses_second = iter([DummyResponse({})])

    def client_factory(*, timeout: Any) -> DummyClient:
        if timeout == 60:
            return DummyClient(timeout=timeout, responses=responses_first, calls=get_calls)
        return DummyClient(timeout=timeout, responses=responses_second, calls=post_calls)

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_factory)

    provider.ensure_model()

    assert get_calls == [("get", 60, f"{provider.base_url}/api/tags")]
    assert post_calls == [
        (
            "post",
            None,
            f"{provider.base_url}/api/pull",
            {"name": provider.model_name},
        )
    ]


def test_generate_combines_context_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    provider = _provider("http://ollama.internal", "gen-model")

    def client_factory(*, timeout: Any) -> DummyClient:
        responses = iter([DummyResponse({"response": "result"})])
        return DummyClient(timeout=timeout, responses=responses, calls=calls)

    monkeypatch.setattr("app.llm.providers.httpx.Client", client_factory)

    result = provider.generate("Hello", context="Context block")

    assert result == "result"
    assert calls == [
        (
            "post",
            None,
            f"{provider.base_url}/api/generate",
            {
                "model": provider.model_name,
                "prompt": "Context block\n\nHello",
                "stream": False,
                "options": {"num_ctx": provider.max_context_tokens},
            },
        )
    ]


def test_ensure_model_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()

    class ExplodingClient:
        def __init__(self, *, timeout: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr("app.llm.providers.httpx.Client", ExplodingClient)

    provider.ensure_model()  # should not raise despite failure
