"""Tests for :mod:`app.ollama_client`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import app.ollama_client as ollama_client


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


def test_ensure_model_skips_pull_when_model_present(monkeypatch: pytest.MonkeyPatch):
    calls: list = []

    def client_stub(*, timeout):
        responses = iter([DummyResponse({"models": [{"name": ollama_client.MODEL_NAME}]})])
        client = DummyClient(timeout=timeout, responses=responses, calls=calls)
        return client

    monkeypatch.setattr(ollama_client.httpx, "Client", client_stub)

    ollama_client.ensure_model()

    assert calls == [("get", 60, f"{ollama_client.OLLAMA_BASE_URL}/api/tags")]


def test_ensure_model_pulls_when_model_missing(monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.setattr(ollama_client.httpx, "Client", client_stub)

    ollama_client.ensure_model()

    assert len(clients) == 2
    assert calls_first == [("get", 60, f"{ollama_client.OLLAMA_BASE_URL}/api/tags")]
    assert calls_second == [
        (
            "post",
            None,
            f"{ollama_client.OLLAMA_BASE_URL}/api/pull",
            {"name": ollama_client.MODEL_NAME},
        )
    ]


def test_generate_posts_prompt_and_returns_response(monkeypatch: pytest.MonkeyPatch):
    calls: list = []
    response_payload = {"response": "generated text"}

    def client_stub(*, timeout):
        responses = iter([DummyResponse(response_payload)])
        return DummyClient(timeout=timeout, responses=responses, calls=calls)

    monkeypatch.setattr(ollama_client.httpx, "Client", client_stub)

    result = ollama_client.generate("hello world")

    assert result == "generated text"
    assert calls == [
        (
            "post",
            None,
            f"{ollama_client.OLLAMA_BASE_URL}/api/generate",
            {"model": ollama_client.MODEL_NAME, "prompt": "hello world", "stream": False},
        )
    ]


def test_ensure_model_swallows_network_errors(monkeypatch: pytest.MonkeyPatch):
    class ExplodingClient:
        def __init__(self, *, timeout):
            pass

        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ollama_client.httpx, "Client", ExplodingClient)

    # Should not raise despite the runtime error.
    ollama_client.ensure_model()
