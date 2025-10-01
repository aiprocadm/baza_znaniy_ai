from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes import ready
from app.chat.store import ChatStore


class _ReadyVectorStore:
    def __init__(self) -> None:
        self.calls = 0

    def ensure_ready(self) -> None:
        self.calls += 1


class _HealthyProvider:
    name = "healthy"

    def __init__(self) -> None:
        self.ready_checks = 0
        self.adapter_checks = 0

    def ensure_model(self) -> None:
        return None

    def ensure_ready(self) -> None:
        self.ready_checks += 1

    def ensure_adapter(self) -> None:
        self.adapter_checks += 1

    def generate(self, prompt: str, *, context=None) -> str:  # pragma: no cover - not used in tests
        return prompt


class _BrokenProvider(_HealthyProvider):
    def ensure_ready(self) -> None:
        raise RuntimeError("model not available")


def _make_request(state: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _build_state(tmp_path: Path, provider, vector_store) -> SimpleNamespace:
    chat_store = ChatStore(str(tmp_path / "chat.sqlite3"))
    settings = SimpleNamespace(chat_db_backend="sqlite")
    return SimpleNamespace(
        chat_store=chat_store,
        settings=settings,
        vector_store=vector_store,
        llm_provider=provider,
    )


def _extract_json(response) -> dict[str, object]:
    if hasattr(response, "content") and isinstance(response.content, dict):
        return response.content
    body = getattr(response, "body", None)
    if isinstance(body, (bytes, bytearray)):
        return json.loads(body.decode())
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return json.loads(text)
    raise AssertionError("Unable to decode JSON response")


def test_ready_endpoint_returns_ok(tmp_path: Path) -> None:
    provider = _HealthyProvider()
    vector_store = _ReadyVectorStore()
    state = _build_state(tmp_path, provider, vector_store)
    request = _make_request(state)

    response = ready(request)
    data = _extract_json(response)

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["details"]["sqlite"]["status"] == "ok"
    assert data["details"]["vector_store"]["status"] == "ok"
    assert data["details"]["llm"]["status"] == "ok"


def test_ready_endpoint_reports_llm_error(tmp_path: Path) -> None:
    provider = _BrokenProvider()
    vector_store = _ReadyVectorStore()
    state = _build_state(tmp_path, provider, vector_store)
    request = _make_request(state)

    response = ready(request)
    data = _extract_json(response)

    assert response.status_code == 503
    assert data["status"] == "error"
    assert "model not available" in data["message"]
    assert data["details"]["llm"]["status"] == "error"


def test_ready_endpoint_reports_missing_vector_store(tmp_path: Path) -> None:
    provider = _HealthyProvider()
    state = _build_state(tmp_path, provider, vector_store=None)
    request = _make_request(state)

    response = ready(request)
    data = _extract_json(response)

    assert response.status_code == 503
    assert data["status"] == "error"
    assert data["details"]["vector_store"]["status"] == "error"
