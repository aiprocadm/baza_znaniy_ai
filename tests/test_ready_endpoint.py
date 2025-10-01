from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.api.routes import build_ready_payload
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


def _build_state(tmp_path: Path, provider, vector_store) -> SimpleNamespace:
    chat_store = ChatStore(str(tmp_path / "chat.sqlite3"))
    settings = SimpleNamespace(chat_db_backend="sqlite")
    return SimpleNamespace(
        chat_store=chat_store,
        settings=settings,
        vector_store=vector_store,
        llm_provider=provider,
    )


def test_ready_endpoint_returns_ok(tmp_path: Path) -> None:
    provider = _HealthyProvider()
    vector_store = _ReadyVectorStore()
    state = _build_state(tmp_path, provider, vector_store)
    status_code, payload = asyncio.run(build_ready_payload(state))

    assert status_code == 200
    assert payload["status"] == "ok"
    assert payload["details"]["sqlite"]["status"] == "ok"
    assert payload["details"]["vector_store"]["status"] == "ok"
    assert payload["details"]["llm"]["status"] == "ok"


def test_ready_endpoint_reports_llm_error(tmp_path: Path) -> None:
    provider = _BrokenProvider()
    vector_store = _ReadyVectorStore()
    state = _build_state(tmp_path, provider, vector_store)
    status_code, payload = asyncio.run(build_ready_payload(state))

    assert status_code == 503
    assert payload["status"] == "error"
    assert "model not available" in payload["message"]
    assert payload["details"]["llm"]["status"] == "error"


def test_ready_endpoint_reports_missing_vector_store(tmp_path: Path) -> None:
    provider = _HealthyProvider()
    state = _build_state(tmp_path, provider, vector_store=None)
    status_code, payload = asyncio.run(build_ready_payload(state))

    assert status_code == 503
    assert payload["status"] == "error"
    assert payload["details"]["vector_store"]["status"] == "error"
