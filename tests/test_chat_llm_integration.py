from __future__ import annotations

import pytest

from tests.stubs.fastapi import Request

from fastapi import HTTPException, status

from app.api.routes import chat
from app.core.app import create_app
from app.core.config import Settings
from app.llm.exceptions import ModelNotFoundError
from app.models.chat import ChatIn


class StubProvider:
    handles_citations = False

    def __init__(self) -> None:
        self.prompts: list[tuple[str, dict[str, object] | None]] = []
        self.ready = False

    def ensure_model(self) -> None:
        self.ready = True

    def generate(self, prompt: str, *, context: dict[str, object] | None = None) -> str:
        assert self.ready, "ensure_model should be called before generate"
        self.prompts.append((prompt, dict(context or {})))
        return "Ответ"


class MissingModelProvider(StubProvider):
    def ensure_model(self) -> None:
        raise ModelNotFoundError("missing.gguf")


SERVICE_UNAVAILABLE = getattr(status, "HTTP_503_SERVICE_UNAVAILABLE", 503)


def _prepare_app(tmp_path, provider) -> tuple[Request, Settings]:
    app = create_app(provider)
    app.state.settings.data_dir = tmp_path
    app.state.chat_store = type(app.state.chat_store)(str(tmp_path / "chat.sqlite3"))
    app.state.fallback_index = [
        {"file": "doc.txt", "page": 1, "text": "пример", "score": 0.42},
    ]
    app.state.vector_store = None
    app.state.rerank_enabled = False
    app.state.reranker = None
    request = Request()
    request.app = app  # type: ignore[attr-defined]
    settings = app.state.settings
    settings.rerank_enabled = False
    return request, settings


def test_chat_endpoint_uses_fallback_index(tmp_path):
    provider = StubProvider()
    request, _settings = _prepare_app(tmp_path, provider)

    payload = chat(
        request,
        ChatIn(user_id="alice", conversation_id=None, message="Привет"),
    )

    assert payload["answer"].startswith("Ответ")
    assert payload["citations"]
    assert provider.prompts, "LLM provider should have been called"


def test_chat_returns_503_when_model_missing(tmp_path):
    provider = MissingModelProvider()
    request, _settings = _prepare_app(tmp_path, provider)

    with pytest.raises(HTTPException) as exc_info:
        chat(
            request,
            ChatIn(user_id="alice", conversation_id=None, message="Привет"),
        )

    assert exc_info.value.status_code == SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "LLM_MODEL_MISSING"


def test_chat_passes_generation_settings(tmp_path):
    provider = StubProvider()
    request, settings = _prepare_app(tmp_path, provider)
    settings.llm_temperature = 0.33
    settings.llm_top_p = 0.77
    settings.llm_top_k = 12
    settings.llm_max_tokens = 222

    chat(
        request,
        ChatIn(user_id="alice", conversation_id=None, message="Привет"),
    )

    assert provider.prompts, "Provider must receive a generation call"
    _prompt, context = provider.prompts[-1]
    assert context["temperature"] == pytest.approx(0.33)
    assert context["top_p"] == pytest.approx(0.77)
    assert context["top_k"] == 12
    assert context["max_tokens"] == 222
