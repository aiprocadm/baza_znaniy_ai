"""Unit tests for chat endpoint interaction with LLM providers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List, Tuple

import pytest

from app.models import ChatRequest, Citation

_CHAT_MODULE_NAME = "test_chat_module"
_CHAT_MODULE_PATH = Path(__file__).resolve().parents[1] / "app" / "api" / "v1" / "chat.py"
_CHAT_SPEC = importlib.util.spec_from_file_location(_CHAT_MODULE_NAME, _CHAT_MODULE_PATH)
assert _CHAT_SPEC and _CHAT_SPEC.loader  # pragma: no cover - import guard
_CHAT_MODULE = importlib.util.module_from_spec(_CHAT_SPEC)
sys.modules[_CHAT_MODULE_NAME] = _CHAT_MODULE
_CHAT_SPEC.loader.exec_module(_CHAT_MODULE)
chat_endpoint = _CHAT_MODULE.chat


class DummyChatStore:
    def __init__(self) -> None:
        self._conversation_id = "conv-1"
        self.recorded: list[Tuple[str, str]] = []

    def ensure_conversation(self, user_id: str, conversation_id: str | None) -> str:
        return self._conversation_id

    def get_summary(self, conversation_id: str) -> str:
        return "Итог: всё хорошо"

    def get_recent_messages(self, conversation_id: str, limit: int) -> Iterable[Tuple[str, str]]:
        return [("user", "привет"), ("assistant", "ответ")]

    def record_exchange(self, conversation_id: str, message: str, answer: str) -> None:
        self.recorded.append((message, answer))

    def messages_since_summary(self, conversation_id: str) -> int:
        return 0


class DummySummarizer:
    def __init__(self) -> None:
        self.called_with: list[str] = []

    def summarize(self, conversation_id: str) -> None:
        self.called_with.append(conversation_id)


class RecordingProvider:
    formats_citations = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, List[Citation]]] = []
        self.ensure_calls = 0

    def ensure_model(self) -> None:
        self.ensure_calls += 1

    def generate(
        self,
        message: str,
        context: str | None = None,
        citations: Iterable[Citation] | None = None,
    ) -> str:
        citations_list = list(citations or [])
        self.calls.append((message, context, citations_list))
        return "Ответ"


class RecordingStubProvider(RecordingProvider):
    formats_citations = True

    def __init__(self) -> None:
        super().__init__()
        self.last_output: str | None = None

    def generate(
        self,
        message: str,
        context: str | None = None,
        citations: Iterable[Citation] | None = None,
    ) -> str:
        result = super().generate(message, context, citations)
        lines = [result]
        if context:
            lines.append("\nКонтекст:" + context)
        entries = []
        for idx, citation in enumerate(self.calls[-1][2], start=1):
            label = citation.file or "неизвестный источник"
            if citation.page is not None:
                label = f"{label} — страница {citation.page}"
            entries.append(f"[{idx}] {label}")
        if entries:
            lines.extend(["", "Источники:", "\n".join(entries)])
        formatted = "\n".join(lines)
        self.last_output = formatted
        return formatted


def _app_state(provider) -> SimpleNamespace:
    return SimpleNamespace(
        chat_store=DummyChatStore(),
        summarizer=DummySummarizer(),
        memory_store=None,
        retrieve_topk=3,
        rerank_topk=3,
        min_citations=1,
        max_citations=2,
        chat_history_limit=5,
        chat_summary_trigger=10,
        context_token_limit=3000,
        llm_provider=provider,
    )


def test_chat_constructs_prompt_and_formats_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RecordingProvider()
    app_state = _app_state(provider)

    monkeypatch.setattr(_CHAT_MODULE, "search", lambda query, top_k: [{"text": "doc"}])
    monkeypatch.setattr(_CHAT_MODULE, "build_context", lambda hits, token_limit: "context")
    monkeypatch.setattr(
        _CHAT_MODULE,
        "select_citations",
        lambda hits, minimum, maximum: ([{"file": "doc.pdf", "page": 2, "score": 0.9}], True),
    )

    payload = ChatRequest(user_id="user", message="Что нового?", conversation_id=None)
    request = SimpleNamespace(app=SimpleNamespace(state=app_state))

    response = chat_endpoint(payload, request=request, tenant="tenant")

    assert provider.ensure_calls == 1
    assert provider.calls, "provider.generate should be invoked"
    message, context, citations = provider.calls[-1]
    assert "Что нового?" in message
    assert "Conversation summary" in message
    assert context == "Retrieved context:\ncontext"
    assert len(citations) == 1 and isinstance(citations[0], Citation)

    assert response.answer.splitlines()[0] == "Ответ"
    assert "Источники:" in response.answer
    assert "[1] doc.pdf — страница 2" in response.answer
    assert response.citations[0].file == "doc.pdf"
    assert response.citations[0].page == 2


def test_chat_uses_stub_provider_output(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RecordingStubProvider()
    app_state = _app_state(provider)

    monkeypatch.setattr(_CHAT_MODULE, "search", lambda query, top_k: [{"text": "stub"}])
    monkeypatch.setattr(_CHAT_MODULE, "build_context", lambda hits, token_limit: "stub-context")
    monkeypatch.setattr(
        _CHAT_MODULE,
        "select_citations",
        lambda hits, minimum, maximum: ([{"file": "stub.txt", "page": None, "score": 1.0}], True),
    )

    payload = ChatRequest(user_id="user", message="ping", conversation_id=None)
    request = SimpleNamespace(app=SimpleNamespace(state=app_state))

    response = chat_endpoint(payload, request=request, tenant="tenant")

    assert provider.ensure_calls == 1
    assert "Источники" in response.answer
    assert response.answer == provider.last_output
