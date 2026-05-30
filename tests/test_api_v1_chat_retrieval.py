"""PR2b: the v1 ChatResponse carries an optional per-query retrieval report."""

from __future__ import annotations

import types
from typing import Any, List

import pytest

import app.observability.retrieval_health as retrieval_health
from app.api.v1 import chat as chat_module
from app.core.config import Settings
from app.models import ChatRequest, ChatResponse
from app.observability.retrieval_health import RetrievalReason, RetrievalReport
from app.services import chat_orchestrator


# --------------------------------------------------------------------------- #
# Task 1 — model: the optional retrieval field exists and coerces a dict.
# --------------------------------------------------------------------------- #
def _base_kwargs() -> dict:
    return {
        "answer": "ok",
        "citations": [],
        "conversation_id": "c1",
        "citations_insufficient": False,
        "latency_ms": 1.0,
    }


def test_chat_response_retrieval_defaults_to_none():
    resp = ChatResponse(**_base_kwargs())
    assert resp.retrieval is None


def test_chat_response_coerces_retrieval_dict():
    resp = ChatResponse(
        **_base_kwargs(),
        retrieval={
            "degraded": True,
            "severity": "critical",
            "reasons": [
                {"reason": "vector_backend_down", "severity": "critical", "detail": "boom"}
            ],
        },
    )
    assert resp.retrieval is not None
    assert resp.retrieval.degraded is True
    assert resp.retrieval.severity == "critical"
    assert resp.retrieval.reasons[0].reason == "vector_backend_down"
    assert resp.retrieval.reasons[0].detail == "boom"


# --------------------------------------------------------------------------- #
# Task 2 — wiring: handle_chat populates retrieval from current_report().
# Mirrors the established v1-chat test harness in tests/test_reranking.py.
# --------------------------------------------------------------------------- #
class _StubChatStore:
    def ensure_conversation(self, user_id: str, conversation_id: str | None) -> str:
        return "conversation-id"

    def get_summary(self, conversation_id: str) -> str:
        return ""

    def get_recent_messages(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        return []

    def record_exchange(self, conversation_id: str, message: str, answer: str) -> None:
        return None

    def messages_since_summary(self, conversation_id: str) -> int:
        return 0


class _StubSummarizer:
    def summarize(self, conversation_id: str) -> None:
        return None


class _StubLLM:
    def ensure_model(self) -> None:
        return None

    def generate(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        return "Ответ"


def _build_request(settings: Settings) -> Any:
    min_citations, max_citations = settings.citations_bounds
    state = types.SimpleNamespace(
        settings=settings,
        chat_store=_StubChatStore(),
        llm_provider=_StubLLM(),
        llm_client=None,
        vector_store=None,
        summarizer=_StubSummarizer(),
        memory_store=None,
        fallback_index=[],
        reranker=None,
        chat_history_limit=settings.chat_history_limit,
        retrieve_topk=settings.retrieve_topk,
        rerank_topk=settings.rerank_limit,
        min_citations=min_citations,
        max_citations=max_citations,
        rerank_enabled=False,
        chat_summary_trigger=settings.chat_summary_trigger,
    )
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state))


_SAMPLE_HITS: List[dict[str, Any]] = [
    {"file": "doc1.pdf", "page": 1, "text": "alpha", "score": 0.5},
    {"file": "doc2.pdf", "page": 2, "text": "beta", "score": 0.4},
]


@pytest.fixture(autouse=True)
def _reset_retrieval_health():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def _run_chat(monkeypatch, stub_search) -> Any:
    settings = Settings().model_copy(
        update={
            "langchain_enabled": False,
            "retrieve_topk": 2,
            "chat_min_citations": 1,
            "chat_max_citations": 2,
        }
    )
    request = _build_request(settings)
    payload = ChatRequest(user_id="u", message="вопрос", conversation_id=None)
    monkeypatch.setattr(chat_orchestrator, "search", stub_search)
    return chat_module.chat(payload, request=request)


def test_chat_carries_retrieval_when_degraded(monkeypatch):
    def stub_search(query: str, top_k: int = 10) -> List[dict[str, Any]]:
        # Faithfully simulate the real vectorstore.search() grep-fallback report.
        retrieval_health.report(
            RetrievalReport(
                source="fallback",
                reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,),
                detail="boom",
            )
        )
        return list(_SAMPLE_HITS)

    response = _run_chat(monkeypatch, stub_search)

    assert response.retrieval is not None
    assert response.retrieval.degraded is True
    assert response.retrieval.severity == "critical"
    assert any(r.reason == "vector_backend_down" for r in response.retrieval.reasons)


def test_chat_omits_retrieval_when_clean(monkeypatch):
    def stub_search(query: str, top_k: int = 10) -> List[dict[str, Any]]:
        retrieval_health.report(RetrievalReport(source="vector"))  # clean run
        return list(_SAMPLE_HITS)

    response = _run_chat(monkeypatch, stub_search)

    assert response.retrieval is None
