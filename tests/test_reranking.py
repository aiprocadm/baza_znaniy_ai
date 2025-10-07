from __future__ import annotations

from typing import Any, List
import types

import pytest
from app.api.v1 import chat as chat_module
from app.core.config import Settings
from app.models import ChatRequest, ChatResponse
from app.retriever.rerank import CrossEncoderReranker


class StubChatStore:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, str]] = []

    def ensure_conversation(self, user_id: str, conversation_id: str | None) -> str:
        return "conversation-id"

    def get_summary(self, conversation_id: str) -> str:
        return ""

    def get_recent_messages(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        return []

    def record_exchange(self, conversation_id: str, message: str, answer: str) -> None:
        self.records.append((conversation_id, message, answer))

    def messages_since_summary(self, conversation_id: str) -> int:
        return 0


class StubSummarizer:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def summarize(self, conversation_id: str) -> None:
        self.calls.append(conversation_id)


class StubLLM:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, dict[str, Any]]] = []

    def ensure_model(self) -> None:
        return None

    def generate(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        self.prompts.append((prompt, dict(context or {})))
        return "Ответ"


class StubVectorStore:
    def __init__(self, hits: List[dict[str, Any]], expected_top_k: int) -> None:
        self._hits = hits
        self._expected_top_k = expected_top_k
        self.calls: list[tuple[str, int]] = []

    def ensure_collection(self) -> None:
        return None

    def search(self, query: str, top_k: int) -> List[dict[str, Any]]:
        self.calls.append((query, top_k))
        assert top_k == self._expected_top_k
        return list(self._hits)


class DummyRequest:
    def __init__(self, state: types.SimpleNamespace) -> None:
        self.app = types.SimpleNamespace(state=state)


def _build_request(
    settings: Settings,
    hits: List[dict[str, Any]],
    reranker: Any,
) -> tuple[DummyRequest, types.SimpleNamespace]:
    min_citations, max_citations = settings.citations_bounds
    state = types.SimpleNamespace(
        settings=settings,
        chat_store=StubChatStore(),
        llm_provider=StubLLM(),
        llm_client=None,
        vector_store=StubVectorStore(hits, settings.retrieve_topk),
        summarizer=StubSummarizer(),
        memory_store=None,
        fallback_index=[],
        reranker=reranker,
        chat_history_limit=settings.chat_history_limit,
        retrieve_topk=settings.retrieve_topk,
        rerank_topk=settings.rerank_limit,
        min_citations=min_citations,
        max_citations=max_citations,
        rerank_enabled=settings.rerank_enabled,
        chat_summary_trigger=settings.chat_summary_trigger,
    )
    return DummyRequest(state), state


@pytest.fixture()
def sample_hits() -> List[dict[str, Any]]:
    return [
        {"file": f"doc{i}.pdf", "page": i, "text": f"text {i}", "score": 0.5 - i * 0.1}
        for i in range(1, 5)
    ]


def test_cross_encoder_reranker_preserves_original_hits() -> None:
    hits = [
        {"file": "doc1.pdf", "page": 1, "text": "first", "score": 0.9},
        {"file": "doc2.pdf", "page": 2, "text": "second", "score": 0.8},
        {"file": "doc3.pdf", "page": 3, "text": "third", "score": 0.7},
    ]

    class StubModel:
        def __init__(self) -> None:
            self.calls: list[list[tuple[str, str]]] = []

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            self.calls.append(list(pairs))
            return [0.2, 0.9, 0.5]

    model = StubModel()
    reranker = CrossEncoderReranker(model=model, batch_size=2)

    result = reranker.rerank("query", hits, top_k=2)

    assert model.calls  # ensure the stub model was used
    assert result == [hits[1], hits[2]]  # highest predicted scores
    assert hits[0]["score"] == 0.9 and hits[1]["score"] == 0.8 and hits[2]["score"] == 0.7


def test_chat_uses_reranker_when_enabled(sample_hits: List[dict[str, Any]]) -> None:
    settings = Settings().model_copy(
        update={
            "retrieve_topk": 4,
            "rerank_topk": 2,
            "rerank_enabled": True,
            "chat_min_citations": 2,
            "chat_max_citations": 2,
        }
    )

    class StubReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, List[dict[str, Any]], int]] = []

        def rerank(
            self, query: str, hits: List[dict[str, Any]], top_k: int
        ) -> List[dict[str, Any]]:
            self.calls.append((query, list(hits), top_k))
            return list(reversed(hits))[:top_k]

    reranker = StubReranker()
    request, state = _build_request(settings, sample_hits, reranker)
    payload = ChatRequest(user_id="u", message="hello", conversation_id=None)

    assert settings.rerank_enabled is True
    assert request.app.state.reranker is reranker

    original_search = chat_module.search

    def stub_search(query: str, top_k: int = 10) -> List[dict[str, Any]]:
        assert top_k == settings.retrieve_topk
        return list(sample_hits)

    chat_module.search = stub_search
    try:
        response = chat_module.chat(payload, request=request)
    finally:
        chat_module.search = original_search

    assert isinstance(response, ChatResponse)
    assert reranker.calls
    rerank_call = reranker.calls[0]
    assert rerank_call[0] == "hello"
    assert rerank_call[2] == settings.rerank_limit
    expected_files = [item["file"] for item in list(reversed(sample_hits))[: settings.rerank_limit]]
    assert [item.file for item in response.citations] == expected_files


def test_chat_skips_reranker_when_disabled(sample_hits: List[dict[str, Any]]) -> None:
    settings = Settings().model_copy(
        update={
            "retrieve_topk": 4,
            "rerank_topk": 2,
            "rerank_enabled": False,
            "chat_min_citations": 2,
            "chat_max_citations": 2,
        }
    )

    class StubReranker:
        def __init__(self) -> None:
            self.called = False

        def rerank(self, *args: Any, **kwargs: Any) -> List[dict[str, Any]]:
            self.called = True
            return []

    reranker = StubReranker()
    request, state = _build_request(settings, sample_hits, reranker)
    payload = ChatRequest(user_id="u", message="hello", conversation_id=None)

    assert settings.rerank_enabled is False
    assert request.app.state.reranker is reranker

    original_search = chat_module.search

    def stub_search(query: str, top_k: int = 10) -> List[dict[str, Any]]:
        assert top_k == settings.retrieve_topk
        return list(sample_hits)

    chat_module.search = stub_search
    try:
        response = chat_module.chat(payload, request=request)
    finally:
        chat_module.search = original_search

    assert isinstance(response, ChatResponse)
    assert reranker.called is False
    expected_files = [item["file"] for item in sample_hits[: settings.rerank_limit]]
    assert [item.file for item in response.citations] == expected_files
