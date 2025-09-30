"""Chat endpoint implementing RAG responses."""

from __future__ import annotations

import logging
import time
from typing import Iterable, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.chat.store import ChatStoreProtocol, ConversationAccessError
from app.core.deps import get_tenant
from app.llm.providers import LLMProviderProtocol, get_llm_provider
from app.memory.store import MemoryStore
from app.models import ChatRequest, ChatResponse, Citation
from app.rag.context import build_context, select_citations
from app.services.vectorstore import search

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _format_answer(answer: str, citations: Iterable[Citation]) -> str:
    answer_text = answer.strip()
    entries: List[str] = []
    for idx, citation in enumerate(citations, start=1):
        location = (
            f" — страница {citation.page}" if citation.page is not None else ""
        )
        entries.append(f"[{idx}] {citation.file or 'неизвестный источник'}{location}")
    if not entries:
        return answer_text
    return "\n\n".join([answer_text, "Источники:", "\n".join(entries)])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request | None = None,
    tenant: str = Depends(get_tenant),
) -> ChatResponse:
    """Return an assistant answer generated via RAG pipeline."""

    LOGGER.debug("Handling chat request", extra={"tenant": tenant})

    if request is None:
        from app.main import app as main_app  # lazy import to avoid cycles

        app_state = main_app.state
    else:
        app_state = request.app.state
    provider: LLMProviderProtocol = getattr(app_state, "llm_provider", get_llm_provider())
    app_state.llm_provider = provider

    chat_store: ChatStoreProtocol = app_state.chat_store
    summarizer = app_state.summarizer
    memory_store = getattr(app_state, "memory_store", None)
    history_limit = getattr(app_state, "chat_history_limit", 12)
    retrieve_topk = payload.top_k or getattr(app_state, "retrieve_topk", 10)
    rerank_topk = getattr(app_state, "rerank_topk", retrieve_topk)
    min_citations = getattr(app_state, "min_citations", 3)
    max_citations = getattr(app_state, "max_citations", max(min_citations, 5))
    context_token_limit = getattr(app_state, "context_token_limit", 3000)

    start = time.perf_counter()
    provider.ensure_model()

    try:
        conversation_id = chat_store.ensure_conversation(payload.user_id, payload.conversation_id)
    except ConversationAccessError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(conversation_id, limit=history_limit)
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

    memory_text = ""
    if isinstance(memory_store, MemoryStore):
        try:
            memory_text = memory_store.load_context(payload.user_id, conversation_id)
        except Exception:  # pragma: no cover - defensive logging path
            LOGGER.exception("Failed to load memory context")
            memory_text = ""

    hits = search(payload.message, top_k=retrieve_topk)
    if len(hits) > rerank_topk:
        hits = hits[:rerank_topk]

    context_text = build_context(hits, token_limit=context_token_limit)
    context_for_provider = "\n".join(
        part
        for part in (
            "Retrieved context:",
            context_text or "(нет подходящего контекста)",
        )
        if part
    )

    citations_raw, has_minimum = select_citations(hits, minimum=min_citations, maximum=max_citations)
    citations: List[Citation] = [
        Citation(
            file=item.get("file"),
            page=item.get("page"),
            score=float(item.get("score", 0.0)),
        )
        for item in citations_raw
    ]

    prompt_parts = [
        "You are a helpful assistant providing concise answers based on the provided documentation context.",
        "Always answer in Russian.",
    ]
    if summary_text:
        prompt_parts.extend(["Conversation summary:", summary_text])
    if history_text:
        prompt_parts.extend(["Recent chat history:", history_text])
    if memory_text:
        prompt_parts.extend(["Long-term memory:", memory_text])
    prompt_parts.extend(
        [
            f"User message: {payload.message}",
            "Сформулируй точный ответ, используя контекст, если он релевантен. Если данных недостаточно, сообщи об этом.",
        ]
    )
    message = "\n".join(part for part in prompt_parts if part)

    answer = provider.generate(message, context=context_for_provider, citations=citations).strip()

    chat_store.record_exchange(conversation_id, payload.message, answer)
    if chat_store.messages_since_summary(conversation_id) >= getattr(app_state, "chat_summary_trigger", 10):
        summarizer.summarize(conversation_id)

    if isinstance(memory_store, MemoryStore):
        try:
            memory_store.record(payload.user_id, conversation_id, payload.message, answer)
        except Exception:  # pragma: no cover - persistence guards
            LOGGER.exception("Failed to persist memory entry")

    if getattr(provider, "formats_citations", False):
        formatted_answer = answer
    else:
        formatted_answer = _format_answer(answer, citations)

    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(
        answer=formatted_answer,
        citations=citations,
        conversation_id=conversation_id,
        citations_insufficient=not has_minimum,
        latency_ms=latency_ms,
    )
