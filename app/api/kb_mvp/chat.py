"""Ask / streaming-ask / conversation endpoints (protected)."""

from __future__ import annotations
import json
import time
from typing import Any, AsyncIterator, List, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse
from app.observability import retrieval_health
from app.services import kb_llm
from app.services.kb_store import Conversation as StoredConversation
from .common import (
    LOGGER,
    protected,
    _conversation_to_out,
    _format_history,
    _hit_to_out,
    _message_to_out,
    _store_for,
)
from .rag import (
    _RAG_SYSTEM_PROMPT,
    _build_rag_prompt,
    _extractive_answer,
    _generate_answer,
    _retrieve_with_rerank,
)
from .schemas import (
    AskRequest,
    AskResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationRename,
    RetrievalReportOut,
)


@protected.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Answer a question using retrieved chunks (with optional rerank) as context.

    Conversation behaviour:

    * No ``conversation_id`` → a fresh conversation is created and its id
      is returned in the response. The user's question and the
      assistant's answer (with sources) are persisted.
    * ``conversation_id`` for an existing conversation → the last
      ``history_limit`` messages are pre-pended to the RAG prompt as
      context, and the new turn is appended.
    * ``conversation_id`` for a missing conversation → 404.
    """

    store = _store_for(request)

    # Resolve the target conversation
    conversation: StoredConversation
    if payload.conversation_id:
        existing = store.get_conversation(payload.conversation_id)
        if existing is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
        conversation = existing
        if payload.history_limit > 0:
            prior = store.recent_messages(conversation.id, limit=payload.history_limit)
        else:
            prior = []
    else:
        conversation = store.create_conversation(seed_text=payload.question)
        prior = []

    history_text = _format_history(prior) if prior else ""

    hits, rerank_info = _retrieve_with_rerank(store, payload.question, payload.top_k)
    retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())
    answer, provider, model, elapsed_ms = _generate_answer(
        payload.question, hits, request, history=history_text
    )

    # Persist the new turn (user + assistant) — never block the response on this
    try:
        store.add_message(conversation.id, "user", payload.question)
        source_payload = [hit_out.model_dump() for hit_out in (_hit_to_out(h) for h in hits)]
        store.add_message(
            conversation.id,
            "assistant",
            answer,
            sources=source_payload,
            provider=provider,
            model=model,
        )
    except (ValueError, LookupError) as exc:
        LOGGER.warning("Failed to persist conversation turn: %s", exc)

    return AskResponse(
        question=payload.question,
        answer=answer,
        sources=[_hit_to_out(hit) for hit in hits],
        provider=provider,
        model=model,
        elapsed_ms=elapsed_ms,
        rerank=rerank_info,
        retrieval=RetrievalReportOut(**retrieval_out) if retrieval_out else None,
        conversation_id=conversation.id,
    )


# ----------------------------------------------------------------------
# Streaming /ask
# ----------------------------------------------------------------------


def _sse_event(event: str, data: Any) -> str:
    """Format an SSE message: event + JSON-encoded data + blank line."""

    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_extractive(text: str) -> AsyncIterator[str]:
    """Yield extractive fallback as one big chunk (no real streaming)."""

    yield text


async def _stream_legacy(legacy, prompt: str) -> AsyncIterator[str]:
    """Yield from a sync legacy provider — one chunk."""

    import asyncio

    generate = getattr(legacy, "generate", None)
    if not callable(generate):
        return
    text = await asyncio.to_thread(generate, prompt)
    cleaned = (str(text) if text is not None else "").strip()
    if cleaned:
        yield cleaned


@protected.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """Streamed RAG answer over Server-Sent Events.

    Event sequence:

    * ``event: meta``  — ``{conversation_id, sources, rerank, retrieval}``
    * ``event: token`` — ``{text: "<delta>"}`` (multiple)
    * ``event: done``  — ``{provider, model, elapsed_ms}``
    * ``event: error`` — ``{message}`` (on transport failure; stream then closes)

    Conversation semantics mirror :func:`ask`: missing ``conversation_id``
    creates a new conversation. User question + final accumulated
    assistant answer are persisted on stream completion.
    """

    store = _store_for(request)

    if payload.conversation_id:
        existing = store.get_conversation(payload.conversation_id)
        if existing is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
        conversation = existing
        if payload.history_limit > 0:
            prior = store.recent_messages(conversation.id, limit=payload.history_limit)
        else:
            prior = []
    else:
        conversation = store.create_conversation(seed_text=payload.question)
        prior = []

    history_text = _format_history(prior) if prior else ""
    hits, rerank_info = _retrieve_with_rerank(store, payload.question, payload.top_k)
    source_payload = [_hit_to_out(hit).model_dump() for hit in hits]
    retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())

    async def event_generator() -> AsyncIterator[str]:
        start = time.perf_counter()

        meta = {
            "conversation_id": conversation.id,
            "sources": source_payload,
            "rerank": rerank_info.model_dump() if rerank_info else None,
            "retrieval": retrieval_out,
        }
        yield _sse_event("meta", meta)

        if not hits:
            answer = (
                "В базе знаний пока нет данных, релевантных вопросу. "
                "Добавьте документы и повторите запрос."
            )
            yield _sse_event("token", {"text": answer})
            try:
                store.add_message(conversation.id, "user", payload.question)
                store.add_message(conversation.id, "assistant", answer, sources=[], provider="none")
            except (ValueError, LookupError) as exc:
                LOGGER.warning("Failed to persist empty-KB turn: %s", exc)
            yield _sse_event(
                "done",
                {
                    "provider": "none",
                    "model": None,
                    "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
                },
            )
            return

        prompt = _build_rag_prompt(payload.question, hits, history=history_text)
        provider = kb_llm.select_provider()
        provider_name = "extractive"
        model_name: Optional[str] = None
        chunks: list[str] = []

        async def emit_stream(source: AsyncIterator[str]) -> bool:
            """Forward chunks from *source*. Returns True on success."""

            received_any = False
            try:
                async for delta in source:
                    if not delta:
                        continue
                    received_any = True
                    chunks.append(delta)
                    yield _sse_event("token", {"text": delta})
            except kb_llm.LLMTransportError as exc:
                LOGGER.warning("LLM stream transport error: %s", exc)
                yield _sse_event("error", {"message": f"LLM transport error: {exc}"})
                return
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("LLM streaming failure")
                yield _sse_event("error", {"message": "internal streaming error"})
                return
            if not received_any:
                yield _sse_event("error", {"message": "empty completion"})

        stream_fn = getattr(provider, "generate_stream", None) if provider is not None else None
        if provider is not None and callable(stream_fn):
            provider_name = provider.name
            model_name = provider.model
            async for evt in emit_stream(stream_fn(prompt, system=_RAG_SYSTEM_PROMPT)):
                yield evt

        # If primary provider produced nothing — try legacy then extractive
        if not chunks:
            legacy = getattr(getattr(request, "app", None), "state", None)
            legacy = getattr(legacy, "llm_provider", None) if legacy is not None else None
            if legacy is not None:
                provider_name = str(getattr(legacy, "name", "legacy"))
                model_name = None
                async for evt in emit_stream(_stream_legacy(legacy, prompt)):
                    yield evt

        if not chunks:
            provider_name = "extractive"
            model_name = None
            extractive = _extractive_answer(hits)
            async for evt in emit_stream(_stream_extractive(extractive)):
                yield evt

        full_answer = "".join(chunks).strip() or "(empty)"
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)

        try:
            store.add_message(conversation.id, "user", payload.question)
            store.add_message(
                conversation.id,
                "assistant",
                full_answer,
                sources=source_payload,
                provider=provider_name,
                model=model_name,
            )
        except (ValueError, LookupError) as exc:
            LOGGER.warning("Failed to persist streamed turn: %s", exc)

        yield _sse_event(
            "done",
            {"provider": provider_name, "model": model_name, "elapsed_ms": elapsed_ms},
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx response buffering
            "Connection": "keep-alive",
        },
    )


# ----------------------------------------------------------------------
# Conversations
# ----------------------------------------------------------------------


@protected.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(payload: ConversationCreate, request: Request) -> ConversationOut:
    """Create an empty conversation. Title defaults to «Новый диалог»."""

    store = _store_for(request)
    conv = store.create_conversation(title=payload.title)
    return _conversation_to_out(conv)


@protected.get("/conversations", response_model=List[ConversationOut])
def list_conversations(request: Request, limit: int = 100) -> List[ConversationOut]:
    """List conversations ordered by most recently updated."""

    limit = max(1, min(int(limit), 500))
    store = _store_for(request)
    return [_conversation_to_out(c) for c in store.list_conversations(limit=limit)]


@protected.get("/conversations/{conv_id}", response_model=ConversationDetail)
def get_conversation_detail(conv_id: str, request: Request) -> ConversationDetail:
    """Return a conversation with all its messages (chronological)."""

    store = _store_for(request)
    conv = store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
    messages = store.list_messages(conv.id)
    return ConversationDetail(
        **_conversation_to_out(conv).model_dump(),
        messages=[_message_to_out(m) for m in messages],
    )


@protected.patch("/conversations/{conv_id}", response_model=ConversationOut)
def rename_conversation(
    conv_id: str, payload: ConversationRename, request: Request
) -> ConversationOut:
    """Update a conversation's display title."""

    store = _store_for(request)
    updated = store.rename_conversation(conv_id, payload.title)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
    return _conversation_to_out(updated)


@protected.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, request: Request) -> dict[str, Any]:
    """Delete a conversation and all its messages."""

    store = _store_for(request)
    if not store.delete_conversation(conv_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
    return {"ok": True, "id": conv_id}
