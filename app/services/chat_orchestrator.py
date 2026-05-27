"""Application-layer orchestration for chat completion requests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable, List, Mapping

from fastapi import HTTPException, status

from app.chat.store import ChatStoreProtocol, ConversationAccessError
from app.llm import LoRAAdapterNotFoundError, ModelNotFoundError, ModelNotReadyError
from app.memory.store import MemoryStore
from app.models import ChatRequest, ChatResponse, Citation
from app.models.user import UserRecord
from app.observability.metrics import record_chat_completion
from app.rag.context import build_context, select_citations
from app.services.vectorstore import search

LOGGER = logging.getLogger(__name__)
_SERVICE_UNAVAILABLE = getattr(status, "HTTP_503_SERVICE_UNAVAILABLE", 503)


class ChatExecutionMode(StrEnum):
    """Execution mode for chat request handling."""

    LEGACY = "legacy"
    LANGCHAIN = "langchain"


def _resolve_execution_mode(runtime: ChatRuntime) -> ChatExecutionMode:
    return ChatExecutionMode.LANGCHAIN if runtime.langchain_enabled else ChatExecutionMode.LEGACY


def _map_langchain_result_to_chat_response(
    result: Mapping[str, Any],
    *,
    conversation_id: str,
    runtime: ChatRuntime,
    latency_ms: float,
    format_answer: Callable[[str, Iterable[Citation]], str],
) -> ChatResponse:
    answer_text = str(result.get("answer") or result.get("output") or "").strip()
    source_items = []
    if runtime.langchain_return_source_docs:
        source_items = list(
            result.get("sources") or result.get("source_documents") or result.get("context") or []
        )

    citations = [
        Citation(
            file=item.get("file") or item.get("source"),
            page=item.get("page"),
            article=item.get("article"),
            clause=item.get("clause"),
            revision=item.get("revision"),
            revision_date=item.get("revision_date"),
            score=float(item.get("score", 0.0)),
        )
        for item in source_items
        if isinstance(item, Mapping)
    ]
    formatted_answer = format_answer(answer_text, citations)
    return ChatResponse(
        answer=formatted_answer,
        citations=citations,
        conversation_id=conversation_id,
        citations_insufficient=False,
        latency_ms=latency_ms,
        max_context_tokens=runtime.llm_ctx,
        max_generation_tokens=runtime.llm_max_tokens,
    )


@dataclass(slots=True)
class ChatRuntime:
    """Dependencies and runtime configuration required to process a chat request."""

    chat_store: ChatStoreProtocol
    summarizer: object
    memory_store: MemoryStore | None
    provider: object
    retrieve_topk: int
    rerank_enabled: bool
    reranker: object | None
    rerank_limit: int
    history_limit: int
    min_citations: int
    max_citations: int
    chat_summary_trigger: int
    llm_ctx: int | None
    llm_max_tokens: int | None
    generation_context: Mapping[str, object]
    langchain_enabled: bool = False
    langchain_use_history_aware: bool = False
    langchain_return_source_docs: bool = False
    settings: Any | None = None


@dataclass(slots=True)
class ChatRequestContext:
    """Request-scoped metadata used for tracing and authorization."""

    tenant: str
    user: UserRecord | None


def _ensure_provider_ready(provider: object) -> None:
    ensure_model = getattr(provider, "ensure_model", None)
    if ensure_model is None:
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_NOT_CONFIGURED")

    try:
        ensure_model()
    except ModelNotFoundError as exc:
        LOGGER.error("LLM model file is missing", extra={"path": str(exc.path)})
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_MODEL_MISSING") from exc
    except LoRAAdapterNotFoundError as exc:
        LOGGER.error("Configured LoRA adapter is missing", extra={"path": str(exc.path)})
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LORA_ADAPTER_MISSING") from exc
    except ModelNotReadyError as exc:
        LOGGER.warning("LLM provider is not ready", exc_info=exc)
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_NOT_READY") from exc


def handle_chat(
    payload: ChatRequest,
    runtime: ChatRuntime,
    context: ChatRequestContext,
    *,
    format_answer: Callable[[str, Iterable[Citation]], str],
) -> ChatResponse:
    """Run full RAG flow and return a normalized chat response."""

    LOGGER.debug(
        "Handling chat request",
        extra={
            "tenant": context.tenant,
            "user": getattr(context.user, "email", None)
            or getattr(context.user, "id", "unknown-user"),
        },
    )

    _ensure_provider_ready(runtime.provider)

    start = time.perf_counter()
    chat_status = "success"
    citations_count = 0
    hits: List[dict[str, object]] = []

    try:
        conversation_id = runtime.chat_store.ensure_conversation(
            payload.user_id, payload.conversation_id
        )
    except ConversationAccessError as exc:
        chat_status = "error"
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    try:
        summary_text = runtime.chat_store.get_summary(conversation_id) or ""
        history = runtime.chat_store.get_recent_messages(
            conversation_id, limit=runtime.history_limit
        )
        history_text = (
            "\n".join(f"{role}: {content}" for role, content in history) if history else ""
        )

        memory_text = ""
        if isinstance(runtime.memory_store, MemoryStore):
            try:
                memory_text = runtime.memory_store.load_context(payload.user_id, conversation_id)
            except Exception:  # pragma: no cover - defensive logging path
                LOGGER.exception("Failed to load memory context")
                memory_text = ""

        execution_mode = _resolve_execution_mode(runtime)
        if execution_mode is ChatExecutionMode.LANGCHAIN:
            from app.langchain.factory import build_chat_chain, rewrite_with_history

            setattr(runtime.settings, "llm_provider", runtime.provider)
            chain = build_chat_chain(
                runtime.settings,
                tenant_id=context.tenant,
                retrieve_topk=runtime.retrieve_topk,
            )
            rewritten_message = (
                rewrite_with_history(payload.message, history_text)
                if runtime.langchain_use_history_aware
                else payload.message
            )
            lc_result = chain(
                payload=payload,
                context={
                    "question": rewritten_message,
                    "history": history_text,
                    "summary": summary_text,
                    "memory": memory_text,
                    "conversation_id": conversation_id,
                    "request_metadata": {
                        "tenant_id": context.tenant,
                        "owner": context.tenant,
                        "user_id": payload.user_id,
                        "conversation_id": conversation_id,
                    },
                },
            )
            answer = str(lc_result.get("answer") or lc_result.get("output") or "").strip()
            runtime.chat_store.record_exchange(conversation_id, payload.message, answer)
            if (
                runtime.chat_store.messages_since_summary(conversation_id)
                >= runtime.chat_summary_trigger
            ):
                runtime.summarizer.summarize(conversation_id)
            latency_ms = (time.perf_counter() - start) * 1000
            return _map_langchain_result_to_chat_response(
                lc_result,
                conversation_id=conversation_id,
                runtime=runtime,
                latency_ms=latency_ms,
                format_answer=format_answer,
            )

        hits = list(search(payload.message, top_k=runtime.retrieve_topk))
        if hits:
            if runtime.rerank_enabled and runtime.reranker is not None:
                try:
                    hits = runtime.reranker.rerank(payload.message, hits, runtime.rerank_limit)
                except Exception:  # pragma: no cover - defensive fallback
                    LOGGER.exception("Reranking failed; falling back to initial ordering")
                    hits = hits[: runtime.rerank_limit]
            elif len(hits) > runtime.rerank_limit:
                hits = hits[: runtime.rerank_limit]

        context_text = build_context(hits, token_limit=3000)

        prompt_sections: list[str] = [
            "### Система",
            "Ты — корпоративный ассистент, отвечающий на вопросы по базе знаний.",
            "Отвечай кратко и по-русски, опираясь на предоставленный контекст.",
        ]
        if summary_text:
            prompt_sections.extend(["\n### Краткое содержание диалога", summary_text])
        if history_text:
            prompt_sections.extend(["\n### Недавняя история", history_text])
        if memory_text:
            prompt_sections.extend(["\n### Долгосрочная память", memory_text])
        prompt_sections.extend(
            [
                "\n### Контекст",
                context_text or "(релевантные фрагменты не найдены)",
                "\n### Вопрос пользователя",
                payload.message,
                "\n### Инструкция",
                "Если контекст не содержит ответа, честно сообщи об этом. Укажи важные детали кратко.",
            ]
        )
        prompt = "\n".join(filter(None, prompt_sections))

        answer = runtime.provider.generate(prompt, context=runtime.generation_context).strip()

        citations_raw, has_minimum = select_citations(
            hits,
            minimum=runtime.min_citations,
            maximum=runtime.max_citations,
        )
        citations = [
            Citation(
                file=item.get("file"),
                page=item.get("page"),
                article=item.get("article"),
                clause=item.get("clause"),
                revision=item.get("revision"),
                revision_date=item.get("revision_date"),
                score=float(item.get("score", 0.0)),
            )
            for item in citations_raw
        ]
        citations_count = len(citations)

        runtime.chat_store.record_exchange(conversation_id, payload.message, answer)
        if (
            runtime.chat_store.messages_since_summary(conversation_id)
            >= runtime.chat_summary_trigger
        ):
            runtime.summarizer.summarize(conversation_id)

        if isinstance(runtime.memory_store, MemoryStore):
            try:
                runtime.memory_store.record(
                    payload.user_id,
                    conversation_id,
                    payload.message,
                    answer,
                )
            except Exception:  # pragma: no cover - persistence guards
                LOGGER.exception("Failed to persist memory entry")

        formatted_answer = format_answer(answer, citations)

        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            answer=formatted_answer,
            citations=citations,
            conversation_id=conversation_id,
            citations_insufficient=not has_minimum,
            latency_ms=latency_ms,
            max_context_tokens=runtime.llm_ctx,
            max_generation_tokens=runtime.llm_max_tokens,
        )
    except HTTPException:
        chat_status = "error"
        raise
    except Exception:
        chat_status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        record_chat_completion(chat_status, duration, hits=len(hits), citations=citations_count)
