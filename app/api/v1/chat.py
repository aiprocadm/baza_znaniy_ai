"""Chat endpoint implementing RAG responses."""

from __future__ import annotations

import logging
from typing import Iterable, List

from fastapi import APIRouter, Depends, HTTPException, Request

try:  # FastAPI<0.115 exposes Depends via param_functions only
    from fastapi.params import Depends as DependsMarker
except ModuleNotFoundError:  # pragma: no cover - compatibility shim
    from fastapi.param_functions import Depends as DependsMarker

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.config import get_settings
from app.models import ChatRequest, ChatResponse, Citation
from app.models.user import UserRecord
from app.services.chat_orchestrator import ChatRequestContext, ChatRuntime, handle_chat

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_SERVICE_UNAVAILABLE = 503


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


def _resolve_user_identifier(user: UserRecord | DependsMarker | None) -> str:
    """Return a safe identifier for logging even if dependency resolution did not run."""

    if user is None:
        return "anonymous"
    if isinstance(user, DependsMarker):
        return "unresolved-user"
    return getattr(user, "email", None) or getattr(user, "id", "unknown-user")


def _resolve_tenant(tenant: str | DependsMarker | None) -> str:
    """Return a tenant label resilient to skipped dependency evaluation."""

    if tenant is None:
        return "unknown-tenant"
    if isinstance(tenant, DependsMarker):
        return "unresolved-tenant"
    return str(tenant)


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request = None,
    user: UserRecord = Depends(get_current_active_user),
    tenant: str = Depends(ensure_tenant_access),
) -> ChatResponse:
    """Return an assistant answer generated via RAG pipeline."""

    if request is None:
        from app.main import app as main_app  # lazy import to avoid cycles

        app_state = main_app.state
    else:
        app_state = request.app.state

    settings = getattr(app_state, "settings", None) or get_settings()
    provider = getattr(app_state, "llm_provider", None)
    if provider is None:
        from app.llm import get_cached_provider

        provider = get_cached_provider(settings)
        app_state.llm_provider = provider

    if provider is None:
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_NOT_CONFIGURED")

    retrieve_topk = payload.top_k or getattr(app_state, "retrieve_topk", 10)
    rerank_limit = getattr(app_state, "rerank_topk", None) or retrieve_topk
    rerank_limit = max(1, min(rerank_limit, retrieve_topk))

    min_citations = getattr(app_state, "min_citations", 3)
    max_citations = getattr(app_state, "max_citations", max(min_citations, 5))

    runtime = ChatRuntime(
        chat_store=app_state.chat_store,
        summarizer=app_state.summarizer,
        memory_store=getattr(app_state, "memory_store", None),
        provider=provider,
        retrieve_topk=retrieve_topk,
        rerank_enabled=getattr(app_state, "rerank_enabled", False),
        reranker=getattr(app_state, "reranker", None),
        rerank_limit=rerank_limit,
        history_limit=getattr(app_state, "chat_history_limit", 12),
        min_citations=min_citations,
        max_citations=max_citations,
        chat_summary_trigger=getattr(app_state, "chat_summary_trigger", 10),
        llm_ctx=getattr(settings, "llm_ctx", None),
        llm_max_tokens=getattr(settings, "llm_max_tokens", None),
        generation_context={
            "temperature": getattr(settings, "llm_temperature", 0.7),
            "top_p": getattr(settings, "llm_top_p", 0.95),
            "top_k": getattr(settings, "llm_top_k", 40),
            "max_tokens": getattr(settings, "llm_max_tokens", None) or 1024,
        },
    )
    context = ChatRequestContext(
        tenant=_resolve_tenant(tenant),
        user=None if isinstance(user, DependsMarker) else user,
    )
    LOGGER.debug(
        "Handling chat request",
        extra={
            "tenant": context.tenant,
            "user": _resolve_user_identifier(user),
        },
    )

    return handle_chat(payload, runtime, context, format_answer=_format_answer)
