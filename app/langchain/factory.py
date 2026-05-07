"""LangChain factory helpers used by chat orchestrator."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from app.langchain.retrievers import TenantFilteredQdrantRetriever
from app.retriever.vector_store import get_vector_store


RewritePromptTemplate = (
    "Учитывая историю диалога и последний вопрос пользователя, "
    "переформулируй вопрос в самодостаточный запрос для поиска."
)

AnswerPromptTemplate = (
    "Ты — корпоративный ассистент. Используй только предоставленный контекст. "
    "Если данных недостаточно, честно сообщи об этом."
)


def get_history_rewrite_prompt() -> str:
    return RewritePromptTemplate


def get_answer_generation_prompt() -> str:
    return AnswerPromptTemplate


def rewrite_with_history(message: str, history: str) -> str:
    history_text = history.strip()
    if not history_text:
        return message
    return f"{history_text}\n\n{message}".strip()


def build_chat_chain(settings: Any, *, tenant_id: str, retrieve_topk: int) -> Callable[..., Mapping[str, Any]]:
    """Build a retrieval QA chain callable using LangChain composition helpers."""

    try:  # pragma: no cover - optional dependency
        from langchain.chains import (
            create_history_aware_retriever,
            create_retrieval_chain,
            create_stuff_documents_chain,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("LangChain package is required when LANGCHAIN_ENABLED=true") from exc

    provider = getattr(settings, "llm_provider", None)
    if provider is None:
        raise RuntimeError("settings.llm_provider is required for LangChain mode")

    store = get_vector_store(settings)
    retriever = TenantFilteredQdrantRetriever(store=store, tenant_id=tenant_id, k=retrieve_topk)

    history_aware_retriever = create_history_aware_retriever(
        llm=provider,
        retriever=retriever,
        prompt=get_history_rewrite_prompt(),
    )
    document_chain = create_stuff_documents_chain(
        llm=provider,
        prompt=get_answer_generation_prompt(),
    )
    chain = create_retrieval_chain(history_aware_retriever, document_chain)

    def _chain(*, payload: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = dict(context.get("request_metadata") or {})
        invoke_payload = {
            "input": str(context.get("question") or getattr(payload, "message", "")),
            "chat_history": context.get("history", ""),
            "metadata": metadata,
        }
        return chain.invoke(invoke_payload)

    return _chain
