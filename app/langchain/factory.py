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


def _coerce_chat_history(value: Any) -> list[Any]:
    """Normalise history into a list of LangChain BaseMessage objects.

    Accepts:
    * ``list[BaseMessage]`` — returned as-is;
    * ``list[(role, content)]`` or ``list[{"role", "content"}]`` — converted;
    * ``str`` produced by ``"\\n".join(f"{role}: {content}" ...)`` — parsed line by line;
    * ``None``/empty — returns ``[]``.

    Unknown roles default to ``human``.
    """

    if not value:
        return []

    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

    def _make(role: str, content: str) -> BaseMessage:
        role = (role or "").strip().lower()
        text = (content or "").strip()
        if role in {"assistant", "ai"}:
            return AIMessage(content=text)
        if role == "system":
            return SystemMessage(content=text)
        return HumanMessage(content=text)

    if isinstance(value, list):
        out: list[BaseMessage] = []
        for item in value:
            if isinstance(item, BaseMessage):
                out.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                out.append(_make(str(item[0]), str(item[1])))
            elif isinstance(item, dict) and "content" in item:
                out.append(_make(str(item.get("role", "human")), str(item["content"])))
        return out

    if isinstance(value, str):
        out = []
        for line in value.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            role, _, content = line.partition(":")
            out.append(_make(role, content))
        return out

    return []


def _build_history_aware_prompt() -> Any:
    """ChatPromptTemplate for the history-aware retriever rewriter."""

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    return ChatPromptTemplate.from_messages(
        [
            ("system", get_history_rewrite_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def _build_answer_prompt() -> Any:
    """ChatPromptTemplate for the answer-generation (stuff-documents) chain."""

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    return ChatPromptTemplate.from_messages(
        [
            ("system", get_answer_generation_prompt() + "\n\nКонтекст:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_chat_chain(
    settings: Any, *, tenant_id: str, retrieve_topk: int
) -> Callable[..., Mapping[str, Any]]:
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
        prompt=_build_history_aware_prompt(),
    )
    document_chain = create_stuff_documents_chain(
        llm=provider,
        prompt=_build_answer_prompt(),
    )
    chain = create_retrieval_chain(history_aware_retriever, document_chain)

    def _chain(*, payload: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = dict(context.get("request_metadata") or {})
        invoke_payload = {
            "input": str(context.get("question") or getattr(payload, "message", "")),
            "chat_history": _coerce_chat_history(context.get("history")),
            "metadata": metadata,
        }
        return chain.invoke(invoke_payload)

    return _chain
