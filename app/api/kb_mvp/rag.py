"""Retrieval (bi-encoder + optional cross-encoder rerank) and answer
generation for the MVP /api/kb endpoints."""

from __future__ import annotations
from typing import List, Optional
from fastapi import Request
from app.services import kb_llm, kb_rerank
from app.services.kb_store import KnowledgeBaseStore, SearchHit
from .common import LOGGER
from .schemas import RerankInfo


def _retrieve_with_rerank(
    store: KnowledgeBaseStore,
    query: str,
    top_k: int,
) -> tuple[List[SearchHit], RerankInfo]:
    """Two-stage retrieval: bi-encoder shortlist → cross-encoder rerank.

    When ``KB_RERANK_ENABLED=true`` we over-fetch ``KB_RERANK_CANDIDATES``
    bi-encoder hits and let the cross-encoder pick the final ``top_k``.
    Otherwise the bi-encoder result is truncated to ``top_k`` directly.
    """

    config = kb_rerank.load_config()
    rerank_info = RerankInfo(
        enabled=config.enabled, model=config.model_name if config.enabled else None
    )

    if not config.enabled:
        hits = store.search(query, top_k=top_k)
        return hits, rerank_info

    shortlist_size = max(top_k, config.candidates)
    shortlist = store.search(query, top_k=shortlist_size)
    if not shortlist:
        return [], rerank_info

    result = kb_rerank.rerank_hits(query, shortlist, config=config, top_n=top_k)
    rerank_info = RerankInfo(
        enabled=True,
        used=True,
        model=result.model,
        candidates=result.candidates,
        elapsed_ms=round(result.elapsed_ms, 2) if result.elapsed_ms else None,
    )
    return result.hits, rerank_info


def _format_context(hits: List[SearchHit]) -> str:
    parts = []
    for index, hit in enumerate(hits, start=1):
        source_label = hit.filename or hit.document_title
        parts.append(f"[{index}] {source_label}\n{hit.text}")
    return "\n\n---\n\n".join(parts)


def _extractive_answer(hits: List[SearchHit], limit: int = 3) -> str:
    lines = ["Ответ собран из найденных фрагментов базы знаний:"]
    for index, hit in enumerate(hits[:limit], start=1):
        snippet = hit.text.strip()
        if len(snippet) > 400:
            cut = snippet[:400].rsplit(" ", 1)[0]
            snippet = cut + "…"
        label = hit.filename or hit.document_title
        lines.append(f"[{index}] {label}: {snippet}")
    return "\n".join(lines)


_RAG_SYSTEM_PROMPT = (
    "Ты — помощник корпоративной базы знаний. Отвечай на русском. "
    "Используй ТОЛЬКО фрагменты из приведённого контекста и не добавляй фактов, которых в нём нет. "
    "Каждое утверждение в ответе сопровождай ссылкой на номер подтверждающего фрагмента в формате [N]. "
    "Если в контексте недостаточно данных, ответь ровно фразой: "
    "Не удалось найти в документах информацию для ответа."
)


def _build_rag_prompt(
    question: str,
    hits: List[SearchHit],
    *,
    history: str = "",
) -> str:
    parts = []
    if history:
        parts.append(history)
    parts.append("Фрагменты базы знаний:\n" + _format_context(hits))
    parts.append(f"Вопрос пользователя: {question}\nОтвет:")
    return "\n\n".join(parts)


def _generate_answer(
    question: str,
    hits: List[SearchHit],
    request: Request,
    *,
    history: str = "",
) -> tuple[str, str, Optional[str], Optional[float]]:
    """Build an answer using the configured LLM or the extractive fallback."""

    if not hits:
        return (
            "В базе знаний пока нет данных, релевантных вопросу. Добавьте документы и повторите запрос.",
            "none",
            None,
            None,
        )

    prompt = _build_rag_prompt(question, hits, history=history)

    provider = kb_llm.select_provider()
    if provider is not None:
        try:
            response = provider.generate(prompt, system=_RAG_SYSTEM_PROMPT)
            return response.text, response.provider, response.model, response.elapsed_ms
        except kb_llm.LLMTransportError as exc:
            LOGGER.warning("LLM %s transport error: %s", provider.name, exc)
        except Exception:  # pragma: no cover - defensive fallback
            LOGGER.exception("LLM provider %s failed; using fallback", provider.name)

    legacy = getattr(getattr(request, "app", None), "state", None)
    legacy = getattr(legacy, "llm_provider", None) if legacy is not None else None
    if legacy is not None:
        try:
            ensure_ready = getattr(legacy, "ensure_ready", None)
            if callable(ensure_ready):
                ensure_ready()
            generate = getattr(legacy, "generate", None)
            if callable(generate):
                raw = generate(prompt)
                text = (str(raw) if raw is not None else "").strip()
                if text:
                    return text, str(getattr(legacy, "name", "legacy")), None, None
        except Exception:  # pragma: no cover
            LOGGER.exception("Legacy LLM provider failed")

    return _extractive_answer(hits), "extractive", None, None
