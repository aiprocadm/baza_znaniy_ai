"""API routes for the knowledge base service."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Mapping

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.chat.store import ConversationAccessError
from app.ingest import parse_and_chunk
from app.llm import (
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    get_cached_provider,
)
from app.memory.store import MemoryStore
from app.models.chat import ChatIn
from app.rag.context import build_context
from app.retriever.rerank import apply_rerank

router = APIRouter()
logger = logging.getLogger(__name__)

_SERVICE_UNAVAILABLE = getattr(status, "HTTP_503_SERVICE_UNAVAILABLE", 503)


@router.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


@router.head("/health")
def health_head() -> JSONResponse:
    return health()


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _index_chunks(request: Request, chunks: Iterable[dict[str, Any]]) -> int:
    items = list(chunks)
    if not items:
        return 0

    fallback_index = getattr(request.app.state, "fallback_index", [])
    vector_store = getattr(request.app.state, "vector_store", None)

    try:  # pragma: no cover - optional dependency initialisation
        if vector_store is None:
            raise RuntimeError("Vector store is not configured")
        vector_store.ensure_ready()
    except Exception:
        logger.exception("Failed to ensure vector store; using fallback index")
        fallback_index.extend(items)
        return len(items)

    try:  # pragma: no cover - optional dependency for full ingestion pipeline
        vector_store.upsert(items)
    except Exception:
        fallback_index.extend(items)
        logger.info("Stored %s chunks in fallback index", len(items))
        return len(items)

    return len(items)


def _citation_key(hit: dict[str, Any]) -> tuple[Any, ...]:
    file_id = hit.get("file")
    page = hit.get("page")
    if file_id is None and page is None:
        return (
            hit.get("sha256"),
            hit.get("id"),
            hit.get("text"),
        )
    return (file_id, page)


def _select_citations(
    hits: Iterable[dict[str, Any]],
    minimum: int,
    maximum: int,
) -> tuple[list[dict[str, Any]], bool]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    max_allowed = maximum if maximum >= minimum else minimum

    for hit in hits:
        key = _citation_key(hit)
        if key in seen:
            continue
        seen.add(key)
        citation = {
            "file": hit.get("file"),
            "page": hit.get("page"),
            "score": float(hit.get("score", 0.0)),
        }
        unique.append(citation)
        if len(unique) >= max_allowed:
            break

    has_minimum = len(unique) >= minimum
    return unique, has_minimum


@router.post("/api/docs/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    conversation_id: str | None = Form(None),
) -> dict[str, Any]:
    filename = (file.filename or "uploaded").strip()
    ext = _normalise_extension(filename)
    if ext not in {"pdf", "docx", "txt"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")

    data = await file.read()
    chunks = parse_and_chunk(filename, data)
    if not chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")

    indexed = _index_chunks(request, chunks)
    return {"ok": True, "chunks": indexed}


@router.post("/api/chat")
def chat(request: Request, inp: ChatIn) -> dict[str, Any]:
    settings = request.app.state.settings
    chat_store = request.app.state.chat_store
    llm_provider = getattr(request.app.state, "llm_provider", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    summarizer = request.app.state.summarizer
    memory_store = getattr(request.app.state, "memory_store", None)
    fallback_index = getattr(request.app.state, "fallback_index", [])
    reranker = getattr(request.app.state, "reranker", None)

    if llm_provider is None and settings is not None:
        llm_provider = get_cached_provider(settings)
        request.app.state.llm_provider = llm_provider

    if llm_provider is None:
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_NOT_CONFIGURED")

    try:
        llm_provider.ensure_model()
    except ModelNotFoundError as exc:
        logger.error("LLM model file is missing", extra={"path": str(exc.path)})
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_MODEL_MISSING") from exc
    except LoRAAdapterNotFoundError as exc:
        logger.error("Configured LoRA adapter is missing", extra={"path": str(exc.path)})
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LORA_ADAPTER_MISSING") from exc
    except ModelNotReadyError as exc:
        logger.warning("LLM provider is not ready", exc_info=exc)
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_NOT_READY") from exc

    if vector_store is not None:
        try:  # pragma: no cover - defensive ensure call
            vector_store.ensure_ready()
        except Exception:
            logger.exception("Failed to ensure vector store for chat")

    start = time.perf_counter()

    try:
        conversation_id = chat_store.ensure_conversation(inp.user_id, inp.conversation_id)
    except ConversationAccessError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(
        conversation_id, limit=settings.chat_history_limit
    )
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

    memory_text = ""
    if isinstance(memory_store, MemoryStore):
        try:
            memory_text = memory_store.load_context(inp.user_id, conversation_id)
        except Exception:  # pragma: no cover - defensive lookup
            logger.exception("Failed to load memory context")
            memory_text = ""

    hits: list[dict[str, Any]] = []
    if vector_store is not None:
        try:
            hits = vector_store.search(inp.message, top_k=settings.retrieve_topk)
        except Exception:
            logger.exception("Vector search failed; using fallback index")
    if not hits and fallback_index:
        hits = fallback_index[: settings.retrieve_topk]

    hits = apply_rerank(
        inp.message,
        hits,
        settings.rerank_limit,
        settings.rerank_enabled,
        reranker,
    )
    context = build_context(hits, token_limit=3000)

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
            context or "(релевантные фрагменты не найдены)",
            "\n### Вопрос пользователя",
            inp.message,
            "\n### Инструкция",
            "Если контекст не содержит ответа, честно сообщи об этом. Укажи важные детали кратко.",
        ]
    )

    prompt = "\n".join(filter(None, prompt_sections))

    min_citations, max_citations = settings.citations_bounds
    citations, has_minimum_citations = _select_citations(hits, min_citations, max_citations)

    generation_context: dict[str, object] = {
        "temperature": getattr(settings, "llm_temperature", 0.7),
        "top_p": getattr(settings, "llm_top_p", 0.95),
        "top_k": getattr(settings, "llm_top_k", 40),
        "max_tokens": getattr(settings, "llm_max_tokens", 1024),
    }
    if citations:
        generation_context["citations"] = citations

    answer = llm_provider.generate(prompt, context=generation_context).strip()

    chat_store.record_exchange(conversation_id, inp.message, answer)
    if chat_store.messages_since_summary(conversation_id) >= settings.chat_summary_trigger:
        summarizer.summarize(conversation_id)

    if isinstance(memory_store, MemoryStore):
        try:
            memory_store.record(inp.user_id, conversation_id, inp.message, answer)
        except Exception:  # pragma: no cover - defensive persistence handling
            logger.exception("Failed to persist memory entry")

    answer_text = answer
    if citations and not getattr(llm_provider, "handles_citations", False):
        formatted = []
        for idx, citation in enumerate(citations, start=1):
            location = citation.get("page")
            if location is None:
                formatted.append(f"[{idx}] {citation.get('file', 'неизвестный источник')}")
            else:
                formatted.append(
                    f"[{idx}] {citation.get('file', 'неизвестный источник')} — страница {location}"
                )
        answer_text = "\n\n".join([answer.strip(), "Источники:", "\n".join(formatted)])

    return {
        "answer": answer_text,
        "citations": citations,
        "conversation_id": conversation_id,
        "citations_insufficient": not has_minimum_citations,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "max_context_tokens": getattr(settings, "llm_ctx", None),
        "max_generation_tokens": getattr(settings, "llm_max_tokens", None),
    }


__all__ = ["router"]
