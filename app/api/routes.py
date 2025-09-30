"""API routes for the knowledge base service."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.chat.store import ConversationAccessError
from app.ingest import parse_and_chunk
from app.memory.store import MemoryStore
from app.models.chat import ChatIn
from app.rag.context import build_context
from app.retriever.rerank import apply_rerank

router = APIRouter()
logger = logging.getLogger(__name__)


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
    llm_provider = request.app.state.llm_provider
    vector_store = getattr(request.app.state, "vector_store", None)
    summarizer = request.app.state.summarizer
    memory_store = getattr(request.app.state, "memory_store", None)
    fallback_index = getattr(request.app.state, "fallback_index", [])
    reranker = getattr(request.app.state, "reranker", None)

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
        # codex/implement-reranking-functionality-and-tests
    rerank_limit = settings.rerank_limit
    if hits:
        if settings.rerank_enabled and reranker is not None:
            try:
                hits = reranker.rerank(inp.message, hits, rerank_limit)
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception("Reranking failed; falling back to initial ordering")
                hits = hits[:rerank_limit]
        elif len(hits) > rerank_limit:
            hits = hits[:rerank_limit]


    reranker = getattr(request.app.state, "reranker", None)
    hits = apply_rerank(
        inp.message,
        hits,
        settings.rerank_limit,
        settings.rerank_enabled,
        reranker,
    )
        # main
    context = build_context(hits, token_limit=3000)

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
    prompt_parts.extend([
        "Retrieved context:",
        context or "(нет подходящего контекста)",
        "",
        f"User message: {inp.message}",
        "Сформулируй точный ответ, используя контекст, если он релевантен. Если данных недостаточно, сообщи об этом.",
    ])
    min_citations, max_citations = settings.citations_bounds
    citations, has_minimum_citations = _select_citations(hits, min_citations, max_citations)

    prompt = "\n".join(part for part in prompt_parts if part is not None)

    provider_context = {"citations": citations} if citations else None
    answer = llm_provider.generate(prompt, context=provider_context).strip()

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
    }


__all__ = ["router"]
