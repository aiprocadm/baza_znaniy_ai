"""API routes for the knowledge base service."""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from http import HTTPStatus
from typing import Any, Iterable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.chat.store import ChatStore, ConversationAccessError
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


def _resolve_app_state(request: Request | None):
    if request is not None:
        return request.app.state
    try:  # pragma: no cover - fallback for test stubs without Request injection
        from app.main import app as main_app

        return main_app.state
    except Exception:  # pragma: no cover - defensive guard
        raise RuntimeError("Не удалось определить состояние приложения")


def _check_sqlite_ready(state) -> dict[str, Any]:
    status_info: dict[str, Any] = {"status": "ok"}
    chat_store = getattr(state, "chat_store", None)
    settings = getattr(state, "settings", None)

    if isinstance(chat_store, ChatStore):
        connect = getattr(chat_store, "_connect", None)
        if callable(connect):
            try:
                with closing(connect()) as connection:
                    connection.execute("SELECT 1")
            except sqlite3.Error as exc:
                message = f"SQLite недоступна: {exc}"
                status_info.update(status="error", detail=message)
            except Exception as exc:  # pragma: no cover - defensive guard
                message = f"Ошибка проверки SQLite: {exc}"
                status_info.update(status="error", detail=message)
        else:  # pragma: no cover - unexpected backend implementation
            status_info.update(status="error", detail="Неподдерживаемый backend чата")
    else:
        backend = getattr(settings, "chat_db_backend", None)
        if backend and backend.lower() != "sqlite":
            status_info.update(status="skipped", detail=f"backend {backend}")
        else:
            status_info.update(status="error", detail="Хранилище чата не инициализировано")

    return status_info


def _check_vector_store_ready(state) -> dict[str, Any]:
    status_info: dict[str, Any] = {"status": "ok"}
    vector_store = getattr(state, "vector_store", None)

    if vector_store is None:
        status_info.update(status="error", detail="Векторное хранилище не инициализировано")
        return status_info

    ensure_ready = getattr(vector_store, "ensure_ready", None)
    if not callable(ensure_ready):  # pragma: no cover - unexpected implementation
        status_info.update(status="error", detail="Векторное хранилище не поддерживает проверку готовности")
        return status_info

    try:
        ensure_ready()
    except Exception as exc:
        status_info.update(status="error", detail=f"Векторное хранилище недоступно: {exc}")

    return status_info


def _check_llm_ready(state) -> dict[str, Any]:
    llm_provider = getattr(state, "llm_provider", None)
    status_info: dict[str, Any] = {
        "status": "ok" if llm_provider is not None else "error",
        "provider": getattr(llm_provider, "name", "unknown"),
    }

    if llm_provider is None:
        status_info["detail"] = "LLM провайдер не настроен"
        return status_info

    try:
        ensure_ready = getattr(llm_provider, "ensure_ready", None)
        if callable(ensure_ready):
            ensure_ready()
            status_info["model"] = "ok"
        else:
            llm_provider.ensure_model()
            status_info["model"] = "ok"

        ensure_adapter = getattr(llm_provider, "ensure_adapter", None)
        adapter_name = getattr(llm_provider, "adapter_name", None)
        if callable(ensure_adapter):
            ensure_adapter()
            if adapter_name:
                status_info["adapter"] = "ok"
        elif adapter_name:
            status_info["adapter"] = "unknown"
    except Exception as exc:
        status_info.update(status="error", detail=f"LLM недоступна: {exc}")

    return status_info


@router.get("/ready", response_class=JSONResponse)
def ready(request: Request | None = None) -> JSONResponse:
    """Return an extended readiness status for orchestrators and health checks."""

    state = _resolve_app_state(request)

    sqlite_status = _check_sqlite_ready(state)
    vector_status = _check_vector_store_ready(state)
    llm_status = _check_llm_ready(state)

    problems: list[str] = []
    for component, result in (
        ("sqlite", sqlite_status),
        ("vector_store", vector_status),
        ("llm", llm_status),
    ):
        if result.get("status") == "error":
            detail = result.get("detail")
            problems.append(f"{component}: {detail}" if detail else component)

    payload = {
        "status": "ok" if not problems else "error",
        "ts": int(time.time()),
        "details": {
            "sqlite": sqlite_status,
            "vector_store": vector_status,
            "llm": llm_status,
        },
    }

    if problems:
        payload["message"] = "; ".join(problems)
        return JSONResponse(payload, status_code=int(HTTPStatus.SERVICE_UNAVAILABLE))

    payload["message"] = "Service ready"
    return JSONResponse(payload, status_code=int(HTTPStatus.OK))


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

    hits = apply_rerank(
        inp.message,
        hits,
        settings.rerank_limit,
        settings.rerank_enabled,
        reranker,
    )
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
        "max_context_tokens": settings.max_context_tokens or None,
        "max_generation_tokens": settings.max_generation_tokens or None,
    }


__all__ = ["router"]
