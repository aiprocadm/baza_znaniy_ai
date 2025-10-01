"""API routes for the knowledge base service."""

from __future__ import annotations

import io
import inspect
import logging
import sqlite3
import time
from contextlib import closing
from http import HTTPStatus
from typing import Any, Iterable, Mapping, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.chat.store import ChatStore, ConversationAccessError
from app.core.deps import DEFAULT_ALLOWED_EXTENSIONS, UploadLimits, get_upload_limits
from app.ingest import parse_and_chunk
from app.llm import (
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    get_cached_provider,
)
from app.memory.store import MemoryStore
from app.models.lora import LoraStatusResponse
from app.models.chat import ChatIn
from app.rag.context import build_context
from app.observability.metrics import (
    record_chat_completion,
    record_index_operation,
    record_search_operation,
)
from app.retriever.rerank import apply_rerank
from app.api.upload_utils import create_upload_file

router = APIRouter()
logger = logging.getLogger(__name__)

_SERVICE_UNAVAILABLE = getattr(status, "HTTP_503_SERVICE_UNAVAILABLE", 503)


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""

    payload = generate_latest()
    return Response(payload, media_type=CONTENT_TYPE_LATEST)


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


async def _check_lora_ready(state) -> dict[str, Any]:
    status_info: dict[str, Any] = {"status": "ok"}
    manager = getattr(state, "lora_manager", None)

    if manager is None:
        status_info.update(status="skipped", detail="LoRA manager not configured")
        return status_info

    get_status = getattr(manager, "get_status", None)
    if not callable(get_status):  # pragma: no cover - unexpected implementation
        status_info.update(status="error", detail="LoRA manager missing status accessor")
        return status_info

    try:
        status = get_status()
        if inspect.isawaitable(status):
            status = await status
        payload = LoraStatusResponse.from_status(status).model_dump()
        status_info["detail"] = payload
        status_info["loaded"] = payload.get("loaded", False)
    except Exception as exc:
        status_info.update(status="error", detail=f"LoRA недоступна: {exc}")

    return status_info


async def build_ready_payload(state) -> Tuple[int, dict[str, Any]]:
    """Assemble the readiness response payload for the given application *state*."""

    sqlite_status = _check_sqlite_ready(state)
    vector_status = _check_vector_store_ready(state)
    llm_status = _check_llm_ready(state)
    lora_status = await _check_lora_ready(state)

    problems: list[str] = []
    for component, result in (
        ("sqlite", sqlite_status),
        ("vector_store", vector_status),
        ("llm", llm_status),
        ("lora", lora_status),
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
            "lora": lora_status,
        },
    }

    if problems:
        payload["message"] = "; ".join(problems)
        return int(HTTPStatus.SERVICE_UNAVAILABLE), payload

    payload["message"] = "Service ready"
    return int(HTTPStatus.OK), payload


@router.get("/ready", response_class=JSONResponse)
async def ready(request: Request) -> JSONResponse:
    """Return an extended readiness status for orchestrators and health checks."""

    state = _resolve_app_state(request)
    status_code, payload = await build_ready_payload(state)
    return JSONResponse(payload, status_code=status_code)


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _get_allowed_extensions(request: Request) -> set[str]:
    state = getattr(request, "app", None)
    state = getattr(state, "state", None)

    if state is not None:
        limits = getattr(state, "upload_limits", None)
        if isinstance(limits, UploadLimits):
            return set(limits.allowed_extensions)

    try:
        limits = get_upload_limits()
    except Exception:
        return set(DEFAULT_ALLOWED_EXTENSIONS)

    if state is not None:
        setattr(state, "upload_limits", limits)

    return set(limits.allowed_extensions)


def _coerce_upload_file(value: Any) -> UploadFile:
    if isinstance(value, UploadFile):
        return value
    if isinstance(value, dict):
        candidate = value.get("files") or value.get("file")
        if candidate is not None:
            return _coerce_upload_file(candidate)
    if isinstance(value, (list, tuple)):
        sequence_candidate = _coerce_sequence(value)
        if sequence_candidate is not None:
            return sequence_candidate
        for item in value:
            if isinstance(item, UploadFile):
                return item
        codex/update-upload-file-handling-and-tests
            if isinstance(item, (list, tuple)):
                nested = _coerce_sequence(item)
                if nested is not None:
                    return nested

            if isinstance(item, (list, tuple)) and item:
                filename = item[0]
                content = item[1] if len(item) > 1 else b""
                content_type = item[2] if len(item) > 2 else None
                return create_upload_file(filename, content, content_type)
        main
    raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_FILE")


def _coerce_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode()
    if payload is None:
        return b""
    read = getattr(payload, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode()
        try:
            return bytes(data)
        except Exception:
            return b""
    try:
        return bytes(payload)
    except Exception:
        return b""


def _coerce_sequence(items: Any) -> UploadFile | None:
    if not isinstance(items, (list, tuple)) or not items:
        return None
    first = items[0]
    if isinstance(first, UploadFile):
        return first
    if isinstance(first, (list, tuple)):
        nested = _coerce_sequence(first)
        if nested is not None:
            return nested
    if isinstance(first, dict):
        nested = _coerce_upload_file(first)
        if nested is not None:
            return nested
    filename = first
    content = items[1] if len(items) > 1 else b""
    return UploadFile(filename=filename, file=io.BytesIO(_coerce_bytes(content)))


def _index_chunks(request: Request, chunks: Iterable[dict[str, Any]]) -> int:
    items = list(chunks)
    if not items:
        return 0

    fallback_index = getattr(request.app.state, "fallback_index", [])
    vector_store = getattr(request.app.state, "vector_store", None)

    operation_start = time.perf_counter()

    try:  # pragma: no cover - optional dependency initialisation
        if vector_store is None:
            raise RuntimeError("Vector store is not configured")
        vector_store.ensure_ready()
        vector_store.upsert(items)
    except Exception:
        duration = time.perf_counter() - operation_start
        record_index_operation("error", "vector", len(items), duration)
        logger.exception("Failed to ensure vector store; using fallback index")

        fallback_start = time.perf_counter()
        fallback_index.extend(items)
        record_index_operation(
            "success", "fallback", len(items), time.perf_counter() - fallback_start
        )
        return len(items)

    record_index_operation("success", "vector", len(items), time.perf_counter() - operation_start)
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
    upload_file = _coerce_upload_file(file)
    filename = (upload_file.filename or "uploaded").strip()
    ext = _normalise_extension(filename)
    allowed_extensions = _get_allowed_extensions(request)
    if ext not in allowed_extensions:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")

    data = await upload_file.read()
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
    chat_status = "success"
    hits: list[dict[str, Any]] = []
    citations_payload: list[dict[str, Any]] = []

    try:
        conversation_id = chat_store.ensure_conversation(inp.user_id, inp.conversation_id)
    except ConversationAccessError as exc:
        chat_status = "error"
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    response: dict[str, Any]
    try:
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

        if vector_store is not None:
            search_start = time.perf_counter()
            try:
                hits = vector_store.search(inp.message, top_k=settings.retrieve_topk)
            except Exception:
                record_search_operation(
                    "chat_vector",
                    "error",
                    time.perf_counter() - search_start,
                    0,
                )
                logger.exception("Vector search failed; using fallback index")
            else:
                record_search_operation(
                    "chat_vector",
                    "success",
                    time.perf_counter() - search_start,
                    len(hits),
                )
        if not hits and fallback_index:
            fallback_start = time.perf_counter()
            hits = fallback_index[: settings.retrieve_topk]
            record_search_operation(
                "chat_fallback",
                "success",
                time.perf_counter() - fallback_start,
                len(hits),
            )

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
        citations_payload = list(citations)

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

        latency_seconds = time.perf_counter() - start
        response = {
            "answer": answer_text,
            "citations": citations,
            "conversation_id": conversation_id,
            "citations_insufficient": not has_minimum_citations,
            "latency_ms": latency_seconds * 1000,
            "max_context_tokens": getattr(settings, "llm_ctx", None),
            "max_generation_tokens": getattr(settings, "llm_max_tokens", None),
        }
    except HTTPException:
        chat_status = "error"
        raise
    except Exception:
        chat_status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        record_chat_completion(
            chat_status,
            duration,
            hits=len(hits),
            citations=len(citations_payload),
        )

    return response


__all__ = ["router"]
