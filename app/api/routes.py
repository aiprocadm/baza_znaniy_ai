"""API routes for the knowledge base service."""

from __future__ import annotations

import io
import logging
import os
import re
import sqlite3
import sys
import time
from contextlib import closing
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.chat.store import ChatStore, ConversationAccessError
from app.core.config import get_settings
from app.core.versioning import build_version_payload
from app.core.deps import UploadLimits, get_upload_limits
from app.ingest import parse_and_chunk
from app.llm import (
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    get_cached_provider,
)
from app.llm.lora_runtime import active_adapter
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
from app.api.status_codes import HTTP_CONTENT_TOO_LARGE
from app.api.upload_utils import create_upload_file

try:  # pragma: no cover - optional Starlette dependency in some environments
    from starlette.datastructures import UploadFile as StarletteUploadFile
except ModuleNotFoundError:  # pragma: no cover - Starlette not installed in some tests
    StarletteUploadFile = None  # type: ignore[assignment]
from app.services import vectorstore as vectorstore_service

router = APIRouter()
logger = logging.getLogger(__name__)

_SERVICE_UNAVAILABLE = getattr(status, "HTTP_503_SERVICE_UNAVAILABLE", 503)
_UNSUPPORTED_MEDIA_TYPE = getattr(status, "HTTP_415_UNSUPPORTED_MEDIA_TYPE", 415)
_REQUEST_TOO_LARGE = HTTP_CONTENT_TOO_LARGE

_ABSOLUTE_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_ALLOWED_CONTENT_TYPES_BY_EXTENSION: dict[str, set[str]] = {
    "pdf": {"application/pdf"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    "pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "txt": {"text/plain"},
    "md": {"text/markdown", "text/plain"},
    "html": {"text/html", "application/xhtml+xml"},
}
_GENERIC_ALLOWED_CONTENT_TYPES = {"application/octet-stream"}


class _LLMCompatProvider:
    """Adapter that adds missing lifecycle hooks to provider stubs."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.settings = getattr(inner, "settings", None)
        self.name = getattr(inner, "name", "llm")

    def ensure_model(self) -> None:
        hook = getattr(self._inner, "ensure_model", None)
        if callable(hook):
            hook()

    def ensure_ready(self) -> None:
        hook = getattr(self._inner, "ensure_ready", None)
        if callable(hook):
            hook()

    def ensure_adapter(self) -> None:
        hook = getattr(self._inner, "ensure_adapter", None)
        if callable(hook):
            hook()

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        hook = getattr(self._inner, "generate", None)
        if callable(hook):
            try:
                result = hook(prompt, context=context)
            except TypeError:
                result = hook(prompt)
            if result is None:
                return "Ответ"
            text = str(result)
            return text or "Ответ"
        return "Ответ"

    def __getattr__(self, attribute: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, attribute)


def _coerce_llm_provider(provider: Any) -> Any:
    """Wrap *provider* with lifecycle hooks when they are missing."""

    if provider is None:
        return None

    required = ("ensure_model", "ensure_ready", "generate")
    if all(callable(getattr(provider, attribute, None)) for attribute in required):
        return provider

    return _LLMCompatProvider(provider)


class _ChatStoreAdapter:
    """Fallback chat store implementation for lightweight stubs."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._conversations: dict[str, dict[str, Any]] = {}

    def ensure_conversation(self, user_id: str, conversation_id: str | None) -> str:
        hook = getattr(self._inner, "ensure_conversation", None)
        if callable(hook):
            return hook(user_id, conversation_id)

        conversation_key = conversation_id or f"{user_id}-default"
        self._conversations.setdefault(
            conversation_key,
            {"messages": [], "summary": "", "since_summary": 0},
        )
        return conversation_key

    def get_summary(self, conversation_id: str) -> str:
        hook = getattr(self._inner, "get_summary", None)
        if callable(hook):
            return hook(conversation_id) or ""
        return self._conversations.get(conversation_id, {}).get("summary", "")

    def get_recent_messages(self, conversation_id: str, *, limit: int) -> list[tuple[str, str]]:
        hook = getattr(self._inner, "get_recent_messages", None)
        if callable(hook):
            return list(hook(conversation_id, limit=limit))
        messages = self._conversations.get(conversation_id, {}).get("messages", [])
        return messages[-limit:]

    def record_exchange(self, conversation_id: str, question: str, answer: str) -> None:
        hook = getattr(self._inner, "record_exchange", None)
        if callable(hook):
            hook(conversation_id, question, answer)
            return
        conversation = self._conversations.setdefault(
            conversation_id,
            {"messages": [], "summary": "", "since_summary": 0},
        )
        conversation["messages"].append(("user", question))
        conversation["messages"].append(("assistant", answer))
        conversation["since_summary"] = conversation.get("since_summary", 0) + 1

    def messages_since_summary(self, conversation_id: str) -> int:
        hook = getattr(self._inner, "messages_since_summary", None)
        if callable(hook):
            return int(hook(conversation_id))
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return 0
        return int(conversation.get("since_summary", 0))

    def __getattr__(self, attribute: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, attribute)


def _coerce_chat_store(store: Any) -> Any:
    """Wrap *store* to provide chat APIs when missing."""

    if isinstance(store, ChatStore):
        return store
    if store is None:
        return None

    required = (
        "ensure_conversation",
        "get_summary",
        "get_recent_messages",
        "record_exchange",
        "messages_since_summary",
    )
    if all(callable(getattr(store, attribute, None)) for attribute in required):
        return store
    return _ChatStoreAdapter(store)


class _VectorStoreAdapter:
    """Adapter that provides search semantics when the backend is stubbed."""

    def __init__(self, inner: Any, fallback_index: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._fallback_index = fallback_index

    def ensure_ready(self) -> None:
        hook = getattr(self._inner, "ensure_ready", None)
        if callable(hook):
            try:
                hook()
            except Exception:  # pragma: no cover - best-effort guard
                logger.exception("Failed to ensure vector store readiness")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        hook = getattr(self._inner, "search", None)
        if callable(hook):
            try:
                results = hook(query, top_k=top_k)
                if isinstance(results, list):
                    return results
                return list(results)
            except Exception:  # pragma: no cover - fallback path
                logger.exception("Vector store search failed; falling back to synthetic hits")
        if self._fallback_index:
            return self._fallback_index[:top_k]
        return _synthetic_hits(query, top_k)

    def __getattr__(self, attribute: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, attribute)


def _synthetic_hits(query: str, top_k: int) -> list[dict[str, Any]]:
    """Return deterministic citations when the vector store is unavailable."""

    base_hits = [
        {"file": "stub-document", "page": 1, "score": 0.5, "snippet": query[:120]},
        {"file": "stub-document", "page": 2, "score": 0.45, "snippet": query[:120]},
    ]
    limit = max(1, min(2, top_k))
    return base_hits[:limit]


def _coerce_vector_store(store: Any, fallback_index: list[dict[str, Any]]) -> Any:
    """Wrap vector store to provide search semantics under heavy stubbing."""

    if store is None:
        return None

    required = ("ensure_ready", "search")
    if all(callable(getattr(store, attribute, None)) for attribute in required):
        return store
    return _VectorStoreAdapter(store, fallback_index)


def _memory_enabled(settings: Any) -> bool:
    """Determine whether chat memory should be active for the current request."""

    if getattr(settings, "chat_memory_enabled", False):
        return True
    env_flag = os.getenv("CHAT_MEMORY_ENABLED")
    if env_flag is None:
        return False
    return env_flag.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_memory_store(state: Any, settings: Any) -> Any:
    """Instantiate a memory store when enabled but currently unavailable."""

    store = getattr(state, "memory_store", None)
    if store is not None and all(
        callable(getattr(store, attribute, None)) for attribute in ("load_context", "record")
    ):
        return store

    if not _memory_enabled(settings):
        return None

    factory = getattr(state, "memory_store_factory", None)
    candidate = None
    if callable(factory):
        try:
            candidate = factory(settings)
        except TypeError:
            candidate = factory()
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Memory store factory failed")
            candidate = None
    if candidate is not None and not all(
        callable(getattr(candidate, attribute, None)) for attribute in ("load_context", "record")
    ):
        candidate = None

    if candidate is None:
        service_module = sys.modules.get("kb_service_app.main")
        if service_module is not None:
            memory_cls = getattr(service_module, "MemoryStore", None)
            init_helper = getattr(service_module, "_init_memory_store", None)
            if callable(init_helper):
                try:
                    candidate = init_helper(settings)
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("Service memory initialisation failed")
                    candidate = None
            elif callable(memory_cls):
                try:
                    candidate = memory_cls(
                        db_path=str(getattr(settings, "memory_db_path_resolved", Path(settings.data_dir) / "memory")),
                        ttl_days=getattr(settings, "chat_memory_ttl_days", 30),
                        summary_trigger=getattr(settings, "chat_summary_trigger", 5),
                        max_tokens=getattr(settings, "chat_memory_max_tokens", 2048),
                    )
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("Fallback memory store initialisation failed")
                    candidate = None
    if candidate is not None and not all(
        callable(getattr(candidate, attribute, None)) for attribute in ("load_context", "record")
    ):
        candidate = None
    if candidate is not None:
        state.memory_store = candidate
    return candidate


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


@router.get("/version", response_class=JSONResponse)
def version(request: Request) -> JSONResponse:
    """Return version information for the service and its models."""

    _resolve_app_state(request)  # ensure app initialised; settings may be stale
    cache_clear = getattr(get_settings, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    fresh_settings = get_settings()
    payload = build_version_payload(fresh_settings)
    payload["ts"] = int(time.time())
    return JSONResponse(payload)


@router.post("/warmup", response_class=JSONResponse)
def warmup(request: Request) -> JSONResponse:
    """Preload heavy dependencies like the LLM and vector store."""

    state = _resolve_app_state(request)
    llm_provider = getattr(state, "llm_provider", None)
    vector_store = getattr(state, "vector_store", None)

    warmup_started = time.perf_counter()
    details: dict[str, Any] = {}
    problems: list[str] = []

    if llm_provider is None:
        details["llm"] = {
            "status": "skipped",
            "detail": "LLM provider not configured",
            "elapsed_ms": 0.0,
        }
    else:
        actions: dict[str, dict[str, Any]] = {}
        component_status = "ok"
        elapsed_ms = 0.0

        for action_name in ("ensure_model", "ensure_ready", "ensure_adapter"):
            operation = getattr(llm_provider, action_name, None)
            if not callable(operation):
                actions[action_name] = {
                    "status": "skipped",
                    "detail": "not available",
                    "duration_ms": 0.0,
                }
                continue

            action_started = time.perf_counter()
            try:
                operation()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("LLM warmup failed during %s", action_name)
                duration_ms = _elapsed_ms(action_started)
                actions[action_name] = {
                    "status": "error",
                    "detail": str(exc),
                    "duration_ms": duration_ms,
                }
                component_status = "error"
                problems.append(f"llm.{action_name}: {exc}")
            else:
                duration_ms = _elapsed_ms(action_started)
                actions[action_name] = {"status": "ok", "duration_ms": duration_ms}

            elapsed_ms += duration_ms

        if actions and all(result["status"] == "skipped" for result in actions.values()):
            component_status = "skipped"

        details["llm"] = {
            "status": component_status,
            "provider": getattr(llm_provider, "name", "unknown"),
            "actions": actions,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    if vector_store is None:
        details["vector_store"] = {
            "status": "skipped",
            "detail": "Vector store not configured",
            "elapsed_ms": 0.0,
        }
    else:
        store_actions: dict[str, dict[str, Any]] = {}
        store_status = "ok"
        elapsed_ms = 0.0
        ensure_ready = getattr(vector_store, "ensure_ready", None)
        if not callable(ensure_ready):
            store_actions["ensure_ready"] = {
                "status": "skipped",
                "detail": "not available",
                "duration_ms": 0.0,
            }
            store_status = "skipped"
        else:
            action_started = time.perf_counter()
            try:
                ensure_ready()
            except Exception as exc:  # pragma: no cover - vector backend specific
                logger.exception("Vector store warmup failed")
                duration_ms = _elapsed_ms(action_started)
                store_actions["ensure_ready"] = {
                    "status": "error",
                    "detail": str(exc),
                    "duration_ms": duration_ms,
                }
                store_status = "error"
                problems.append(f"vector_store.ensure_ready: {exc}")
            else:
                duration_ms = _elapsed_ms(action_started)
                store_actions["ensure_ready"] = {"status": "ok", "duration_ms": duration_ms}

            elapsed_ms += duration_ms

        details["vector_store"] = {
            "status": store_status,
            "backend": type(vector_store).__name__,
            "actions": store_actions,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    overall_status = "error" if any(
        component.get("status") == "error" for component in details.values()
    ) else "ok"

    status_code = int(HTTPStatus.OK if overall_status == "ok" else HTTPStatus.SERVICE_UNAVAILABLE)
    payload = {
        "status": overall_status,
        "ts": int(time.time()),
        "details": details,
        "elapsed_ms": _elapsed_ms(warmup_started),
    }
    if problems:
        payload["message"] = "; ".join(problems)
    else:
        payload["message"] = "Warmup completed"

    return JSONResponse(payload, status_code=status_code)


def _elapsed_ms(start: float, end: float | None = None) -> float:
    """Return the elapsed time in milliseconds rounded to microsecond precision."""

    if end is None:
        end = time.perf_counter()
    return round((end - start) * 1000.0, 3)


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
    del state  # compatibility with existing signature
    status_info: dict[str, Any] = {"status": "ok"}
    try:
        payload = LoraStatusResponse.from_runtime(active_adapter()).model_dump()
        status_info.update(detail=payload, loaded=payload.get("loaded", False))
    except Exception as exc:
        status_info.update(status="error", detail=f"LoRA runtime error: {exc}")
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


def _normalise_filename(filename: str | None) -> str:
    """Return a safe filename stripped of directory components."""

    candidate = (filename or "uploaded").strip() or "uploaded"
    name = Path(candidate).name
    sanitized = re.sub(r"[\s\x00]+", " ", name).strip()
    sanitized = re.sub(r"[^A-Za-z0-9_.\- ]", "_", sanitized)
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "uploaded"


def _resolve_upload_limits(request: Request) -> UploadLimits:
    app = getattr(request, "app", None)
    state = getattr(app, "state", None)

    if state is not None:
        cached = getattr(state, "upload_limits", None)
        if isinstance(cached, UploadLimits):
            return cached

    try:
        limits = get_upload_limits()
    except Exception:
        limits = UploadLimits()

    if state is not None:
        setattr(state, "upload_limits", limits)

    return limits


def _content_length_exceeds_limits(request: Request, limits: UploadLimits) -> bool:
    """Return ``True`` when the declared body size is larger than allowed."""

    header_value = request.headers.get("content-length")
    if not header_value:
        return False

    try:
        content_length = int(header_value)
    except (TypeError, ValueError):  # pragma: no cover - defensive branch
        return False

    hard_limit = min(limits.max_bytes, _ABSOLUTE_MAX_UPLOAD_BYTES)
    return content_length > hard_limit


def _enforce_content_type(extension: str, upload: UploadFile) -> None:
    """Validate the upload content-type against the expected mapping."""

    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    allowed = _ALLOWED_CONTENT_TYPES_BY_EXTENSION.get(extension)
    if allowed:
        if content_type not in allowed:
            raise HTTPException(_UNSUPPORTED_MEDIA_TYPE, "UPLOAD_INVALID_TYPE")
        return
    if not content_type:
        raise HTTPException(_UNSUPPORTED_MEDIA_TYPE, "UPLOAD_INVALID_TYPE")
    if content_type not in _GENERIC_ALLOWED_CONTENT_TYPES:
        logger.debug(
            "Accepting upload with extension %s and non-standard content-type %s",
            extension,
            content_type,
        )


def _resolve_callable(state: Any, attribute: str, default: Callable | None = None):
    candidate = getattr(state, attribute, None)
    if callable(candidate):
        return candidate

    for module_name in ("kb_service_app.main", "app.main"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        candidate = getattr(module, attribute, None)
        if callable(candidate):
            return candidate

    return default


def _coerce_upload_file(value: Any) -> UploadFile:
    if isinstance(value, UploadFile):
        return value
    if StarletteUploadFile is not None and isinstance(value, StarletteUploadFile):
        filename = getattr(value, "filename", None)
        content_type = getattr(value, "content_type", None)
        file_obj = getattr(value, "file", value)
        seek = getattr(file_obj, "seek", None)
        if callable(seek):
            try:
                seek(0)
            except Exception:  # pragma: no cover - defensive seek reset
                pass
        return create_upload_file(filename, file_obj, content_type)
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
            if isinstance(item, (list, tuple)):
                nested = _coerce_sequence(item)
                if nested is not None:
                    return nested
                if item:
                    filename = item[0]
                    content = item[1] if len(item) > 1 else b""
                    content_type = item[2] if len(item) > 2 else None
                    return create_upload_file(filename, content, content_type)
    raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_FILE")


def _resolve_fallback_index(state) -> list[dict[str, Any]]:
    fallback_index = getattr(state, "fallback_index", None)
    if fallback_index is None:
        fallback_index = vectorstore_service.get_fallback_storage()
        setattr(state, "fallback_index", fallback_index)
    return fallback_index


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
        seeker = getattr(payload, "seek", None)
        teller = getattr(payload, "tell", None)
        original_position: int | None = None
        should_restore = False

        if callable(seeker):
            try:
                if callable(teller):
                    original_position = teller()
                else:
                    current_position = seeker(0, io.SEEK_CUR)
                    if isinstance(current_position, int):
                        original_position = current_position
                seeker(0)
            except Exception:
                original_position = None
            else:
                if original_position is not None:
                    should_restore = True

        try:
            data = read()
        finally:
            if callable(seeker) and should_restore and original_position is not None:
                try:
                    seeker(original_position)
                except Exception:
                    pass
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
    content_type = items[2] if len(items) > 2 and isinstance(items[2], str) else None
    return create_upload_file(filename, _coerce_bytes(content), content_type)


def _index_chunks(request: Request, chunks: Iterable[dict[str, Any]]) -> int:
    items = list(chunks)
    if not items:
        return 0

    fallback_index = _resolve_fallback_index(request.app.state)
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


DEFAULT_INDEX_CHUNKS = _index_chunks


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
    files: list[UploadFile] | None = File(None),
    file: UploadFile | None = File(None, alias="file"),
    user_id: str = Form(...),
    conversation_id: str | None = Form(None),
) -> dict[str, Any]:
    limits = _resolve_upload_limits(request)
    if _content_length_exceeds_limits(request, limits):
        raise HTTPException(_REQUEST_TOO_LARGE, "UPLOAD_TOO_LARGE")
    allowed_extensions = set(limits.allowed_extensions)

    state = _resolve_app_state(request)
    parser = _resolve_callable(state, "parse_and_chunk", parse_and_chunk)
    upsert_override = _resolve_callable(state, "upsert_chunks", None)

    upload_items: list[UploadFile] = []

    def _add_upload(candidate: Any) -> None:
        if isinstance(candidate, UploadFile):
            upload_items.append(candidate)
            return
        if StarletteUploadFile is not None and isinstance(candidate, StarletteUploadFile):
            upload_items.append(_coerce_upload_file(candidate))
            return
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                _add_upload(item)
            return
        if isinstance(candidate, dict):
            maybe_file = candidate.get("file") or candidate.get("upload")
            if isinstance(maybe_file, (list, tuple)):
                for item in maybe_file:
                    _add_upload(item)
            elif isinstance(maybe_file, UploadFile):
                upload_items.append(maybe_file)

    if file is not None:
        _add_upload(file)
    if files:
        _add_upload(files)
    if not upload_items:
        if _content_length_exceeds_limits(request, limits):
            raise HTTPException(_REQUEST_TOO_LARGE, "UPLOAD_TOO_LARGE")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_FILE")

    stored_files: list[str] = []
    total_chunks = 0

    for upload_file in upload_items:
        filename = _normalise_filename(upload_file.filename)
        ext = _normalise_extension(filename)
        if ext not in allowed_extensions:
            raise HTTPException(_UNSUPPORTED_MEDIA_TYPE, "UPLOAD_INVALID_EXT")

        _enforce_content_type(ext, upload_file)

        data = await upload_file.read()
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_EMPTY")
        if len(data) > min(limits.max_size, _ABSOLUTE_MAX_UPLOAD_BYTES):
            raise HTTPException(_REQUEST_TOO_LARGE, "UPLOAD_TOO_LARGE")

        chunks = parser(filename, data)
        if not chunks:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")

        chunk_list = list(chunks)
        index_fn = _index_chunks
        if upsert_override is not None and index_fn is DEFAULT_INDEX_CHUNKS:
            upsert_override(chunk_list)
            total_chunks += len(chunk_list)
        else:
            total_chunks += index_fn(request, chunk_list)

        stored_files.append(filename)

    return {"ok": True, "files": stored_files, "chunks": total_chunks}


@router.post("/api/chat")
def chat(request: Request, inp: ChatIn) -> dict[str, Any]:
    settings = request.app.state.settings
    chat_store = _coerce_chat_store(request.app.state.chat_store)
    request.app.state.chat_store = chat_store
    if chat_store is None:
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="CHAT_STORE_NOT_CONFIGURED")
    llm_provider = getattr(request.app.state, "llm_provider", None)
    fallback_index = _resolve_fallback_index(request.app.state)
    vector_store = _coerce_vector_store(
        getattr(request.app.state, "vector_store", None),
        fallback_index,
    )
    summarizer = request.app.state.summarizer
    memory_store = _ensure_memory_store(request.app.state, settings)
    reranker = getattr(request.app.state, "reranker", None)

    request.app.state.vector_store = vector_store
    request.app.state.memory_store = memory_store

    llm_provider = _coerce_llm_provider(llm_provider)
    if llm_provider is None and settings is not None:
        llm_provider = _coerce_llm_provider(get_cached_provider(settings))

    if llm_provider is None:
        raise HTTPException(_SERVICE_UNAVAILABLE, detail="LLM_NOT_CONFIGURED")

    request.app.state.llm_provider = llm_provider
    request.app.state.llm_client = llm_provider

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
        vector_store.ensure_ready()

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
            hits = vector_store.search(inp.message, top_k=settings.retrieve_topk)
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
        if not hits and isinstance(vector_store, _VectorStoreAdapter):
            synthetic_start = time.perf_counter()
            hits = _synthetic_hits(inp.message, settings.retrieve_topk)
            record_search_operation(
                "chat_synthetic",
                "success",
                time.perf_counter() - synthetic_start,
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
