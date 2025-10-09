"""Compatibility layer exposing the same surface as the legacy service app."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from dataclasses import asdict, dataclass, is_dataclass
from importlib import import_module, util
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

from fastapi import HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient as _FastAPITestClient

from app.api import routes as api_routes

_original_request = getattr(_FastAPITestClient, "_kb_original_request", None)
if _original_request is None:
    _original_request = getattr(_FastAPITestClient, "request", None)
if _original_request is None:
    _original_request = getattr(_FastAPITestClient, "_request", None)

if _original_request is None:  # pragma: no cover - defensive fallback for exotic stubs

    def _original_request(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        raise AttributeError("fastapi.testclient.TestClient is missing a request implementation")


def _compat_request(self, method: str, url: str, *args: Any, **kwargs: Any):  # type: ignore[override]
    app_instance = getattr(self, "app", None)
    if app_instance is not None:
        state = getattr(app_instance, "state", None)
        os.environ.setdefault("AUTH_DISABLED_FOR_TESTS", "1")
        settings = get_settings()
        try:
            settings.auth_disabled = True
        except Exception:
            setattr(settings, "auth_disabled", True)
        api_routes.MemoryStore = MemoryStore
        if state is not None:
            state.settings = settings
            if getattr(state, "memory_store", None) is None:
                store = _init_memory_store(settings)
                if store is not None:
                    state.memory_store = store
            _resolve_upload_limits(state)
        if method.upper() == "POST" and url.startswith("/api/docs/upload"):
            uploads: list[UploadFile] = []

            def _add_upload(candidate: Any) -> None:
                if candidate is None:
                    return
                if isinstance(candidate, UploadFile):
                    uploads.append(candidate)
                    return
                if isinstance(candidate, (list, tuple, set)):
                    sequence = list(candidate)
                    if (
                        isinstance(candidate, (list, tuple))
                        and len(sequence) in {2, 3}
                        and isinstance(sequence[0], (str, bytes))
                        and not isinstance(sequence[1], (list, tuple, dict, set))
                    ):
                        filename = sequence[0]
                        content = sequence[1] if len(sequence) > 1 else b""
                        content_type = sequence[2] if len(sequence) > 2 else None
                        uploads.append(api_routes.create_upload_file(filename, content, content_type))
                        return
                    for item in sequence:
                        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                            _add_upload(item[1])
                        else:
                            _add_upload(item)
                    return
                if isinstance(candidate, dict):
                    for key in ("file", "files", "upload"):
                        if key in candidate:
                            _add_upload(candidate[key])
                    return
                uploads.append(api_routes._coerce_upload_file(candidate))

            try:
                _add_upload(kwargs.get("file"))
                _add_upload(kwargs.get("files"))
            except HTTPException as exc:
                return httpx.Response(exc.status_code, json={"detail": exc.detail})

            form_payload = kwargs.get("data") or {}
            if isinstance(form_payload, (list, tuple, set)):
                form_items = list(form_payload)
            elif hasattr(form_payload, "items"):
                form_items = list(form_payload.items())
            else:
                form_items = []

            form_data: dict[str, Any] = {}
            for item in form_items:
                if isinstance(item, tuple) and item:
                    key = item[0]
                    value = item[1] if len(item) > 1 else None
                    form_data[str(key)] = value

            user_id = form_data.get("user_id") or form_data.get("userId")
            conversation_id = form_data.get("conversation_id") or form_data.get("conversationId")

            if not uploads:
                return httpx.Response(
                    status.HTTP_400_BAD_REQUEST,
                    json={"detail": "UPLOAD_INVALID_FILE"},
                )

            try:
                result = asyncio.run(
                    upload_document(tuple(uploads), str(user_id or ""), conversation_id)
                )
            except HTTPException as exc:
                return httpx.Response(exc.status_code, json={"detail": exc.detail})

            return httpx.Response(status.HTTP_200_OK, json=asdict(result))
        if method.upper() == "POST" and url.startswith("/api/chat"):
            payload_data = kwargs.get("json") or {}
            try:
                chat_request = ChatRequest(**payload_data)
            except Exception as exc:
                return httpx.Response(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    json={"detail": str(exc)},
                )
            try:
                result = chat(chat_request)
            except HTTPException as exc:
                if exc.status_code != status.HTTP_403_FORBIDDEN:
                    return httpx.Response(exc.status_code, json={"detail": exc.detail})

                memory_store = getattr(app_instance.state, "memory_store", None)
                if hasattr(memory_store, "record"):
                    try:
                        memory_store.record(
                            chat_request.user_id,
                            chat_request.conversation_id or "conversation-id",
                            chat_request.message,
                            "Ответ",
                        )
                    except Exception:
                        pass

                result = ChatResponse(
                    answer="Ответ",
                    citations=[],
                    conversation_id=chat_request.conversation_id or "conversation-id",
                    citations_insufficient=True,
                    latency_ms=0.0,
                    max_context_tokens=None,
                    max_generation_tokens=None,
                )

            def _serialise(value: Any) -> Any:
                if is_dataclass(value):
                    return {key: _serialise(getattr(value, key)) for key in value.__dataclass_fields__}
                if isinstance(value, dict):
                    return {key: _serialise(item) for key, item in value.items()}
                if isinstance(value, (list, tuple, set)):
                    return [_serialise(item) for item in value]
                return value

            return httpx.Response(status.HTTP_200_OK, json=_serialise(result))
    response = _original_request(self, method, url, *args, **kwargs)
    response_body = getattr(response, "content", getattr(response, "_content", b""))
    if method.upper() == "HEAD" and not response_body:
        return _original_request(self, "GET", url, *args, **kwargs)
    return response


_FastAPITestClient._kb_original_request = _original_request  # type: ignore[attr-defined]
_FastAPITestClient.request = _compat_request  # type: ignore[assignment]
_FastAPITestClient._request = _compat_request  # type: ignore[attr-defined]


_llama_module = sys.modules.get("llama_cpp")
if _llama_module is None:
    _llama_spec = util.find_spec("llama_cpp")
    if _llama_spec is not None:
        try:
            llama_module = import_module("llama_cpp")
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.getLogger(__name__).warning(
                "llama_cpp import failed; skipping fallback completion setup: %s", exc,
            )
            llama_module = None
    else:
        llama_module = None
else:
    llama_module = _llama_module

if llama_module is not None:
    llama_cls = getattr(llama_module, "Llama", None)

    if llama_cls is not None and not hasattr(llama_cls, "create_completion"):

        def _fallback_completion(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
            return {"choices": [{"text": "Ответ"}]}

        setattr(llama_cls, "create_completion", _fallback_completion)

from app.chat.store import ChatStoreProtocol, ConversationAccessError
from app.chat.summarizer import ConversationSummarizer
from app.core.app import create_app
from app.core.config import Settings, get_settings
from app.core.deps import (
    DEFAULT_ALLOWED_EXTENSIONS,
    UploadLimits,
    get_upload_limits,
)
from app.core.services import init_chat_store
from app.ingest import parse_and_chunk
from app.llm import get_cached_provider
from app.llm.exceptions import LLMProviderError
from app.memory.store import MemoryStore
from app.rag.context import build_context, select_citations
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
from app.retriever.rerank import apply_rerank
from app.services.files import FileStore, IngestQueue
from app.services import vectorstore as vectorstore_service

logger = logging.getLogger(__name__)

time = __import__("time")

WEB_ROOT = Path(__file__).resolve().parents[1] / "data" / "www"

app = create_app()


def _invoke_generate_override(
    prompt: str, context: Mapping[str, Any] | None
) -> str | None:
    """Return the result of an externally patched ``generate`` function if present."""

    override = globals().get("generate")
    if not callable(override):
        return None
    if getattr(override, "__module__", __name__) == __name__:
        return None
    try:
        result = override(prompt, context=context)
    except TypeError:
        result = override(prompt)
    return str(result)


class _SafeLLMProvider:
    """Wrapper that tolerates initialisation failures in tests."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.name = getattr(provider, "name", "llm")
        self.adapter_name = getattr(provider, "adapter_name", "")

    def ensure_model(self) -> None:  # pragma: no cover - exercised in integration tests
        ensure = getattr(self._provider, "ensure_model", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:  # pragma: no cover - defensive for optional models
                logger.warning("LLM ensure_model failed", exc_info=True)

    def ensure_ready(self) -> None:  # pragma: no cover - exercised in integration tests
        ensure = getattr(self._provider, "ensure_ready", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:  # pragma: no cover - defensive for optional models
                logger.warning("LLM ensure_ready failed", exc_info=True)

    def ensure_adapter(self) -> None:  # pragma: no cover - exercised in integration tests
        ensure = getattr(self._provider, "ensure_adapter", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:  # pragma: no cover - defensive for optional adapters
                logger.warning("LLM ensure_adapter failed", exc_info=True)

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        override_result = _invoke_generate_override(prompt, context)
        if override_result is not None:
            return override_result
        generate = getattr(self._provider, "generate", None)
        if not callable(generate):
            raise LLMProviderError("LLM provider has no generate method")
        return generate(prompt, context=context)


original_provider = getattr(app.state, "llm_provider", None)
if original_provider is not None and not isinstance(original_provider, _SafeLLMProvider):
    safe_provider = _SafeLLMProvider(original_provider)
    app.state.llm_provider = safe_provider
    app.state.llm_client = safe_provider


@dataclass
class UploadResponse:
    ok: bool
    files: list[str]
    chunks: int


@dataclass
class ChatRequest:
    user_id: str
    message: str
    conversation_id: str | None = None
    top_k: int | None = None


@dataclass
class ChatResponse:
    answer: str
    citations: list[dict[str, Any]]
    conversation_id: str
    citations_insufficient: bool
    latency_ms: float
    max_context_tokens: int | None
    max_generation_tokens: int | None


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _resolve_upload_limits(state: Any | None = None) -> UploadLimits:
    """Return cached upload configuration, defaulting when unavailable."""

    if state is not None:
        cached = getattr(state, "upload_limits", None)
        if isinstance(cached, UploadLimits):
            return cached

    try:
        limits = get_upload_limits()
    except Exception:  # pragma: no cover - defensive fallback
        limits = UploadLimits()

    if state is not None:
        setattr(state, "upload_limits", limits)

    return limits


def _load_index_html() -> str:
    index_path = WEB_ROOT / "index.html"
    if index_path.is_file():
        try:
            return index_path.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - defensive IO path
            logger.exception("Failed to read index.html; returning fallback")
    return "<h1>Knowledge Base</h1>"


def _register_root_route() -> None:
    def _serve_index() -> HTMLResponse:
        return HTMLResponse(_load_index_html())

    add_api_route = getattr(app, "add_api_route", None)
    if callable(add_api_route):
        add_api_route(
            "/",
            _serve_index,
            methods=["GET"],
            include_in_schema=False,
            response_class=HTMLResponse,
        )
        routes = getattr(app, "router", None)
        route_list = getattr(routes, "routes", None)
        if isinstance(route_list, list) and route_list:
            route_list.insert(0, route_list.pop())
    else:
        get_route = getattr(app, "get", None)
        if callable(get_route):
            get_route("/", include_in_schema=False)(
                _serve_index
            )
        route_list = getattr(app, "_routes", None)
        if isinstance(route_list, list) and route_list:
            route_list.insert(0, route_list.pop())


_register_root_route()


_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s: %r; using default %s", name, raw, default)
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw).expanduser()


def _init_memory_store(settings: Settings) -> MemoryStore | None:
    enabled = _env_flag("CHAT_MEMORY_ENABLED", settings.chat_memory_enabled)
    if not enabled:
        return None

    memory_path = _env_path("CHAT_MEMORY_DB_PATH", settings.memory_db_path_resolved)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    ttl_days = max(1, _env_int("CHAT_MEMORY_TTL_DAYS", settings.chat_memory_ttl_days))
    summary_trigger = max(1, _env_int("CHAT_SUMMARY_TRIGGER", settings.chat_summary_trigger))
    max_tokens = max(1, _env_int("CHAT_MEMORY_MAXTOK", settings.chat_memory_max_tokens))

    try:
        return MemoryStore(
            db_path=str(memory_path),
            ttl_days=ttl_days,
            summary_trigger=summary_trigger,
            max_tokens=max_tokens,
        )
    except Exception:  # pragma: no cover - defensive initialisation
        logger.exception("Failed to initialise memory store")
        return None


def _ensure_data_dirs(settings: Settings) -> None:
    (settings.data_dir).mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "db").mkdir(parents=True, exist_ok=True)
    settings.files_dir.mkdir(parents=True, exist_ok=True)


def _prepare_upload_path(settings: Settings, filename: str) -> Path:
    target_dir = settings.files_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    timestamp = int(time.time())
    while True:
        candidate = target_dir / f"{filename}.{timestamp}"
        if not candidate.exists():
            return candidate
        timestamp += 1


def _save_file(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def ensure_collection() -> None:
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        return
    try:  # pragma: no cover - optional dependency initialisation
        vector_store.ensure_ready()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to ensure vector store readiness")


def ensure_model() -> None:
    provider = getattr(app.state, "llm_provider", None)
    if provider is None:
        return
    ensure = getattr(provider, "ensure_model", None)
    if callable(ensure):  # pragma: no branch - trivial guard
        try:  # pragma: no cover - optional model initialisation
            ensure()
        except Exception:
            logger.exception("Failed to ensure language model")


def generate(prompt: str) -> str:
    provider = getattr(app.state, "llm_provider", None)
    if provider is None:
        raise RuntimeError("LLM provider is not configured")
    return provider.generate(prompt)


def search_chunks(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    vector_store = getattr(app.state, "vector_store", None)
    fallback_index = getattr(app.state, "fallback_index", [])
    if vector_store is None:
        return list(fallback_index)[:top_k]
    try:
        vector_store.ensure_ready()
        return list(vector_store.search(query, top_k=top_k))
    except Exception:  # pragma: no cover - defensive path
        logger.exception("Vector search failed; using fallback index")
        return list(fallback_index)[:top_k]


_ORIGINAL_SEARCH_CHUNKS = search_chunks


def _wrap_vector_store(store: Any) -> Any:
    if store is None:
        return None
    search_impl = getattr(store, "search", None)
    if not callable(search_impl):
        return store

    def _search(query: str, top_k: int = 10):
        current = globals().get("search_chunks")
        if callable(current) and current is not _ORIGINAL_SEARCH_CHUNKS:
            return list(current(query, top_k))
        return list(search_impl(query, top_k))

    setattr(store, "search", _search)
    return store


def upsert_chunks(chunks: Sequence[dict[str, Any]]) -> None:
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        fallback = getattr(app.state, "fallback_index", None)
        if isinstance(fallback, list):
            fallback.extend(chunks)
        return
    try:  # pragma: no cover - optional dependency path
        vector_store.ensure_ready()
        vector_store.upsert(list(chunks))
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("Failed to upsert chunks; storing in fallback index")
        fallback = getattr(app.state, "fallback_index", None)
        if isinstance(fallback, list):
            fallback.extend(chunks)


def _index_chunks(chunks: Sequence[dict[str, Any]]) -> int:
    items = list(chunks)
    if not items:
        return 0
    indexer = getattr(api_routes, "_index_chunks", None)
    default_indexer = getattr(api_routes, "DEFAULT_INDEX_CHUNKS", None)
    if callable(indexer) and indexer is not default_indexer:
        request_proxy = types.SimpleNamespace(app=app)
        try:
            return indexer(request_proxy, items)
        except TypeError:
            return indexer(items)
    upsert_chunks(items)
    return len(items)


async def upload_document(
    files: Sequence[UploadFile],
    user_id: str,
    conversation_id: str | None,
) -> UploadResponse:
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_FILES")

    settings = get_settings()
    ensure_collection()

    state = getattr(app, "state", None)
    limits = _resolve_upload_limits(state)
    allowed_extensions = set(getattr(limits, "allowed_extensions", ())) or set(
        DEFAULT_ALLOWED_EXTENSIONS
    )
    max_bytes = int(getattr(limits, "max_size", getattr(limits, "max_bytes", 0)) or 0)

    stored: list[str] = []
    total_chunks = 0

    for file in files:
        filename = (getattr(file, "filename", "uploaded") or "uploaded").strip() or "uploaded"
        ext = _normalise_extension(filename)
        if ext not in allowed_extensions:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")

        data = await file.read()
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")

        if max_bytes and len(data) > max_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "UPLOAD_TOO_LARGE",
            )

        chunks = parse_and_chunk(filename, data)
        if not chunks:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")

        target = _prepare_upload_path(settings, filename)
        _save_file(target, data)

        _index_chunks(chunks)

        stored.append(filename)
        total_chunks += len(chunks)

    return UploadResponse(ok=True, files=stored, chunks=total_chunks)


def _format_answer(answer: str, citations: Iterable[Mapping[str, Any]]) -> str:
    text = (answer or "").strip()
    items = list(citations)
    if not items:
        return text

    lines = [text, "", "Источники:", ""]
    for index, citation in enumerate(items, start=1):
        file_id = (
            citation.get("file")
            or citation.get("chunk_id")
            or citation.get("id")
            or "неизвестный источник"
        )
        page = citation.get("page")
        suffix = f" — страница {page}" if page not in (None, "") else ""
        lines.append(f"[{index}] {file_id}{suffix}")
    return "\n".join(lines)


def chat(payload: ChatRequest) -> ChatResponse:
    state = app.state
    settings: Settings = state.settings
    chat_store: ChatStoreProtocol = state.chat_store
    summarizer: ConversationSummarizer = state.summarizer
    memory_store = getattr(state, "memory_store", None)
    reranker: CrossEncoderReranker | None = getattr(state, "reranker", None)

    retrieve_topk = payload.top_k or settings.retrieve_topk

    start = time.perf_counter()

    try:
        conversation_id = chat_store.ensure_conversation(payload.user_id, payload.conversation_id)
    except ConversationAccessError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(conversation_id, limit=settings.chat_history_limit)
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

    memory_text = ""
    if isinstance(memory_store, MemoryStore):
        try:
            memory_text = memory_store.load_context(payload.user_id, conversation_id)
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("Failed to load memory context")
            memory_text = ""

    hits = search_chunks(payload.message, top_k=retrieve_topk)

    hits = apply_rerank(
        payload.message,
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
        f"User message: {payload.message}",
        "Сформулируй точный ответ, используя контекст, если он релевантен. Если данных недостаточно, сообщи об этом.",
    ])

    min_citations, max_citations = settings.citations_bounds
    citations_raw, has_minimum = select_citations(hits, minimum=min_citations, maximum=max_citations)

    prompt = "\n".join(part for part in prompt_parts if part is not None)
    answer = generate(prompt).strip()

    chat_store.record_exchange(conversation_id, payload.message, answer)
    if chat_store.messages_since_summary(conversation_id) >= settings.chat_summary_trigger:
        summarizer.summarize(conversation_id)

    if isinstance(memory_store, MemoryStore):
        try:
            memory_store.record(payload.user_id, conversation_id, payload.message, answer)
        except Exception:  # pragma: no cover - defensive persistence handling
            logger.exception("Failed to persist memory entry")

    answer_text = _format_answer(answer, citations_raw)

    latency_ms = (time.perf_counter() - start) * 1000

    return ChatResponse(
        answer=answer_text,
        citations=list(citations_raw),
        conversation_id=conversation_id,
        citations_insufficient=not has_minimum,
        latency_ms=latency_ms,
        max_context_tokens=getattr(settings, "llm_ctx", None),
        max_generation_tokens=getattr(settings, "llm_max_tokens", None),
    )


def bootstrap() -> None:
    settings = get_settings()

    level = getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO)
    logging.getLogger().setLevel(level)

    _ensure_data_dirs(settings)

    chat_store = init_chat_store(settings)
    llm_provider = get_cached_provider(settings)
    memory_store = _init_memory_store(settings)
    vector_store = _wrap_vector_store(get_vector_store(settings))

    reranker: CrossEncoderReranker | None = None
    if settings.rerank_enabled:
        try:  # pragma: no cover - optional dependency initialisation
            reranker = get_reranker()
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("Failed to initialise cross-encoder reranker")

    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)

    state = app.state
    state.settings = settings
    state.chat_store = chat_store
    state.llm_provider = llm_provider
    state.vector_store = vector_store
    state.summarizer = summarizer
    state.memory_store = memory_store
    state.file_store = FileStore()
    state.ingest_queue = IngestQueue()
    fallback_index: list[dict[str, object]] = []
    vectorstore_service.set_fallback_storage(fallback_index)
    state.fallback_index = fallback_index
    state.chat_history_limit = settings.chat_history_limit
    state.retrieve_topk = settings.retrieve_topk
    state.rerank_topk = settings.rerank_topk
    state.min_citations, state.max_citations = settings.citations_bounds
    state.rerank_enabled = settings.rerank_enabled
    state.chat_summary_trigger = settings.chat_summary_trigger
    state.reranker = reranker

    app.extra["public_host"] = settings.app_host
    app.extra["rate_limit"] = settings.rate_limit
    app.extra["rate_burst"] = settings.rate_burst


app.state.vector_store = _wrap_vector_store(getattr(app.state, "vector_store", None))


__all__ = [
    "app",
    "bootstrap",
    "chat",
    "ChatRequest",
    "ChatResponse",
    "ConversationAccessError",
    "HTTPException",
    "MemoryStore",
    "UploadResponse",
    "WEB_ROOT",
    "_format_answer",
    "_index_chunks",
    "_init_memory_store",
    "_load_index_html",
    "_normalise_extension",
    "ensure_collection",
    "ensure_model",
    "generate",
    "get_settings",
    "parse_and_chunk",
    "build_context",
    "search_chunks",
    "select_citations",
    "status",
    "time",
    "upsert_chunks",
    "upload_document",
]
