"""Compatibility layer exposing the same surface as the legacy service app."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastapi import HTTPException, UploadFile, status

from app.chat.store import ChatStoreProtocol, ConversationAccessError
from app.chat.summarizer import ConversationSummarizer
from app.core.app import create_app
from app.core.config import Settings, get_settings
from app.core.services import init_chat_store
from app.ingest import parse_and_chunk
from app.llm import get_cached_provider
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


def _load_index_html() -> str:
    index_path = WEB_ROOT / "index.html"
    if index_path.is_file():
        try:
            return index_path.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - defensive IO path
            logger.exception("Failed to read index.html; returning fallback")
    return "<h1>Knowledge Base</h1>"


def _init_memory_store(settings: Settings) -> MemoryStore | None:
    if not settings.chat_memory_enabled:
        return None

    memory_path = settings.memory_db_path_resolved
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return MemoryStore(
            db_path=str(memory_path),
            ttl_days=settings.chat_memory_ttl_days,
            summary_trigger=settings.chat_summary_trigger,
            max_tokens=settings.chat_memory_max_tokens,
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

    stored: list[str] = []
    total_chunks = 0

    for file in files:
        filename = (getattr(file, "filename", "uploaded") or "uploaded").strip() or "uploaded"
        ext = _normalise_extension(filename)
        if ext not in {"pdf", "docx", "txt"}:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")

        data = await file.read()
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")

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
    vector_store = get_vector_store(settings)

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
