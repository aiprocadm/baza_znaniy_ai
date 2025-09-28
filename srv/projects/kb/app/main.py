"""FastAPI entry point for the knowledge base service."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .store import ChatStore, ConversationAccessError
from .summarizer import ConversationSummarizer
from .config import get_settings
from .ingest import parse_and_chunk
from .memory import MemoryStore
from .models import ChatRequest, ChatResponse, UploadResponse
from .ollama_client import ensure_model, generate
from .qdrant_client import ensure_collection, search_chunks, upsert_chunks
from .rag import build_context, select_citations

load_dotenv()

logger = logging.getLogger("kb.service")

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_ROOT = BASE_DIR / "data" / "www"

app = FastAPI(title="Knowledge Base API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_directories(base: Path) -> None:
    (base / "files").mkdir(parents=True, exist_ok=True)
    (base / "db").mkdir(parents=True, exist_ok=True)


def _get_env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _init_memory_store(settings) -> MemoryStore | None:
    enabled_value = _get_env_value("CHAT_MEMORY_ENABLED", "MEMORY_ENABLED", default="")
    enabled = str(enabled_value or "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    db_path_value = _get_env_value(
        "CHAT_MEMORY_DB_PATH", "MEMORY_DB_PATH", default=str(settings.memory_database)
    )
    ttl_value = _get_env_value("CHAT_MEMORY_TTL_DAYS", "MEMORY_TTL_DAYS")
    trigger_value = _get_env_value("CHAT_SUMMARY_TRIGGER", "MEMORY_SUMMARY_TRIGGER")
    max_tokens_value = _get_env_value("CHAT_MEMORY_MAXTOK", "MEMORY_MAX_TOKENS")

    ttl_days = int(ttl_value) if ttl_value else settings.chat_memory_ttl_days
    summary_trigger = int(trigger_value) if trigger_value else settings.chat_summary_trigger
    max_tokens = int(max_tokens_value) if max_tokens_value else settings.chat_memory_max_tokens

    try:
        return MemoryStore(
            db_path=str(Path(db_path_value or settings.memory_database)),
            ttl_days=ttl_days,
            summary_trigger=summary_trigger,
            max_tokens=max_tokens,
        )
    except Exception:  # pragma: no cover - defensive initialisation
        logger.exception("Failed to initialise memory store")
        return None


@app.on_event("startup")
def bootstrap() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    _ensure_directories(settings.data_dir)

    app.state.chat_store = ChatStore(
        str(settings.chat_database),
        secret=settings.app_secret or None,
    )
    app.state.summarizer = ConversationSummarizer(app.state.chat_store, generate)
    app.state.memory_store = _init_memory_store(settings)
    app.extra.update(
        {
            "public_host": settings.app_host,
            "rate_limit": settings.rate_limit,
            "rate_burst": settings.rate_burst,
        }
    )


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _save_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _index_chunks(chunks: Iterable[dict[str, Any]]) -> int:
    chunk_list = list(chunks)
    if not chunk_list:
        return 0
    upsert_chunks(chunk_list)
    return len(chunk_list)


def _health_response() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


@app.get("/health", response_class=JSONResponse, tags=["system"])
def health() -> JSONResponse:
    return _health_response()


@app.head("/health", response_class=JSONResponse, tags=["system"])
def health_head() -> JSONResponse:
    return _health_response()


@app.post("/api/docs/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    conversation_id: str | None = Form(None),
) -> UploadResponse:
    settings = get_settings()
    target_dir = settings.files_dir
    ensure_collection()

    processed: List[str] = []
    total_chunks = 0

    for file in files:
        filename = (file.filename or "uploaded").strip()
        ext = _normalise_extension(filename)
        if ext not in {"pdf", "docx", "txt"}:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")

        data = await file.read()
        if not data:
            continue

        chunks = parse_and_chunk(filename, data)
        if not chunks:
            continue

        safe_name = filename or f"upload-{int(time.time())}.txt"
        save_path = target_dir / safe_name
        if save_path.exists():
            suffix = int(time.time())
            save_path = target_dir / f"{safe_name}.{suffix}"
        _save_file(save_path, data)
        indexed = _index_chunks(chunks)
        if indexed:
            processed.append(filename)
            total_chunks += indexed

    if not processed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")

    return UploadResponse(ok=True, files=processed, chunks=total_chunks)


def _format_answer(answer: str, citations: List[dict[str, Any]]) -> str:
    if not citations:
        return answer.strip()

    formatted: List[str] = []
    for idx, citation in enumerate(citations, start=1):
        location = citation.get("page")
        if location is None:
            formatted.append(f"[{idx}] {citation.get('file', 'неизвестный источник')}")
        else:
            formatted.append(
                f"[{idx}] {citation.get('file', 'неизвестный источник')} — страница {location}"
            )
    return "\n\n".join([answer.strip(), "Источники:", "\n".join(formatted)])


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest) -> ChatResponse:
    settings = get_settings()
    ensure_model()
    ensure_collection()

    chat_store: ChatStore = app.state.chat_store
    summarizer: ConversationSummarizer = app.state.summarizer
    memory_store: MemoryStore | None = getattr(app.state, "memory_store", None)

    try:
        conversation_id = chat_store.ensure_conversation(payload.user_id, payload.conversation_id)
    except ConversationAccessError as exc:  # pragma: no cover - access control
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CONVERSATION_FORBIDDEN") from exc

    start = time.perf_counter()

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(conversation_id, limit=settings.chat_history_limit)
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

    memory_text = ""
    if isinstance(memory_store, MemoryStore):
        try:
            memory_text = memory_store.load_context(payload.user_id, conversation_id)
        except Exception:  # pragma: no cover - defensive lookup
            logger.exception("Failed to load memory context")
            memory_text = ""

    hits = search_chunks(payload.message, top_k=settings.retrieve_topk)
    if settings.rerank_topk and settings.rerank_topk > 0:
        hits = list(hits)[: settings.rerank_topk]
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
    prompt = "\n".join(part for part in prompt_parts if part is not None)

    answer = generate(prompt).strip()

    minimum = max(1, settings.chat_min_citations)
    maximum = max(minimum, settings.chat_max_citations)
    citations, has_minimum = select_citations(hits, minimum=minimum, maximum=maximum)

    chat_store.record_exchange(conversation_id, payload.message, answer)
    if chat_store.messages_since_summary(conversation_id) >= settings.chat_summary_trigger:
        summarizer.summarize(conversation_id)

    if isinstance(memory_store, MemoryStore):
        try:
            memory_store.record(payload.user_id, conversation_id, payload.message, answer)
        except Exception:  # pragma: no cover - defensive persistence
            logger.exception("Failed to persist memory entry")

    answer_text = _format_answer(answer, citations)
    latency_ms = (time.perf_counter() - start) * 1000

    return ChatResponse(
        answer=answer_text,
        citations=citations,
        conversation_id=conversation_id,
        citations_insufficient=not has_minimum,
        latency_ms=latency_ms,
    )


def _load_index_html() -> str:
    index_path = WEB_ROOT / "index.html"
    try:
        return index_path.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover - defensive fallback
        return "<h1>Knowledge Base</h1>"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(_load_index_html())


__all__ = ["app"]
