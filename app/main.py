from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.chat.store import ChatStore, ConversationAccessError
from app.chat.summarizer import ConversationSummarizer
from app.ingest import parse_and_chunk
from app.memory.store import MemoryStore
from app.ollama_client import ensure_model, generate
from app.qdrant_client import ensure_collection, search_chunks
from app.rag.context import build_context

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(FILES_ROOT / "db" / "chat_history.sqlite")))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "12"))
CHAT_SUMMARY_TRIGGER = int(os.getenv("CHAT_SUMMARY_TRIGGER", "10"))
MIN_CITATIONS = max(1, int(os.getenv("CHAT_MIN_CITATIONS", "3")))
MAX_CITATIONS = max(MIN_CITATIONS, int(os.getenv("CHAT_MAX_CITATIONS", "5")))

chat_store = ChatStore(str(CHAT_DB_PATH))
summarizer = ConversationSummarizer(chat_store, generate)


def _init_memory_store() -> MemoryStore | None:
    enabled = os.getenv("MEMORY_ENABLED", "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    memory_db_path = Path(
        os.getenv("MEMORY_DB_PATH", str(FILES_ROOT / "db" / "memory.sqlite"))
    )
    ttl_days = int(os.getenv("MEMORY_TTL_DAYS", "90"))
    summary_trigger = int(os.getenv("MEMORY_SUMMARY_TRIGGER", str(CHAT_SUMMARY_TRIGGER)))
    max_tokens = int(os.getenv("MEMORY_MAX_TOKENS", "2000"))

    try:
        return MemoryStore(
            db_path=str(memory_db_path),
            ttl_days=ttl_days,
            summary_trigger=summary_trigger,
            max_tokens=max_tokens,
        )
    except Exception:  # pragma: no cover - defensive initialisation
        logger.exception("Failed to initialise memory store")
        return None


app.mem = _init_memory_store()

class ChatIn(BaseModel):
    user_id: str
    message: str
    conversation_id: str | None = None


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


@app.head("/health")
def health_head() -> JSONResponse:
    return health()


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _index_chunks(chunks: Iterable[dict[str, Any]]) -> int:
    items = list(chunks)
    if not items:
        return 0

    try:  # pragma: no cover - optional dependency initialisation
        ensure_collection()
    except Exception:
        logger.exception("Failed to ensure vector store; using fallback index")
        _FALLBACK_INDEX.extend(items)
        return len(items)

    try:  # pragma: no cover - optional dependency for full ingestion pipeline
        from app.qdrant_client import upsert_chunks  # type: ignore
    except Exception:  # pragma: no cover - gracefully degrade when unavailable
        _FALLBACK_INDEX.extend(items)
        logger.info("Stored %s chunks in fallback index", len(items))
        return len(items)

    upsert_chunks(items)
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


def _select_citations(hits: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    maximum = MAX_CITATIONS if MAX_CITATIONS >= MIN_CITATIONS else MIN_CITATIONS

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
        if len(unique) >= maximum:
            break

    has_minimum = len(unique) >= MIN_CITATIONS
    return unique, has_minimum


_FALLBACK_INDEX: list[dict[str, Any]] = []
app.document_index = _FALLBACK_INDEX


@app.post("/api/docs/upload")
async def upload(
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

    indexed = _index_chunks(chunks)
    return {"ok": True, "chunks": indexed}


@app.post("/api/chat")
def chat(
    inp: ChatIn,
) -> dict[str, Any]:
    ensure_model()
    ensure_collection()

    start = time.perf_counter()

    try:
        conversation_id = chat_store.ensure_conversation(inp.user_id, inp.conversation_id)
    except ConversationAccessError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(conversation_id, limit=CHAT_HISTORY_LIMIT)
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

    memory_text = ""
    active_memory = getattr(app, "mem", None)
    if isinstance(active_memory, MemoryStore):
        try:
            memory_text = active_memory.load_context(inp.user_id, conversation_id)
        except Exception:  # pragma: no cover - defensive lookup
            logger.exception("Failed to load memory context")
            memory_text = ""

    hits = search_chunks(inp.message, top_k=int(os.getenv("RETRIEVE_TOPK", "10")))
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
    prompt = "\n".join(part for part in prompt_parts if part is not None)

    answer = generate(prompt).strip()

    citations, has_minimum_citations = _select_citations(hits)

    chat_store.record_exchange(conversation_id, inp.message, answer)
    if chat_store.messages_since_summary(conversation_id) >= CHAT_SUMMARY_TRIGGER:
        summarizer.summarize(conversation_id)

    if isinstance(active_memory, MemoryStore):
        try:
            active_memory.record(inp.user_id, conversation_id, inp.message, answer)
        except Exception:  # pragma: no cover - defensive persistence handling
            logger.exception("Failed to persist memory entry")

    answer_text = answer
    if citations:
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


__all__ = ["app"]
