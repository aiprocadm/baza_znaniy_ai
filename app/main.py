from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
        codex/add-post-/api/chat-endpoint
from sqlalchemy.orm import Session

from app.auth import (
    get_current_user as _auth_get_current_user,
    require_admin,
    require_staff,
    router as auth_router,
    setup_defaults,
)
from app.db.models import ChatLog, User
from app.db.session import get_session, init_db
from app.chat.store import ChatStore, ConversationAccessError
from app.chat.summarizer import ConversationSummarizer

from app.memory.store import MemoryStore
        codex/create-qdrant-and-ingest-modules-kfqtqh

        codex/create-qdrant-and-ingest-modules
        main
from app.ollama_client import ensure_model, generate
from app.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.ingest import parse_and_chunk
from app.security import create_access_token, verify_password

logger = logging.getLogger(__name__)

        main
from app.models.ollama_client import ensure_model, generate
from app.models.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.rag.ingest import parse_and_chunk
        main

app = FastAPI(title="kb")

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(FILES_ROOT / "db" / "chat_history.sqlite")))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "12"))
CHAT_SUMMARY_TRIGGER = int(os.getenv("CHAT_SUMMARY_TRIGGER", "10"))

chat_store = ChatStore(str(CHAT_DB_PATH))
summarizer = ConversationSummarizer(chat_store, lambda prompt: generate(prompt))

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


@app.post("/api/docs/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    conversation_id: str | None = Form(None),
) -> dict[str, Any]:
    name = (file.filename or "uploaded").strip()
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in {"pdf", "docx", "txt"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")

    data = await file.read()
    ensure_collection()
    chunks = parse_and_chunk(name, data)
    if not chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")
    upsert_chunks(chunks)
    return {"ok": True, "chunks": len(chunks)}


@app.post("/api/chat")
def chat(
    inp: ChatIn,
) -> dict[str, Any]:
    ensure_model()
    ensure_collection()

    start = time.perf_counter()
        codex/add-post-/api/chat-endpoint
    if str(user.id) != inp.user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_MISMATCH")

    try:
        conversation_id = chat_store.ensure_conversation(inp.user_id, inp.conversation_id)
    except ConversationAccessError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(conversation_id, limit=CHAT_HISTORY_LIMIT)
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

    memory_key = inp.user_id
    memory_text = mem.load_context(memory_key, inp.conversation_id) if MEMORY_ENABLED else ""
        main

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
    prompt_parts.extend([
        "Retrieved context:",
        context or "(нет подходящего контекста)",
        "",
        f"User message: {inp.message}",
        "Сформулируй точный ответ, используя контекст, если он релевантен. Если данных недостаточно, сообщи об этом.",
    ])
    prompt = "\n".join(part for part in prompt_parts if part is not None)

    answer = generate(prompt).strip()

    selected_hits, has_minimum_citations = select_citations(hits, minimum=3, maximum=5)
    citations = [
        {"file": hit.get("file"), "page": hit.get("page"), "score": float(hit.get("score", 0.0))}
        for hit in selected_hits
    ]
    citations_insufficient = not has_minimum_citations

    chat_store.record_exchange(conversation_id, inp.message, answer)
    if chat_store.messages_since_summary(conversation_id) >= CHAT_SUMMARY_TRIGGER:
        summarizer.summarize(conversation_id)

    latency_ms = (time.perf_counter() - start) * 1000

    summary = answer.strip()
    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."

        codex/add-post-/api/chat-endpoint
    log = ChatLog(
        user_id=user.id,
        conversation_id=conversation_id,
        question=inp.message,
        response_summary=summary,
        citations=citations,
        latency_ms=latency_ms,
    )
    try:
        db.add(log)
        db.commit()
    except Exception:  # pragma: no cover - defensive persistence handling
        db.rollback()
        logger.exception("Failed to persist chat log")

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

    response: dict[str, Any] = {
        "answer": answer_text,
        "citations": citations,
        "conversation_id": conversation_id,
    }

    response: dict[str, Any] = {"answer": answer, "citations": citations}
        main
    if citations_insufficient:
        response["citations_insufficient"] = True

    return response


__all__ = ["app"]
