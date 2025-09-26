from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import router as auth_router, require_active_user, require_admin, setup_defaults
from app.db.models import ChatLog, User
from app.db.session import get_session, init_db
from app.memory.store import MemoryStore
from app.models.ollama_client import ensure_model, generate
from app.models.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.rag.ingest import parse_and_chunk

FILES_ROOT = os.getenv("FILES_ROOT", "/opt/knowlab/data/files")
DB_PATH = os.path.join(FILES_ROOT, "db", "kb.sqlite")
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() == "true"

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")
app.include_router(auth_router)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

mem = MemoryStore(
    db_path=DB_PATH,
    ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS", "90")),
    summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER", "10")),
    max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK", "2000")),
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    setup_defaults()


@app.get("/health", methods=["GET", "HEAD"])
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


DatabaseSession = Annotated[Session, Depends(get_session)]
AuthenticatedUser = Annotated[User, Depends(require_active_user)]


@app.post("/api/docs/upload")
async def upload(
    file: UploadFile = File(...),
    user: Annotated[User, Depends(require_admin)] = None,
) -> dict[str, Any]:
    del user  # access already enforced by dependency

    name = file.filename or ""
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
    user: AuthenticatedUser,
    db: DatabaseSession,
) -> dict[str, Any]:
    ensure_model()
    ensure_collection()
    start = time.perf_counter()

    memory_key = str(user.id)
    memory_text = mem.load_context(memory_key, inp.conversation_id) if MEMORY_ENABLED else ""
    hits = search_chunks(inp.message, top_k=int(os.getenv("RETRIEVE_TOPK", "24")))
    context = build_context(hits, token_limit=3000)

    prompt = "\n".join(
        [
            "You are a helpful assistant providing concise answers based on documentation.",
            "Context:",
            context,
            "",
            "Memory:",
            memory_text,
            "",
            f"Question: {inp.message}",
        ]
    )
    answer = generate(prompt)

    selected_hits = select_citations(hits, minimum=3, maximum=5)
    citations = [
        {"file": hit.get("file"), "page": hit.get("page"), "score": float(hit.get("score", 0.0))}
        for hit in selected_hits
    ]

    if MEMORY_ENABLED:
        mem.record(memory_key, inp.conversation_id, inp.message, answer)

    latency_ms = (time.perf_counter() - start) * 1000
    summary = answer.strip()
    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."

    log = ChatLog(
        user_id=str(user.id),
        conversation_id=inp.conversation_id,
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

    return {"answer": answer, "citations": citations}


@app.get("/admin/chat-logs", response_class=HTMLResponse)
def chat_logs(
    request: Request,
    db: DatabaseSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str | None = Query(None),
    conversation_id: str | None = Query(None),
    _: Annotated[User, Depends(require_admin)] = None,
) -> HTMLResponse:
    query = db.query(ChatLog)
    if user_id:
        query = query.filter(ChatLog.user_id == user_id)
    if conversation_id:
        query = query.filter(ChatLog.conversation_id == conversation_id)

    total = query.count()
    pages = math.ceil(total / page_size) if total else 1
    logs = (
        query.order_by(ChatLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return templates.TemplateResponse(
        "chat_logs.html",
        {
            "request": request,
            "logs": logs,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": pages,
            "user_id": user_id or "",
            "conversation_id": conversation_id or "",
        },
    )


__all__ = ["app"]
