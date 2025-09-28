from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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
from app.memory.store import MemoryStore
from app.ollama_client import ensure_model, generate
from app.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.ingest import parse_and_chunk
from app.security import create_access_token, verify_password

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")
app.include_router(auth_router)

get_current_user = _auth_get_current_user

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
DB_PATH = FILES_ROOT / "db" / "kb.sqlite"
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() == "true"

mem = MemoryStore(
    db_path=str(DB_PATH),
    ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS", "90")),
    summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER", "10")),
    max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK", "2000")),
)

SessionDep = Annotated[Session, Depends(get_session)]


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    must_change_password: bool


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    setup_defaults()


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


@app.head("/health")
def health_head() -> JSONResponse:
    return health()


@app.post("/api/auth/token", response_model=TokenOut)
def login(db: SessionDep, form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> TokenOut:
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")

    token = create_access_token({"sub": str(user.id)})
    return TokenOut(access_token=token, must_change_password=user.must_change_password)


@app.post("/api/docs/upload")
async def upload(
    file: UploadFile = File(...),
    _: Annotated[User, Depends(require_admin)] = None,
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
    user: Annotated[User, Depends(require_staff)],
    db: SessionDep,
) -> dict[str, Any]:
    ensure_model()
    ensure_collection()

    start = time.perf_counter()
    memory_key = str(user.id)
    memory_text = mem.load_context(memory_key, inp.conversation_id) if MEMORY_ENABLED else ""

    hits = search_chunks(inp.message, top_k=int(os.getenv("RETRIEVE_TOPK", "10")))
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

    selected_hits, has_minimum_citations = select_citations(hits, minimum=3, maximum=5)
    citations = [
        {"file": hit.get("file"), "page": hit.get("page"), "score": float(hit.get("score", 0.0))}
        for hit in selected_hits
    ]
    citations_insufficient = not has_minimum_citations

    if MEMORY_ENABLED:
        mem.record(memory_key, inp.conversation_id, inp.message, answer)

    latency_ms = (time.perf_counter() - start) * 1000

    summary = answer.strip()
    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."

    log = ChatLog(
        user_id=user.id,
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

    response: dict[str, Any] = {"answer": answer, "citations": citations}
    if citations_insufficient:
        response["citations_insufficient"] = True

    return response


@app.get("/admin/chat-logs", response_class=HTMLResponse)
def chat_logs(
    request: Request,
    db: SessionDep,
    _: Annotated[User, Depends(require_admin)] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str | None = Query(None),
    conversation_id: str | None = Query(None),
) -> HTMLResponse:
    query = db.query(ChatLog)
    if user_id:
        query = query.filter(ChatLog.user_id == int(user_id))
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


__all__ = ["app", "get_current_user"]
