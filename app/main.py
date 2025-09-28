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
from app.chat.store import ChatStore, ConversationAccessError
from app.chat.summarizer import ConversationSummarizer
from app.models.ollama_client import ensure_model, generate
from app.models.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.rag.ingest import parse_and_chunk
from app.security import create_access_token, verify_password

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")
app.include_router(auth_router)

get_current_user = _auth_get_current_user

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", str(FILES_ROOT / "db" / "chat_history.sqlite")))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "12"))
CHAT_SUMMARY_TRIGGER = int(os.getenv("CHAT_SUMMARY_TRIGGER", "10"))

chat_store = ChatStore(str(CHAT_DB_PATH))
summarizer = ConversationSummarizer(chat_store, lambda prompt: generate(prompt))

SessionDep = Annotated[Session, Depends(get_session)]


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    must_change_password: bool


class ChatIn(BaseModel):
    user_id: str
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
    if str(user.id) != inp.user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_MISMATCH")

    try:
        conversation_id = chat_store.ensure_conversation(inp.user_id, inp.conversation_id)
    except ConversationAccessError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="CONVERSATION_FORBIDDEN") from exc

    summary_text = chat_store.get_summary(conversation_id) or ""
    history = chat_store.get_recent_messages(conversation_id, limit=CHAT_HISTORY_LIMIT)
    history_text = "\n".join(f"{role}: {content}" for role, content in history) if history else ""

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
