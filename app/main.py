from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
        codex/refactor-modules-to-remove-codex-markers
from typing import Annotated, Any

from typing import Annotated
        main

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

        codex/refactor-modules-to-remove-codex-markers
from app.auth import router as auth_router, require_active_user, require_admin, setup_defaults
from app.db.models import ChatLog, User
from app.db.session import get_session, init_db

from app.auth import router as auth_router, setup_defaults
from app.db.models import ChatLog, User, UserRole
from app.db.session import get_session
        main
from app.memory.store import MemoryStore
from app.models.ollama_client import ensure_model, generate
from app.models.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.rag.ingest import parse_and_chunk
        codex/refactor-modules-to-remove-codex-markers

FILES_ROOT = os.getenv("FILES_ROOT", "/opt/knowlab/data/files")
DB_PATH = os.path.join(FILES_ROOT, "db", "kb.sqlite")
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() == "true"

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")
app.include_router(auth_router)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

mem = MemoryStore(
    db_path=DB_PATH,

from app.security import create_access_token, decode_token, hash_password, verify_password

LOGGER = logging.getLogger(__name__)

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
DB_PATH = FILES_ROOT / "db" / "kb.sqlite"
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() == "true"

mem = MemoryStore(
    db_path=str(DB_PATH),
        main
    ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS", "90")),
    summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER", "10")),
    max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK", "2000")),
)

        codex/refactor-modules-to-remove-codex-markers

@app.on_event("startup")
def on_startup() -> None:
    init_db()
    setup_defaults()


@app.get("/health", methods=["GET", "HEAD"])
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


app = FastAPI(title="kb")
app.include_router(auth_router)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
        main

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
SessionDep = Annotated[Session, Depends(get_session)]


        codex/refactor-modules-to-remove-codex-markers
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


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    must_change_password: bool


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


@app.on_event("startup")
def on_startup() -> None:
    setup_defaults()


@app.get("/health", response_class=JSONResponse)
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: SessionDep) -> User:
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_TOKEN") from exc
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="USER_NOT_FOUND")
    if user.must_change_password:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PASSWORD_CHANGE_REQUIRED")
    return user


def require_role(required: UserRole):
    def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        if required == UserRole.ADMIN and user.role != UserRole.ADMIN:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="INSUFFICIENT_ROLE")
        if required == UserRole.STAFF and user.role not in {UserRole.STAFF, UserRole.ADMIN}:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="INSUFFICIENT_ROLE")
        return user

    return dependency


@app.post("/api/auth/token", response_model=TokenOut)
def login(db: SessionDep, form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> TokenOut:
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")
    token = create_access_token({"sub": str(user.id)})
    return TokenOut(access_token=token, must_change_password=user.must_change_password)


@app.post("/api/auth/change-password")
def change_password(payload: ChangePasswordIn, db: SessionDep, user: Annotated[User, Depends(get_current_user)]):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="INVALID_PASSWORD")
    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    db.add(user)
    db.commit()
    return {"ok": True}


@app.post("/api/docs/upload")
async def upload(
    file: UploadFile = File(...),
    _: Annotated[User, Depends(require_role(UserRole.ADMIN))] = None,
):
    name = file.filename or "uploaded"
    ext = name.rsplit(".", 1)[-1].lower()
    if ext not in {"pdf", "docx", "txt"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "UPLOAD_INVALID_EXT")
        main
    data = await file.read()
    ensure_collection()
    chunks = parse_and_chunk(name, data)
    if not chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "NO_TEXT_FOUND")
    upsert_chunks(chunks)
    return {"ok": True, "chunks": len(chunks)}


@app.post("/api/chat")
def chat(
        codex/refactor-modules-to-remove-codex-markers
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

    payload: ChatIn,
    db: SessionDep,
    user: Annotated[User, Depends(require_role(UserRole.STAFF))],
):
    ensure_model()
    ensure_collection()

    start = time.perf_counter()
    memory_key = str(user.id)
    memory_text = ""
    if MEMORY_ENABLED:
        memory_text = mem.load_context(memory_key, payload.conversation_id)

    hits = search_chunks(payload.message, top_k=int(os.getenv("RETRIEVE_TOPK", "24")))
    context = build_context(hits, token_limit=3000)

    prompt = f"Context:\n{context}\n\nMemory:\n{memory_text}\n\nQuestion: {payload.message}\nAnswer in Russian with citations."
        main
    answer = generate(prompt)

    selected_hits = select_citations(hits, minimum=3, maximum=5)
    citations = [
        {"file": hit.get("file"), "page": hit.get("page"), "score": float(hit.get("score", 0.0))}
        for hit in selected_hits
    ]

        codex/refactor-modules-to-remove-codex-markers
    if MEMORY_ENABLED:
        mem.record(memory_key, inp.conversation_id, inp.message, answer)

    latency_ms = (time.perf_counter() - start) * 1000

    elapsed = time.perf_counter() - start

    if MEMORY_ENABLED:
        mem.record(memory_key, payload.conversation_id, payload.message, answer)

        main
    summary = answer.strip()
    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."

        codex/refactor-modules-to-remove-codex-markers
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

    try:
        log = ChatLog(
            user_id=user.id,
            conversation_id=payload.conversation_id,
            question=payload.message,
            response_summary=summary,
            citations=citations,
            response_time=elapsed,
        )
        db.add(log)
        db.commit()
    except Exception:  # pragma: no cover - persistence failures shouldn't break chat
        db.rollback()
        LOGGER.exception("Failed to persist chat log")

    return {"answer": answer, "citations": citations, "latency_ms": int(elapsed * 1000)}


@app.get("/admin/logs", response_class=HTMLResponse)
def admin_logs(
    request: Request,
    db: SessionDep,
    _: Annotated[User, Depends(require_role(UserRole.ADMIN))],
):
    limit = int(os.getenv("ADMIN_LOGS_LIMIT", "200"))
    logs = (
        db.query(ChatLog)
        .order_by(ChatLog.answered_at.desc())
        .limit(limit)
        .all()
    )
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})
        main


@app.get("/admin/chat-logs", response_class=HTMLResponse)
def chat_logs(
    request: Request,
        codex/refactor-modules-to-remove-codex-markers
    db: DatabaseSession,

    db: SessionDep,
        main
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str | None = Query(None),
    conversation_id: str | None = Query(None),
        codex/refactor-modules-to-remove-codex-markers
    _: Annotated[User, Depends(require_admin)] = None,
) -> HTMLResponse:

    _: Annotated[User, Depends(require_role(UserRole.ADMIN))] = None,
):
        main
    query = db.query(ChatLog)
    if user_id:
        query = query.filter(ChatLog.user_id == int(user_id))
    if conversation_id:
        query = query.filter(ChatLog.conversation_id == conversation_id)

    total = query.count()
    pages = math.ceil(total / page_size) if total else 1
    logs = (
        query.order_by(ChatLog.answered_at.desc())
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
        codex/refactor-modules-to-remove-codex-markers


__all__ = ["app"]

        main
