from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine, get_db
from app.memory.store import MemoryStore
from app.models.db_models import ChatLog, Role, User
from app.models.ollama_client import ensure_model, generate
from app.models.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.ingest import parse_and_chunk
from app.security import create_access_token, decode_token, hash_password, verify_password

MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() == "true"

app = FastAPI(title="kb")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

mem = MemoryStore(
    db_path="/srv/projects/kb/data/db/kb.sqlite",
    ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS", "90")),
    summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER", "10")),
    max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK", "2000")),
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    must_change_password: bool


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        _ensure_roles(db)
        _ensure_admin_user(db)


def _ensure_roles(db: Session) -> None:
    required = {"admin", "staff"}
    existing = {role.name for role in db.query(Role).filter(Role.name.in_(required)).all()}
    created = False
    for name in required - existing:
        db.add(Role(name=name))
        created = True
    if created:
        db.commit()


def _ensure_admin_user(db: Session) -> None:
    admin_role = db.query(Role).filter(Role.name == "admin").first()
    if not admin_role:
        return
    admin_user = db.query(User).filter(User.username == "admin").first()
    if not admin_user:
        admin_user = User(
            username="admin",
            password_hash=hash_password("admin"),
            role=admin_role,
            must_change_password=True,
        )
        db.add(admin_user)
        db.commit()
        return
    if admin_user.role_id != admin_role.id:
        admin_user.role = admin_role
    if verify_password("admin", admin_user.password_hash) and not admin_user.must_change_password:
        admin_user.must_change_password = True
    db.add(admin_user)
    db.commit()


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "ts": int(time.time())})


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = decode_token(token)
        sub = payload.get("sub")
        user_id = int(sub)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="INVALID_TOKEN",
        )
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="USER_NOT_FOUND",
        )
    return user


def require_role(required_role: str):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.must_change_password:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="PASSWORD_CHANGE_REQUIRED",
            )
        role_name = user.role.name if user.role else None
        allowed = {required_role}
        if required_role == "staff":
            allowed.add("admin")
        if role_name not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="INSUFFICIENT_ROLE",
            )
        return user

    return dependency


@app.post("/api/auth/token", response_model=TokenOut)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
) -> TokenOut:
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="INVALID_CREDENTIALS",
        )
    token = create_access_token({"sub": str(user.id)})
    return TokenOut(access_token=token, must_change_password=user.must_change_password)


@app.post("/api/auth/change-password")
def change_password(
    payload: ChangePasswordIn,
    user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
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
    user: Annotated[User, Depends(require_role("admin"))] = None,
):
    del user  # hint for linters; role check performed in dependency
    name = file.filename
    ext = name.rsplit(".", 1)[-1].lower()
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
    user: Annotated[User, Depends(require_role("staff"))],
    db: Session = Depends(get_db),
):
    ensure_model()
    ensure_collection()
    start_ts = time.perf_counter()
    memory_key = str(user.id)
    memory = mem.load_context(memory_key, inp.conversation_id) if MEMORY_ENABLED else ""
    hits = search_chunks(inp.message, top_k=int(os.getenv("RETRIEVE_TOPK", "24")))
    context = "\n\n".join(h["text"] for h in hits[:8])
    prompt = f"""Ты помощник по нормативным документам. Отвечай кратко и давай точные цитаты с указанием файла и страницы.
Контекст:
{context}

Память:
{memory}

Вопрос: {inp.message}
"""
    answer = generate(prompt)
    citations = [
        {"file": h["file"], "page": h.get("page"), "score": float(h["score"])}
        for h in hits[:5]
    ]
    elapsed = time.perf_counter() - start_ts
    if MEMORY_ENABLED:
        mem.record(memory_key, inp.conversation_id, inp.message, answer)
    db.add(
        ChatLog(
            user_id=user.id,
            question=inp.message,
            citations=citations,
            response_time=elapsed,
        )
    )
    db.commit()
    return {"answer": answer, "citations": citations}


@app.get("/admin/logs", response_class=HTMLResponse)
def admin_logs(
    request: Request,
    user: Annotated[User, Depends(require_role("admin"))],
    db: Session = Depends(get_db),
):
    del user
    limit = int(os.getenv("ADMIN_LOGS_LIMIT", "200"))
    logs = (
        db.query(ChatLog)
        .order_by(ChatLog.answered_at.desc())
        .limit(limit)
        .all()
    )
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})
