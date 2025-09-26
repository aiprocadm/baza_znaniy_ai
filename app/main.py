import logging
import math
import os
import time
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.models.qdrant_client import ensure_collection, upsert_chunks, search_chunks
from app.models.ollama_client import ensure_model, generate
from app.rag.ingest import parse_and_chunk
from app.memory.store import MemoryStore
from app.db import get_db, init_db
from app.db.models import ChatLog

APP_SECRET = os.getenv("APP_SECRET","dev")
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED","true").lower()=="true"

logger = logging.getLogger(__name__)

app = FastAPI(title="kb")
templates = Jinja2Templates(directory="app/templates")

mem = MemoryStore(
  db_path="/srv/projects/kb/data/db/kb.sqlite",
  ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS","90")),
  summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER","10")),
  max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK","2000"))
)

@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.api_route("/health", methods=["GET","HEAD"])
def health():
    return JSONResponse({"status": "ok", "ts": int(time.time())})

class ChatIn(BaseModel):
    user_id: str
    message: str
    conversation_id: str | None = None

@app.post("/api/docs/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename
    ext = name.rsplit(".",1)[-1].lower()
    if ext not in {"pdf","docx","txt"}:
        raise HTTPException(400, "UPLOAD_INVALID_EXT")
    data = await file.read()
    ensure_collection()
    chunks = parse_and_chunk(name, data)
    if not chunks:
        raise HTTPException(400, "NO_TEXT_FOUND")
    upsert_chunks(chunks)
    return {"ok": True, "chunks": len(chunks)}

def require_admin(
    secret_header: str | None = Header(None, alias="X-App-Secret"),
    secret_query: str | None = Query(None, alias="secret"),
) -> None:
    secret = secret_header or secret_query
    if secret != APP_SECRET:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")


@app.post("/api/chat")
def chat(inp: ChatIn, db: Session = Depends(get_db)):
    start = time.perf_counter()
    ensure_model()
    ensure_collection()
    memory = mem.load_context(inp.user_id, inp.conversation_id) if MEMORY_ENABLED else ""
    hits = search_chunks(inp.message, top_k=int(os.getenv("RETRIEVE_TOPK","24")))
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
    if MEMORY_ENABLED:
        mem.record(inp.user_id, inp.conversation_id, inp.message, answer)
    latency_ms = (time.perf_counter() - start) * 1000
    citation_payload: list[dict[str, Any]] = [
        {"file": c.get("file"), "page": c.get("page")}
        for c in citations
    ]
    summary = answer.strip()
    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."
    try:
        log = ChatLog(
            user_id=inp.user_id,
            conversation_id=inp.conversation_id,
            question=inp.message,
            response_summary=summary,
            citations=citation_payload,
            latency_ms=latency_ms,
        )
        db.add(log)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist chat log")
    return {"answer": answer, "citations": citations}


@app.get("/admin/chat-logs", response_class=HTMLResponse)
def chat_logs(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str | None = Query(None),
    conversation_id: str | None = Query(None),
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
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
