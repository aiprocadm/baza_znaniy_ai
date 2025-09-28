from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.memory.store import MemoryStore
from app.models.ollama_client import ensure_model, generate
from app.models.qdrant_client import ensure_collection, search_chunks, upsert_chunks
from app.rag.context import build_context, select_citations
from app.rag.ingest import parse_and_chunk

app = FastAPI(title="kb")

FILES_ROOT = Path(os.getenv("FILES_ROOT", "/opt/knowlab/data/files"))
DB_PATH = FILES_ROOT / "db" / "kb.sqlite"
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() == "true"

mem = MemoryStore(
    db_path=str(DB_PATH),
    ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS", "90")),
    summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER", "10")),
    max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK", "2000")),
)

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
    memory_key = inp.user_id
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

    response: dict[str, Any] = {"answer": answer, "citations": citations}
    if citations_insufficient:
        response["citations_insufficient"] = True

    return response


__all__ = ["app"]
