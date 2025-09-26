import os, time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.models.qdrant_client import ensure_collection, upsert_chunks, search_chunks
from app.models.ollama_client import ensure_model, generate
from app.rag.ingest import parse_and_chunk
from app.memory.store import MemoryStore

APP_SECRET = os.getenv("APP_SECRET","dev")
MEMORY_ENABLED = os.getenv("CHAT_MEMORY_ENABLED","true").lower()=="true"

app = FastAPI(title="kb")

mem = MemoryStore(
  db_path="/srv/projects/kb/data/files/kb.sqlite",
  ttl_days=int(os.getenv("CHAT_MEMORY_TTL_DAYS","90")),
  summary_trigger=int(os.getenv("CHAT_SUMMARY_TRIGGER","10")),
  max_tokens=int(os.getenv("CHAT_MEMORY_MAXTOK","2000"))
)

@app.api_route("/health", methods=["GET","HEAD"])
def health():
    return JSONResponse({"status":"ok","ts":int(time.time())})

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

@app.post("/api/chat")
def chat(inp: ChatIn):
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
    citations = [{"file":h["file"], "page":h.get("page"), "score":float(h["score"])} for h in hits[:5]]
    if MEMORY_ENABLED:
        mem.record(inp.user_id, inp.conversation_id, inp.message, answer)
    return {"answer": answer, "citations": citations}
