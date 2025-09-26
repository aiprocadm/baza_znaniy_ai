"""Entry-point for the knowledge base web service."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .memory import DocumentMemory
from .models import Document, DocumentCreate, QueryRequest, QueryResponse
from .rag import retrieve

load_dotenv()
settings = config.get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("kb")

app = FastAPI(title="Knowledge Base API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = DocumentMemory(Path(settings.data_dir) / "documents.json")


@app.on_event("startup")
def bootstrap() -> None:
    logger.info("Knowledge base service starting with %d documents", len(memory.all()))


@app.get("/health", tags=["system"])
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/documents", response_model=List[Document], tags=["documents"])
def list_documents() -> List[Document]:
    return memory.all()


@app.post("/documents", response_model=Document, tags=["documents"], status_code=201)
def create_document(payload: DocumentCreate) -> Document:
    document = memory.add(payload)
    logger.info("Added document %s", document.id)
    return document


@app.delete("/documents/{document_id}", tags=["documents"])
def delete_document(document_id: str) -> JSONResponse:
    if not memory.remove(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info("Removed document %s", document_id)
    return JSONResponse({"status": "deleted", "id": document_id})


@app.post("/query", response_model=QueryResponse, tags=["retrieval"])
def query(payload: QueryRequest) -> QueryResponse:
    matches = retrieve(payload.question, memory.all(), limit=payload.limit)
    documents = [item[0] for item in matches]
    if not documents:
        logger.info("No matches for query: %s", payload.question)
    return QueryResponse(question=payload.question, matches=documents)
