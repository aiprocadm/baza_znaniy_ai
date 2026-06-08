"""MVP knowledge-base endpoints mounted under ``/api/kb``.

Auth-free contract for the simple frontend in ``data/www/index.html``.
The full multi-tenant API stays under ``/api/v1/*``. ``/ask`` prefers
:func:`app.services.kb_llm.select_provider` (DeepSeek/Groq/OpenRouter/
Ollama/…), falls back to ``state.llm_provider`` (legacy), then to an
extractive answer stitched from the top retrieved chunks.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.api.kb_auth import auth_status, require_api_key
from app.observability import retrieval_health
from app.services import kb_embeddings, kb_llm, kb_rerank
from app.services.kb_store import (
    Conversation as StoredConversation,
    DEFAULT_HISTORY_LIMIT,
    Document as StoredDocument,
    KnowledgeBaseStore,
    MAX_CONVERSATION_TITLE,
    MAX_QUERY_LEN,
    MAX_TEXT_LEN,
    Message as StoredMessage,
    SearchHit,
    get_store,
)

LOGGER = logging.getLogger(__name__)

# Top-level router that aggregates a public sub-router (health/providers,
# no auth) and a protected one (everything that mutates or reads private
# content). Anything new should go to `protected` unless explicitly
# whitelisted as public.
router = APIRouter(tags=["kb-mvp"])
public = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB
SUPPORTED_UPLOAD_EXT = {
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "txt",
    "md",
    "markdown",
    "html",
    "htm",
}


class DocumentCreate(BaseModel):
    """Payload accepted by ``POST /api/kb/documents``."""

    title: str = Field(default="", max_length=300)
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("text is empty")
        if len(cleaned) > MAX_TEXT_LEN:
            raise ValueError(f"text exceeds {MAX_TEXT_LEN} characters")
        return cleaned

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        return (value or "").strip()


class DocumentOut(BaseModel):
    id: int
    title: str
    text: Optional[str] = None
    created_at: str
    chunks: int
    source: str = "text"
    filename: Optional[str] = None
    mime_type: Optional[str] = None


class DocumentListItem(BaseModel):
    id: int
    title: str
    created_at: str
    chunks: int
    source: str = "text"
    filename: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("query is empty")
        return cleaned


class HitOut(BaseModel):
    """A single ranked chunk — used by both ``/search`` and ``/ask``."""

    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    source: str = "text"
    filename: Optional[str] = None
    page: Optional[int] = None
    has_original: bool = False


class RerankInfo(BaseModel):
    """Reranker diagnostics returned with /search and /ask responses."""

    enabled: bool
    used: bool = False
    model: Optional[str] = None
    candidates: int = 0
    elapsed_ms: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    hits: List[HitOut]
    rerank: Optional[RerankInfo] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = Field(default=4, ge=1, le=20)
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation UUID. Omit to start a new one and receive its id in the response.",
        max_length=64,
    )
    history_limit: int = Field(
        default=DEFAULT_HISTORY_LIMIT,
        ge=0,
        le=50,
        description="How many previous messages to include in the LLM prompt context.",
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("question is empty")
        return cleaned

    @field_validator("conversation_id")
    @classmethod
    def _strip_conv_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class RetrievalReasonOut(BaseModel):
    reason: str
    severity: str
    detail: str = ""


class RetrievalReportOut(BaseModel):
    degraded: bool
    severity: str
    reasons: List[RetrievalReasonOut] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[HitOut]
    provider: str
    model: Optional[str] = None
    elapsed_ms: Optional[float] = None
    rerank: Optional[RerankInfo] = None
    retrieval: Optional[RetrievalReportOut] = None
    conversation_id: str = Field(
        ..., description="Conversation UUID — same as request, or freshly created."
    )


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=MAX_CONVERSATION_TITLE)


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=MAX_CONVERSATION_TITLE)


class MessageOut(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    sources: List[HitOut] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None


class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = Field(default_factory=list)


def _store_for(request: Request) -> KnowledgeBaseStore:
    """Return the MVP store associated with the FastAPI application."""

    state = getattr(request, "app", None)
    state = getattr(state, "state", None) if state is not None else None
    cached = getattr(state, "kb_mvp_store", None) if state is not None else None
    if isinstance(cached, KnowledgeBaseStore):
        return cached
    store = get_store()
    if state is not None:
        state.kb_mvp_store = store
    return store


def _doc_to_out(doc: StoredDocument, *, include_text: bool = False) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        title=doc.title,
        text=doc.text if include_text else None,
        created_at=doc.created_at,
        chunks=doc.chunks,
        source=doc.source,
        filename=doc.filename,
        mime_type=doc.mime_type,
    )


def _hit_to_out(hit: SearchHit) -> HitOut:
    return HitOut(
        document_id=hit.document_id,
        document_title=hit.document_title,
        chunk_index=hit.chunk_index,
        text=hit.text,
        score=round(hit.score, 6),
        source=hit.source,
        filename=hit.filename,
        page=hit.page,
        has_original=hit.has_original,
    )


def _conversation_to_out(conv: StoredConversation) -> ConversationOut:
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=conv.message_count,
    )


def _sources_payload_to_hit_out(items: List[Any]) -> List[HitOut]:
    out: List[HitOut] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_val = raw.get("page")
            page_int: Optional[int]
            if page_val is None:
                page_int = None
            else:
                try:
                    page_int = int(page_val)
                except (TypeError, ValueError):
                    page_int = None
            out.append(
                HitOut(
                    document_id=int(raw.get("document_id") or 0),
                    document_title=str(raw.get("document_title") or ""),
                    chunk_index=int(raw.get("chunk_index") or 0),
                    text=str(raw.get("text") or ""),
                    score=float(raw.get("score") or 0.0),
                    source=str(raw.get("source") or "text"),
                    filename=raw.get("filename") if isinstance(raw.get("filename"), str) else None,
                    page=page_int,
                    has_original=bool(raw.get("has_original")),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def _message_to_out(msg: StoredMessage) -> MessageOut:
    return MessageOut(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at,
        sources=_sources_payload_to_hit_out(list(msg.sources)),
        provider=msg.provider,
        model=msg.model,
    )


def _format_history(messages: List[StoredMessage]) -> str:
    if not messages:
        return ""
    lines = ["Контекст предыдущего диалога:"]
    role_label = {"user": "Пользователь", "assistant": "Ассистент", "system": "Система"}
    for msg in messages:
        label = role_label.get(msg.role, msg.role)
        content = msg.content.strip()
        if len(content) > 1000:
            content = content[:1000].rsplit(" ", 1)[0] + "…"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _retrieve_with_rerank(
    store: KnowledgeBaseStore,
    query: str,
    top_k: int,
) -> tuple[List[SearchHit], RerankInfo]:
    """Two-stage retrieval: bi-encoder shortlist → cross-encoder rerank.

    When ``KB_RERANK_ENABLED=true`` we over-fetch ``KB_RERANK_CANDIDATES``
    bi-encoder hits and let the cross-encoder pick the final ``top_k``.
    Otherwise the bi-encoder result is truncated to ``top_k`` directly.
    """

    config = kb_rerank.load_config()
    rerank_info = RerankInfo(
        enabled=config.enabled, model=config.model_name if config.enabled else None
    )

    if not config.enabled:
        hits = store.search(query, top_k=top_k)
        return hits, rerank_info

    shortlist_size = max(top_k, config.candidates)
    shortlist = store.search(query, top_k=shortlist_size)
    if not shortlist:
        return [], rerank_info

    result = kb_rerank.rerank_hits(query, shortlist, config=config, top_n=top_k)
    rerank_info = RerankInfo(
        enabled=True,
        used=True,
        model=result.model,
        candidates=result.candidates,
        elapsed_ms=round(result.elapsed_ms, 2) if result.elapsed_ms else None,
    )
    return result.hits, rerank_info


def _format_context(hits: List[SearchHit]) -> str:
    parts = []
    for index, hit in enumerate(hits, start=1):
        source_label = hit.filename or hit.document_title
        parts.append(f"[{index}] {source_label}\n{hit.text}")
    return "\n\n---\n\n".join(parts)


def _extractive_answer(hits: List[SearchHit], limit: int = 3) -> str:
    lines = ["Ответ собран из найденных фрагментов базы знаний:"]
    for index, hit in enumerate(hits[:limit], start=1):
        snippet = hit.text.strip()
        if len(snippet) > 400:
            cut = snippet[:400].rsplit(" ", 1)[0]
            snippet = cut + "…"
        label = hit.filename or hit.document_title
        lines.append(f"[{index}] {label}: {snippet}")
    return "\n".join(lines)


_RAG_SYSTEM_PROMPT = (
    "Ты — помощник корпоративной базы знаний. Отвечай на русском. "
    "Используй ТОЛЬКО фрагменты из приведённого контекста и не добавляй фактов, которых в нём нет. "
    "Каждое утверждение в ответе сопровождай ссылкой на номер подтверждающего фрагмента в формате [N]. "
    "Если в контексте недостаточно данных, ответь ровно фразой: "
    "Не удалось найти в документах информацию для ответа."
)


def _build_rag_prompt(
    question: str,
    hits: List[SearchHit],
    *,
    history: str = "",
) -> str:
    parts = []
    if history:
        parts.append(history)
    parts.append("Фрагменты базы знаний:\n" + _format_context(hits))
    parts.append(f"Вопрос пользователя: {question}\nОтвет:")
    return "\n\n".join(parts)


def _generate_answer(
    question: str,
    hits: List[SearchHit],
    request: Request,
    *,
    history: str = "",
) -> tuple[str, str, Optional[str], Optional[float]]:
    """Build an answer using the configured LLM or the extractive fallback."""

    if not hits:
        return (
            "В базе знаний пока нет данных, релевантных вопросу. Добавьте документы и повторите запрос.",
            "none",
            None,
            None,
        )

    prompt = _build_rag_prompt(question, hits, history=history)

    provider = kb_llm.select_provider()
    if provider is not None:
        try:
            response = provider.generate(prompt, system=_RAG_SYSTEM_PROMPT)
            return response.text, response.provider, response.model, response.elapsed_ms
        except kb_llm.LLMTransportError as exc:
            LOGGER.warning("LLM %s transport error: %s", provider.name, exc)
        except Exception:  # pragma: no cover - defensive fallback
            LOGGER.exception("LLM provider %s failed; using fallback", provider.name)

    legacy = getattr(getattr(request, "app", None), "state", None)
    legacy = getattr(legacy, "llm_provider", None) if legacy is not None else None
    if legacy is not None:
        try:
            ensure_ready = getattr(legacy, "ensure_ready", None)
            if callable(ensure_ready):
                ensure_ready()
            generate = getattr(legacy, "generate", None)
            if callable(generate):
                raw = generate(prompt)
                text = (str(raw) if raw is not None else "").strip()
                if text:
                    return text, str(getattr(legacy, "name", "legacy")), None, None
        except Exception:  # pragma: no cover
            LOGGER.exception("Legacy LLM provider failed")

    return _extractive_answer(hits), "extractive", None, None


def _extension_for(filename: str) -> str:
    name = (filename or "").strip().lower()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1]


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _resolve_data_dir() -> Path:
    """Return the data directory using env-var fallback chain.

    Resolution order: DATA_DIR env → FILES_ROOT env → get_settings().data_dir →
    './var/data'. The 4-level fallback exists because the project's
    Settings instantiation has a known AliasChoices bug (see
    app/core/config.py:82) that breaks in some test environments; the
    env-var fast path bypasses it.
    """
    base_str = os.environ.get("DATA_DIR") or os.environ.get("FILES_ROOT")
    if base_str:
        return Path(base_str)
    try:
        from app.core.config import get_settings

        return Path(get_settings().data_dir)
    except Exception:
        return Path("./var/data")


def _resolve_kb_files_dir() -> Path:
    """Return <data_dir>/kb_files, creating it if needed."""
    kb_files_dir = _resolve_data_dir() / "kb_files"
    kb_files_dir.mkdir(parents=True, exist_ok=True)
    return kb_files_dir


def _parse_file_bytes(filename: str, data: bytes) -> tuple[str, str]:
    """Return ``(plain_text, mime_type)`` extracted from an uploaded file.

    Uses :func:`app.ingest.chunking.parse_document` for rich formats and
    falls back to simple decoders otherwise. Always returns text — empty
    string means «no extractable content».
    """

    ext = _extension_for(filename)
    if not ext:
        return _decode_text(data), "text/plain"

    if ext in {"txt", "md", "markdown"}:
        return _decode_text(data), "text/markdown" if ext != "txt" else "text/plain"

    try:
        from app.ingest.chunking import parse_document
    except Exception as exc:  # pragma: no cover - optional dependency missing
        LOGGER.warning("parse_document unavailable (%s); decoding as text", exc)
        return _decode_text(data), "application/octet-stream"

    try:
        result = parse_document(filename, data)
    except Exception as exc:
        LOGGER.exception("Failed to parse %s: %s", filename, exc)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"FAILED_TO_PARSE: {exc}",
        ) from exc

    pages = getattr(result, "pages", []) or []
    text_parts: list[str] = []
    for page_number, page_text in pages:
        if not page_text:
            continue
        text_parts.append(str(page_text).strip())
    full_text = "\n\n".join(part for part in text_parts if part)
    mime = (result.metadata.get("document", {}) or {}).get(
        "mime_type"
    ) or "application/octet-stream"
    return full_text, mime


def _parse_file_bytes_with_pages(filename: str, data: bytes) -> tuple[list[tuple[int, str]], str]:
    """Like :func:`_parse_file_bytes` but preserves per-page structure.

    Returns ``(pages, mime_type)`` where ``pages`` is a list of
    ``(page_number, text)`` tuples (page numbers 1-indexed). Empty pages
    are dropped. Plain-text formats produce a single virtual page.

    Raises ``HTTPException`` on parse failure (same contract as
    ``_parse_file_bytes``).
    """

    ext = _extension_for(filename)
    if not ext:
        text = _decode_text(data).strip()
        return ([(1, text)] if text else []), "text/plain"

    if ext in {"txt", "md", "markdown"}:
        text = _decode_text(data).strip()
        mime = "text/markdown" if ext != "txt" else "text/plain"
        return ([(1, text)] if text else []), mime

    try:
        from app.ingest.chunking import parse_document
    except Exception as exc:  # pragma: no cover - optional dependency missing
        LOGGER.warning("parse_document unavailable (%s); decoding as text", exc)
        text = _decode_text(data).strip()
        return ([(1, text)] if text else []), "application/octet-stream"

    try:
        result = parse_document(filename, data)
    except Exception as exc:
        LOGGER.exception("Failed to parse %s: %s", filename, exc)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"FAILED_TO_PARSE: {exc}",
        ) from exc

    raw_pages = getattr(result, "pages", []) or []
    pages: list[tuple[int, str]] = []
    for page_number, page_text in raw_pages:
        text = (str(page_text) if page_text is not None else "").strip()
        if text:
            pages.append((int(page_number), text))

    mime = (result.metadata.get("document", {}) or {}).get(
        "mime_type"
    ) or "application/octet-stream"
    return pages, mime


@public.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness probe with LLM, embedder, reranker, auth, KB stats and compliance."""

    import shutil as _shutil
    import sqlite3 as _sqlite3

    store = _store_for(request)
    db_path = Path(store.db_path)
    documents_count = 0
    chunks_count = 0
    distinct_dims = 0
    db_size_bytes = 0
    last_indexed_at: Optional[str] = None
    if db_path.is_file():
        db_size_bytes = db_path.stat().st_size
        try:
            conn = _sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()
                if row:
                    documents_count = int(row[0])
                row = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()
                if row:
                    chunks_count = int(row[0])
                row = conn.execute("SELECT MAX(created_at) FROM kb_documents").fetchone()
                if row and row[0]:
                    last_indexed_at = str(row[0])
                row = conn.execute("SELECT COUNT(DISTINCT dim) FROM kb_chunks").fetchone()
                distinct_dims = int(row[0]) if row else 0
            finally:
                conn.close()
        except _sqlite3.Error:
            pass

    try:
        disk_target = db_path.parent if db_path.parent.is_dir() else Path.cwd()
        disk_free_bytes = _shutil.disk_usage(str(disk_target)).free
    except OSError:
        disk_free_bytes = 0

    compliance_mode = os.environ.get("KB_COMPLIANCE_MODE") or None

    extra: list[tuple] = []
    try:
        # Prefer the store's own embedder to detect degradation; fall back to global.
        _store_embedder = getattr(store, "embedder", None)
        _embedder_name = getattr(_store_embedder, "name", None)
        if _embedder_name is None:
            _embedder_name = kb_embeddings.embedder_status().get("name")
        if _embedder_name == "hash":
            extra.append((retrieval_health.RetrievalReason.HASHING_EMBEDDER, "embedder=hash"))
    except Exception:  # pragma: no cover - never let a probe break health
        pass
    if distinct_dims > 1:
        extra.append(
            (
                retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH,
                f"{distinct_dims} distinct embedding dims present",
            )
        )
    retrieval = retrieval_health.snapshot(extra=tuple(extra))

    return {
        "status": "ok",
        "degraded": retrieval["degraded"],
        "retrieval": retrieval,
        "llm": kb_llm.provider_status(),
        "embedder": kb_embeddings.embedder_status(),
        "reranker": kb_rerank.reranker_status(),
        "auth": auth_status(),
        "kb_stats": {
            "documents_count": documents_count,
            "chunks_count": chunks_count,
            "db_size_bytes": db_size_bytes,
            "disk_free_bytes": disk_free_bytes,
            "last_indexed_at": last_indexed_at,
        },
        "compliance_mode": compliance_mode,
        "compliance_implemented": False,
    }


@public.get("/providers")
def providers() -> dict[str, Any]:
    """Detailed snapshot of LLM providers seen by the service."""

    return kb_llm.provider_status()


@protected.post(
    "/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
)
def create_document(payload: DocumentCreate, request: Request) -> DocumentOut:
    """Persist a text document and index its chunks."""

    store = _store_for(request)
    try:
        doc = store.add_document(payload.title, payload.text, source="text")
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _doc_to_out(doc)


@protected.post(
    "/documents/upload",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
) -> DocumentOut:
    """Accept a binary file, parse it, and store as a document."""

    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="FILENAME_REQUIRED")

    ext = _extension_for(filename)
    if ext not in SUPPORTED_UPLOAD_EXT:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"UNSUPPORTED_EXT: .{ext}. Allowed: {', '.join(sorted(SUPPORTED_UPLOAD_EXT))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="EMPTY_FILE")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"FILE_TOO_LARGE: max {MAX_UPLOAD_BYTES} bytes",
        )

    # For PDFs we keep the raw blob so the viewer can show the original.
    # Write to a tmp name first; rename to <doc_id>.pdf AFTER the DB INSERT
    # succeeds, so we never leave a blob without a matching row.
    tmp_blob: Optional[Path] = None
    kb_files_dir: Optional[Path] = None
    if ext == "pdf":
        kb_files_dir = _resolve_kb_files_dir()
        tmp_blob = kb_files_dir / f".tmp-{uuid.uuid4().hex}.pdf"
        tmp_blob.write_bytes(data)

    try:
        pages, mime_type = _parse_file_bytes_with_pages(filename, data)
        if not pages:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="NO_EXTRACTABLE_TEXT")

        effective_title = (title or "").strip() or filename
        store = _store_for(request)
        try:
            doc = store.add_document(
                effective_title,
                pages=pages,
                source="file",
                filename=filename,
                mime_type=mime_type,
            )
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        # Promote tmp blob to final name and mark the doc.
        if tmp_blob is not None and kb_files_dir is not None:
            final_blob = kb_files_dir / f"{doc.id}.pdf"
            tmp_blob.rename(final_blob)
            tmp_blob = None  # ownership transferred
            store.update_file_metadata(doc.id, file_relpath=f"kb_files/{doc.id}.pdf")
            # Refresh the in-memory Document so we return up-to-date flags
            refreshed = store.get_document(doc.id)
            if refreshed is not None:
                doc = refreshed

        return _doc_to_out(doc)
    finally:
        # Clean up orphan tmp on any error path
        if tmp_blob is not None:
            try:
                tmp_blob.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("failed to remove tmp blob %s", tmp_blob)


@protected.get("/documents", response_model=List[DocumentListItem])
def list_documents(request: Request) -> List[DocumentListItem]:
    """Return all documents ordered by id descending."""

    store = _store_for(request)
    return [
        DocumentListItem(
            id=d.id,
            title=d.title,
            created_at=d.created_at,
            chunks=d.chunks,
            source=d.source,
            filename=d.filename,
        )
        for d in store.list_documents()
    ]


@protected.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, request: Request) -> DocumentOut:
    """Fetch a document by its numeric id including the full text."""

    store = _store_for(request)
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")
    return _doc_to_out(doc, include_text=True)


@protected.get("/documents/{doc_id}/file")
def get_document_file(doc_id: int, request: Request) -> FileResponse:
    """Stream the original blob for documents with has_original_file=true.

    Returns ``application/pdf`` with ``inline`` disposition so the
    browser/PDF.js can render it. Auth-gated by the ``protected``
    router — when ``KB_API_KEY`` is set, requires the standard
    ``X-API-Key`` header.
    """

    store = _store_for(request)
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")

    if not doc.has_original_file or not doc.file_relpath:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="NO_ORIGINAL_FILE")

    data_dir = _resolve_data_dir().resolve()
    absolute = (data_dir / doc.file_relpath).resolve()
    expected_root = (data_dir / "kb_files").resolve()

    # Path-traversal guard: resolved path must live under <data_dir>/kb_files/
    try:
        absolute.relative_to(expected_root)
    except ValueError:
        LOGGER.error("Path traversal attempted for doc %d: %s", doc_id, doc.file_relpath)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STORAGE_ERROR")

    if not absolute.is_file():
        LOGGER.warning("Original file missing for doc %d: %s", doc_id, absolute)
        raise HTTPException(status.HTTP_410_GONE, detail="FILE_DELETED")

    safe_name = (doc.filename or f"{doc_id}.pdf").replace('"', "_")
    return FileResponse(
        absolute,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{safe_name}"',
        },
    )


@protected.delete("/documents/{doc_id}")
def delete_document(doc_id: int, request: Request) -> dict[str, Any]:
    """Delete a document, its chunks, and the original blob (if any).

    The blob is removed BEFORE the DB row so an orphaned filesystem entry
    is never possible. If the file vanished (race / manual cleanup), we
    log a warning but still drop the DB row — the goal is to satisfy
    DELETE, not to fail because of dangling state.
    """

    store = _store_for(request)
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")

    if doc.has_original_file and doc.file_relpath:
        data_dir = _resolve_data_dir().resolve()
        expected_root = (data_dir / "kb_files").resolve()
        candidate = (data_dir / doc.file_relpath).resolve()
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            LOGGER.error(
                "Path traversal attempted on delete for doc %d: %s",
                doc_id,
                doc.file_relpath,
            )
            # Refuse to unlink anything outside kb_files/ — but still drop the row
            # so the corruption is at least cleared from the DB
            candidate = None
        if candidate is not None:
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("failed to remove blob for doc %d: %s", doc_id, exc)

    if not store.delete_document(doc_id):
        # Race: someone else deleted it between get_document and delete.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")
    return {"ok": True, "id": doc_id}


@protected.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest, request: Request) -> SearchResponse:
    """Run a similarity search, optionally followed by cross-encoder rerank."""

    store = _store_for(request)
    hits, rerank_info = _retrieve_with_rerank(store, payload.query, payload.top_k)
    return SearchResponse(
        query=payload.query,
        hits=[_hit_to_out(hit) for hit in hits],
        rerank=rerank_info,
    )


@protected.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Answer a question using retrieved chunks (with optional rerank) as context.

    Conversation behaviour:

    * No ``conversation_id`` → a fresh conversation is created and its id
      is returned in the response. The user's question and the
      assistant's answer (with sources) are persisted.
    * ``conversation_id`` for an existing conversation → the last
      ``history_limit`` messages are pre-pended to the RAG prompt as
      context, and the new turn is appended.
    * ``conversation_id`` for a missing conversation → 404.
    """

    store = _store_for(request)

    # Resolve the target conversation
    conversation: StoredConversation
    if payload.conversation_id:
        existing = store.get_conversation(payload.conversation_id)
        if existing is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
        conversation = existing
        if payload.history_limit > 0:
            prior = store.recent_messages(conversation.id, limit=payload.history_limit)
        else:
            prior = []
    else:
        conversation = store.create_conversation(seed_text=payload.question)
        prior = []

    history_text = _format_history(prior) if prior else ""

    hits, rerank_info = _retrieve_with_rerank(store, payload.question, payload.top_k)
    retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())
    answer, provider, model, elapsed_ms = _generate_answer(
        payload.question, hits, request, history=history_text
    )

    # Persist the new turn (user + assistant) — never block the response on this
    try:
        store.add_message(conversation.id, "user", payload.question)
        source_payload = [hit_out.model_dump() for hit_out in (_hit_to_out(h) for h in hits)]
        store.add_message(
            conversation.id,
            "assistant",
            answer,
            sources=source_payload,
            provider=provider,
            model=model,
        )
    except (ValueError, LookupError) as exc:
        LOGGER.warning("Failed to persist conversation turn: %s", exc)

    return AskResponse(
        question=payload.question,
        answer=answer,
        sources=[_hit_to_out(hit) for hit in hits],
        provider=provider,
        model=model,
        elapsed_ms=elapsed_ms,
        rerank=rerank_info,
        retrieval=RetrievalReportOut(**retrieval_out) if retrieval_out else None,
        conversation_id=conversation.id,
    )


# ----------------------------------------------------------------------
# Streaming /ask
# ----------------------------------------------------------------------


def _sse_event(event: str, data: Any) -> str:
    """Format an SSE message: event + JSON-encoded data + blank line."""

    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_extractive(text: str) -> AsyncIterator[str]:
    """Yield extractive fallback as one big chunk (no real streaming)."""

    yield text


async def _stream_legacy(legacy, prompt: str) -> AsyncIterator[str]:
    """Yield from a sync legacy provider — one chunk."""

    import asyncio

    generate = getattr(legacy, "generate", None)
    if not callable(generate):
        return
    text = await asyncio.to_thread(generate, prompt)
    cleaned = (str(text) if text is not None else "").strip()
    if cleaned:
        yield cleaned


@protected.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """Streamed RAG answer over Server-Sent Events.

    Event sequence:

    * ``event: meta``  — ``{conversation_id, sources, rerank, retrieval}``
    * ``event: token`` — ``{text: "<delta>"}`` (multiple)
    * ``event: done``  — ``{provider, model, elapsed_ms}``
    * ``event: error`` — ``{message}`` (on transport failure; stream then closes)

    Conversation semantics mirror :func:`ask`: missing ``conversation_id``
    creates a new conversation. User question + final accumulated
    assistant answer are persisted on stream completion.
    """

    store = _store_for(request)

    if payload.conversation_id:
        existing = store.get_conversation(payload.conversation_id)
        if existing is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
        conversation = existing
        if payload.history_limit > 0:
            prior = store.recent_messages(conversation.id, limit=payload.history_limit)
        else:
            prior = []
    else:
        conversation = store.create_conversation(seed_text=payload.question)
        prior = []

    history_text = _format_history(prior) if prior else ""
    hits, rerank_info = _retrieve_with_rerank(store, payload.question, payload.top_k)
    source_payload = [_hit_to_out(hit).model_dump() for hit in hits]
    retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())

    async def event_generator() -> AsyncIterator[str]:
        start = time.perf_counter()

        meta = {
            "conversation_id": conversation.id,
            "sources": source_payload,
            "rerank": rerank_info.model_dump() if rerank_info else None,
            "retrieval": retrieval_out,
        }
        yield _sse_event("meta", meta)

        if not hits:
            answer = (
                "В базе знаний пока нет данных, релевантных вопросу. "
                "Добавьте документы и повторите запрос."
            )
            yield _sse_event("token", {"text": answer})
            try:
                store.add_message(conversation.id, "user", payload.question)
                store.add_message(conversation.id, "assistant", answer, sources=[], provider="none")
            except (ValueError, LookupError) as exc:
                LOGGER.warning("Failed to persist empty-KB turn: %s", exc)
            yield _sse_event(
                "done",
                {
                    "provider": "none",
                    "model": None,
                    "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
                },
            )
            return

        prompt = _build_rag_prompt(payload.question, hits, history=history_text)
        provider = kb_llm.select_provider()
        provider_name = "extractive"
        model_name: Optional[str] = None
        chunks: list[str] = []

        async def emit_stream(source: AsyncIterator[str]) -> bool:
            """Forward chunks from *source*. Returns True on success."""

            received_any = False
            try:
                async for delta in source:
                    if not delta:
                        continue
                    received_any = True
                    chunks.append(delta)
                    yield _sse_event("token", {"text": delta})
            except kb_llm.LLMTransportError as exc:
                LOGGER.warning("LLM stream transport error: %s", exc)
                yield _sse_event("error", {"message": f"LLM transport error: {exc}"})
                return
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("LLM streaming failure")
                yield _sse_event("error", {"message": "internal streaming error"})
                return
            if not received_any:
                yield _sse_event("error", {"message": "empty completion"})

        stream_fn = getattr(provider, "generate_stream", None) if provider is not None else None
        if provider is not None and callable(stream_fn):
            provider_name = provider.name
            model_name = provider.model
            async for evt in emit_stream(stream_fn(prompt, system=_RAG_SYSTEM_PROMPT)):
                yield evt

        # If primary provider produced nothing — try legacy then extractive
        if not chunks:
            legacy = getattr(getattr(request, "app", None), "state", None)
            legacy = getattr(legacy, "llm_provider", None) if legacy is not None else None
            if legacy is not None:
                provider_name = str(getattr(legacy, "name", "legacy"))
                model_name = None
                async for evt in emit_stream(_stream_legacy(legacy, prompt)):
                    yield evt

        if not chunks:
            provider_name = "extractive"
            model_name = None
            extractive = _extractive_answer(hits)
            async for evt in emit_stream(_stream_extractive(extractive)):
                yield evt

        full_answer = "".join(chunks).strip() or "(empty)"
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)

        try:
            store.add_message(conversation.id, "user", payload.question)
            store.add_message(
                conversation.id,
                "assistant",
                full_answer,
                sources=source_payload,
                provider=provider_name,
                model=model_name,
            )
        except (ValueError, LookupError) as exc:
            LOGGER.warning("Failed to persist streamed turn: %s", exc)

        yield _sse_event(
            "done",
            {"provider": provider_name, "model": model_name, "elapsed_ms": elapsed_ms},
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx response buffering
            "Connection": "keep-alive",
        },
    )


# ----------------------------------------------------------------------
# Conversations
# ----------------------------------------------------------------------


@protected.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(payload: ConversationCreate, request: Request) -> ConversationOut:
    """Create an empty conversation. Title defaults to «Новый диалог»."""

    store = _store_for(request)
    conv = store.create_conversation(title=payload.title)
    return _conversation_to_out(conv)


@protected.get("/conversations", response_model=List[ConversationOut])
def list_conversations(request: Request, limit: int = 100) -> List[ConversationOut]:
    """List conversations ordered by most recently updated."""

    limit = max(1, min(int(limit), 500))
    store = _store_for(request)
    return [_conversation_to_out(c) for c in store.list_conversations(limit=limit)]


@protected.get("/conversations/{conv_id}", response_model=ConversationDetail)
def get_conversation_detail(conv_id: str, request: Request) -> ConversationDetail:
    """Return a conversation with all its messages (chronological)."""

    store = _store_for(request)
    conv = store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
    messages = store.list_messages(conv.id)
    return ConversationDetail(
        **_conversation_to_out(conv).model_dump(),
        messages=[_message_to_out(m) for m in messages],
    )


@protected.patch("/conversations/{conv_id}", response_model=ConversationOut)
def rename_conversation(
    conv_id: str, payload: ConversationRename, request: Request
) -> ConversationOut:
    """Update a conversation's display title."""

    store = _store_for(request)
    updated = store.rename_conversation(conv_id, payload.title)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
    return _conversation_to_out(updated)


@protected.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, request: Request) -> dict[str, Any]:
    """Delete a conversation and all its messages."""

    store = _store_for(request)
    if not store.delete_conversation(conv_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="CONVERSATION_NOT_FOUND")
    return {"ok": True, "id": conv_id}


# Wire the public and protected sub-routers into the top-level ``router``
# that ``app.api.router`` mounts under ``/api/kb``.
router.include_router(public)
router.include_router(protected)

# W4 — live feedback collection endpoints
from app.api.kb_feedback import router as kb_feedback_router  # noqa: E402

router.include_router(kb_feedback_router)


__all__ = ["router"]
