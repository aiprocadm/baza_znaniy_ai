"""Shared infrastructure for the MVP /api/kb endpoint modules:
routers, constants, store accessor, model converters, file parsing."""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.api.kb_auth import require_api_key
from app.services.kb_store import (
    Conversation as StoredConversation,
    Document as StoredDocument,
    KnowledgeBaseStore,
    Message as StoredMessage,
    SearchHit,
    get_store,
)
from .schemas import ConversationOut, DocumentOut, HitOut, MessageOut

LOGGER = logging.getLogger(__name__)

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
