"""Document CRUD + upload endpoints (protected)."""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, List, Optional
from fastapi import File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from .common import (
    LOGGER,
    MAX_UPLOAD_BYTES,
    SUPPORTED_UPLOAD_EXT,
    protected,
    _doc_to_out,
    _extension_for,
    _parse_file_bytes_with_pages,
    _resolve_data_dir,
    _resolve_kb_files_dir,
    _store_for,
)
from .schemas import DocumentCreate, DocumentListItem, DocumentOut


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
        to_unlink: Path | None = candidate
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
            to_unlink = None
        if to_unlink is not None:
            try:
                to_unlink.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("failed to remove blob for doc %d: %s", doc_id, exc)

    if not store.delete_document(doc_id):
        # Race: someone else deleted it between get_document and delete.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")
    return {"ok": True, "id": doc_id}
