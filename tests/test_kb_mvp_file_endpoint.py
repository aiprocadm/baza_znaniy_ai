"""Test GET /api/kb/documents/{id}/file endpoint."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_and_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import config as _cfg
    if hasattr(_cfg, "get_settings"):
        _cfg.get_settings.cache_clear()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app, store, tmp_path


@pytest.fixture
def uploaded_pdf(app_and_store, monkeypatch):
    """Upload one PDF and return (client, doc_id, tmp_path)."""
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha"), (2, "beta")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    pdf_bytes = b"%PDF-1.4\nhello\n"
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    return client, resp.json()["id"], tmp_path, pdf_bytes


def test_file_endpoint_returns_pdf(uploaded_pdf):
    client, doc_id, _, pdf_bytes = uploaded_pdf
    resp = client.get(f"/api/kb/documents/{doc_id}/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    # Exact match — guards against double-extension regression
    assert resp.headers["content-disposition"] == 'inline; filename="doc.pdf"'
    assert resp.content == pdf_bytes


def test_file_endpoint_404_for_unknown_doc(app_and_store):
    app, _, _ = app_and_store
    client = TestClient(app)
    resp = client.get("/api/kb/documents/99999/file")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "DOCUMENT_NOT_FOUND"


def test_file_endpoint_404_for_doc_without_original(app_and_store):
    app, store, _ = app_and_store
    store.add_document("txt", text="hello")
    client = TestClient(app)
    resp = client.get("/api/kb/documents/1/file")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "NO_ORIGINAL_FILE"


def test_file_endpoint_410_when_blob_missing(uploaded_pdf):
    client, doc_id, tmp_path, _ = uploaded_pdf
    # Remove blob from disk while DB still references it
    blob_path = tmp_path / "kb_files" / f"{doc_id}.pdf"
    blob_path.unlink()
    resp = client.get(f"/api/kb/documents/{doc_id}/file")
    assert resp.status_code == 410
    assert resp.json()["detail"] == "FILE_DELETED"


def test_file_endpoint_path_traversal_returns_500(app_and_store):
    """If DB has been tampered to contain ../ in file_relpath, refuse."""
    app, store, tmp_path = app_and_store
    store.add_document("pdf", pages=[(1, "x")], source="file", filename="x.pdf")

    # Inject a malicious relpath directly via sqlite
    import sqlite3
    conn = sqlite3.connect(store.db_path)
    conn.execute(
        "UPDATE kb_documents SET has_original_file=1, file_relpath=? WHERE id=1",
        ("kb_files/../../../etc/passwd",),
    )
    conn.commit()
    conn.close()

    client = TestClient(app)
    resp = client.get("/api/kb/documents/1/file")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "STORAGE_ERROR"


def test_file_endpoint_requires_auth_when_key_set(app_and_store, monkeypatch):
    """When KB_API_KEY is set, the endpoint demands X-API-Key header."""
    app, store, tmp_path = app_and_store
    monkeypatch.setenv("KB_API_KEY", "secret-key-xxx")
    # Force kb_auth to re-read env
    from app.api import kb_auth as _ka
    if hasattr(_ka, "_load_api_key"):
        _ka._load_api_key.cache_clear()

    class FakeResult:
        pages = [(1, "alpha")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    # Upload requires auth too
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4\n"), "application/pdf")},
        headers={"X-API-Key": "secret-key-xxx"},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    # Without header → 401
    resp = client.get(f"/api/kb/documents/{doc_id}/file")
    assert resp.status_code == 401

    # With header → 200
    resp = client.get(
        f"/api/kb/documents/{doc_id}/file",
        headers={"X-API-Key": "secret-key-xxx"},
    )
    assert resp.status_code == 200


def test_file_endpoint_sanitises_quote_in_filename(app_and_store, monkeypatch):
    """If a malicious filename slipped past upload validation, the header
    is still well-formed (no broken HTTP header parsing)."""
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    # FastAPI/Starlette would normally reject this at upload, but for the
    # endpoint test we inject the doc directly via the store to focus on
    # the response-side sanitisation.
    doc = store.add_document(
        "evil",
        pages=[(1, "alpha")],
        source="file",
        filename='evil".pdf',
        mime_type="application/pdf",
    )
    # Write a real blob so we get 200, not 410
    kb_dir = tmp_path / "kb_files"
    kb_dir.mkdir(parents=True, exist_ok=True)
    blob = kb_dir / f"{doc.id}.pdf"
    blob.write_bytes(b"%PDF-1.4\n")
    store.update_file_metadata(doc.id, file_relpath=f"kb_files/{doc.id}.pdf")

    resp = client.get(f"/api/kb/documents/{doc.id}/file")
    assert resp.status_code == 200
    # The unsafe `"` is replaced with `_`
    assert resp.headers["content-disposition"] == 'inline; filename="evil_.pdf"'
