"""Test PDF blob persistence in upload_document."""
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
    # Pin data_dir to tmp_path so the test owns var/data/kb_files/
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Force settings reload — pattern depends on app.core.config caching
    from app.core import config as _cfg
    if hasattr(_cfg, "get_settings"):
        _cfg.get_settings.cache_clear()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")

    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app, store, tmp_path


def _minimal_pdf_bytes() -> bytes:
    """Minimal PDF header bytes; real parsing is patched in the tests
    that use this. The bytes are just the bytes that will be persisted
    as the blob — content doesn't matter."""
    return b"%PDF-1.4\n%minimal\n"


def test_upload_pdf_persists_blob(app_and_store, monkeypatch):
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha beta " * 30), (2, "gamma delta " * 30)]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    pdf_bytes = _minimal_pdf_bytes()
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"title": "Doc"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    doc_id = body["id"]

    # Blob saved
    blob_path = tmp_path / "kb_files" / f"{doc_id}.pdf"
    assert blob_path.exists(), f"blob missing at {blob_path}"
    assert blob_path.read_bytes() == pdf_bytes

    # has_original_file should now be true
    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc.has_original_file is True
    assert doc.file_relpath == f"kb_files/{doc_id}.pdf"


def test_upload_non_pdf_does_not_save_blob(app_and_store, monkeypatch):
    app, store, tmp_path = app_and_store

    client = TestClient(app)
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    kb_files = tmp_path / "kb_files"
    if kb_files.exists():
        assert not any(kb_files.iterdir()), "non-PDF should not produce a blob"

    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc.has_original_file is False
    assert doc.file_relpath is None


def test_upload_pdf_orphan_tmp_cleaned_on_parse_error(app_and_store, monkeypatch):
    """If parse_document raises, no tmp-* blob should remain."""
    app, store, tmp_path = app_and_store

    def broken_parse(*_):
        raise RuntimeError("synthetic parse failure")

    monkeypatch.setattr("app.ingest.chunking.parse_document", broken_parse)

    client = TestClient(app)
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4\n"), "application/pdf")},
    )
    assert resp.status_code == 422, resp.text

    kb_files = tmp_path / "kb_files"
    if kb_files.exists():
        leftovers = [p for p in kb_files.iterdir() if p.name.startswith(".tmp-")]
        assert leftovers == [], f"orphan tmp blobs found: {leftovers}"


def test_delete_document_removes_blob(app_and_store, monkeypatch):
    """DELETE on a doc with original_file removes both DB row and blob."""
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4\n"), "application/pdf")},
    )
    doc_id = resp.json()["id"]
    blob_path = tmp_path / "kb_files" / f"{doc_id}.pdf"
    assert blob_path.exists()

    resp = client.delete(f"/api/kb/documents/{doc_id}")
    assert resp.status_code == 200, resp.text

    assert not blob_path.exists(), "blob should be removed after delete"
    assert store.get_document(doc_id) is None


def test_delete_document_without_blob_no_error(app_and_store):
    """DELETE on a non-PDF doc completes even though no blob exists."""
    app, store, tmp_path = app_and_store

    client = TestClient(app)
    client.post(
        "/api/kb/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")},
    )
    doc_id = 1  # first doc

    resp = client.delete(f"/api/kb/documents/{doc_id}")
    assert resp.status_code == 200
