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
    """Return a tiny valid PDF that Docling can parse, or a stub that
    `parse_document` recognises. Use a real header + one page."""
    # PDF.js will parse this; for upload-side tests we patch the parser
    # instead of relying on real Docling parsing.
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
