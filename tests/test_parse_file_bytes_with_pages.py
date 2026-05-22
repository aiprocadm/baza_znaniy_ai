"""Test _parse_file_bytes_with_pages helper."""
from __future__ import annotations

import pytest

from app.api.kb_mvp import _parse_file_bytes_with_pages


def test_parse_txt_returns_single_page():
    pages, mime = _parse_file_bytes_with_pages("notes.txt", b"hello world")
    assert pages == [(1, "hello world")]
    assert mime == "text/plain"


def test_parse_md_returns_single_page():
    pages, mime = _parse_file_bytes_with_pages("notes.md", b"# Title\n\nbody")
    assert pages == [(1, "# Title\n\nbody")]
    assert mime == "text/markdown"


def test_parse_empty_extension_falls_back_to_text():
    pages, mime = _parse_file_bytes_with_pages("noext", b"raw bytes")
    assert pages == [(1, "raw bytes")]
    assert mime == "text/plain"


def test_parse_rich_format_returns_multiple_pages(monkeypatch):
    """When parse_document yields pages, helper preserves them."""
    class FakeResult:
        pages = [(1, "page 1 text"), (2, "page 2 text")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    def fake_parse_document(filename, data):
        return FakeResult()

    import app.api.kb_mvp as kb_mvp
    monkeypatch.setattr("app.ingest.chunking.parse_document", fake_parse_document)

    pages, mime = _parse_file_bytes_with_pages("doc.pdf", b"%PDF-1.4")
    assert pages == [(1, "page 1 text"), (2, "page 2 text")]
    assert mime == "application/pdf"


def test_parse_rich_format_drops_empty_pages(monkeypatch):
    class FakeResult:
        pages = [(1, ""), (2, "real"), (3, "  ")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())
    pages, mime = _parse_file_bytes_with_pages("doc.pdf", b"%PDF-1.4")
    assert pages == [(2, "real")]
    assert mime == "application/pdf"
