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


@pytest.mark.parametrize("filename,expected_mime_default", [
    ("report.docx", "application/octet-stream"),
    ("deck.pptx", "application/octet-stream"),
    ("data.xlsx", "application/octet-stream"),
])
def test_parse_rich_formats_reach_parse_document(monkeypatch, filename, expected_mime_default):
    """Non-PDF rich formats route through parse_document (not the txt branch)."""
    class FakeResult:
        pages = [(1, "sheet content")]
        metadata = {"document": {}}  # no mime → fallback to octet-stream

    captured: dict = {}

    def fake_parse_document(name, data):
        captured["filename"] = name
        return FakeResult()

    monkeypatch.setattr("app.ingest.chunking.parse_document", fake_parse_document)

    pages, mime = _parse_file_bytes_with_pages(filename, b"binary blob")
    assert pages == [(1, "sheet content")]
    assert mime == expected_mime_default
    assert captured["filename"] == filename


def test_parse_no_extension_strips_whitespace():
    """Unlike legacy _parse_file_bytes, the new helper strips no-ext input
    so trailing whitespace doesn't produce a chunk with a trailing newline."""
    pages, mime = _parse_file_bytes_with_pages("noext", b"  hello world  \n")
    assert pages == [(1, "hello world")]
    assert mime == "text/plain"


def test_parse_no_extension_empty_after_strip_returns_no_pages():
    """Whitespace-only no-ext input results in zero pages."""
    pages, _mime = _parse_file_bytes_with_pages("noext", b"   \n\n  ")
    assert pages == []
