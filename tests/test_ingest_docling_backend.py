from __future__ import annotations

import importlib


def test_docling_backend_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("DOCUMENT_PARSER_BACKEND", "docling")
    monkeypatch.setenv("DOCLING_ENABLED", "false")
    from app.ingest import chunking

    importlib.reload(chunking)
    assert chunking._resolve_parser_backend() == "legacy"


def test_docling_backend_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("DOCUMENT_PARSER_BACKEND", "unexpected")
    from app.ingest import chunking

    importlib.reload(chunking)
    assert chunking._resolve_parser_backend() == "legacy"


def test_parse_document_auto_success_docling(monkeypatch):
    from app.ingest import chunking

    monkeypatch.setenv("DOCUMENT_PARSER_BACKEND", "auto")
    monkeypatch.setenv("DOCLING_ENABLED", "true")

    class FakeAdapter:
        SUPPORTED_MIME = chunking.DoclingParserAdapter.SUPPORTED_MIME

        def parse(self, filename, raw_bytes):
            return [(1, "docling parsed")]

    monkeypatch.setattr(chunking, "DoclingParserAdapter", FakeAdapter)
    monkeypatch.setattr(chunking, "_resolve_parser_backend", lambda explicit_backend=None: "auto")

    result = chunking.parse_document("sample.pdf", b"pdf")
    assert result.parser_backend_used == "docling"
    assert result.fallback_reason is None
    assert result.pages == [(1, "docling parsed")]


def test_parse_document_auto_fallback_to_legacy(monkeypatch):
    from app.ingest import chunking

    monkeypatch.setattr(chunking, "_resolve_parser_backend", lambda explicit_backend=None: "auto")

    class FakeAdapter:
        SUPPORTED_MIME = chunking.DoclingParserAdapter.SUPPORTED_MIME

        def parse(self, filename, raw_bytes):
            raise RuntimeError("boom")

    monkeypatch.setattr(chunking, "DoclingParserAdapter", FakeAdapter)
    monkeypatch.setattr(chunking, "_iter_pdf_pages", lambda data: iter([(1, "legacy text")]))

    result = chunking.parse_document("sample.pdf", b"pdf")
    assert result.parser_backend_used == "legacy"
    assert "boom" in (result.fallback_reason or "")
    assert result.pages == [(1, "legacy text")]
