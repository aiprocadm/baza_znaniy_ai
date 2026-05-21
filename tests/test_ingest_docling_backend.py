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


def test_parse_document_auto_docling(monkeypatch):
    from app.ingest import chunking

    monkeypatch.setattr(chunking, "_resolve_parser_backend", lambda explicit_backend=None: "auto")

    class FakeAdapter:
        SUPPORTED_MIME = chunking.DoclingBackend.SUPPORTED_MIME

        def parse(self, filename, raw_bytes):
            return [(1, "docling parsed")]

    monkeypatch.setattr(chunking, "DoclingBackend", FakeAdapter)

    result = chunking.parse_document("sample.pdf", b"pdf")
    assert result.parser_backend_used == "docling"
    assert result.fallback_reason is None
    assert result.pages == [(1, "docling parsed")]


def test_parse_document_auto_legacy_fallback(monkeypatch):
    from app.ingest import chunking

    monkeypatch.setattr(chunking, "_resolve_parser_backend", lambda explicit_backend=None: "auto")

    class FakeAdapter:
        SUPPORTED_MIME = chunking.DoclingBackend.SUPPORTED_MIME

        def parse(self, filename, raw_bytes):
            raise RuntimeError("boom")

    monkeypatch.setattr(chunking, "DoclingBackend", FakeAdapter)
    monkeypatch.setattr(chunking, "_iter_pdf_pages", lambda data: iter([(1, "legacy text")]))

    result = chunking.parse_document("sample.pdf", b"pdf")
    assert result.parser_backend_used == "legacy"
    assert "boom" in (result.fallback_reason or "")
    assert result.pages == [(1, "legacy text")]


def test_extract_page_texts_prefers_markdown() -> None:
    """Markdown export должен иметь приоритет над raw page.text для RAG."""

    from app.ingest.docling_backend import _extract_page_texts

    class FakePage:
        text = "Plain text without tables"

        def export_to_markdown(self) -> str:
            return "# Heading\n\n| A | B |\n|---|---|\n| 1 | 2 |"

    class FakeDocument:
        pages = [FakePage()]

    class FakeResult:
        document = FakeDocument()

    pages = _extract_page_texts(FakeResult())
    assert len(pages) == 1
    assert pages[0][0] == 1
    # Markdown с таблицей должен выиграть у plain text
    assert "| A | B |" in pages[0][1]
    assert "Plain text" not in pages[0][1]


def test_extract_page_texts_falls_back_to_text_when_markdown_empty() -> None:
    """Если export_to_markdown() пустой — берём page.text как fallback."""

    from app.ingest.docling_backend import _extract_page_texts

    class FakePage:
        text = "Plain text fallback"

        def export_to_markdown(self) -> str:
            return ""

    class FakeDocument:
        pages = [FakePage()]

    class FakeResult:
        document = FakeDocument()

    pages = _extract_page_texts(FakeResult())
    assert pages == [(1, "Plain text fallback")]


def test_parse_document_docling_hard_fail_with_fallback(monkeypatch):
    from app.ingest import chunking

    monkeypatch.setattr(chunking, "_resolve_parser_backend", lambda explicit_backend=None: "docling")

    class FakeAdapter:
        SUPPORTED_MIME = chunking.DoclingBackend.SUPPORTED_MIME

        def parse(self, filename, raw_bytes):
            raise RuntimeError("docling unavailable")

    monkeypatch.setattr(chunking, "DoclingBackend", FakeAdapter)
    monkeypatch.setattr(chunking, "_iter_pdf_pages", lambda data: iter([(1, "legacy text")]))

    result = chunking.parse_document("sample.pdf", b"pdf")
    assert result.parser_backend_used == "legacy"
    assert "docling unavailable" in (result.fallback_reason or "")
    assert result.pages == [(1, "legacy text")]
