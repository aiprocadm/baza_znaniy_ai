"""Docling-backed parsing helpers for ingestion."""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings


@dataclass(frozen=True)
class DoclingParseOptions:
    timeout: float
    max_pages: int | None
    ocr_enabled: bool


def _options_from_settings() -> DoclingParseOptions:
    settings = get_settings()
    return DoclingParseOptions(
        timeout=float(getattr(settings, "docling_timeout", 60.0)),
        max_pages=getattr(settings, "docling_max_pages", None),
        ocr_enabled=bool(getattr(settings, "docling_ocr_enabled", False)),
    )


def _extract_page_texts(result: Any) -> list[tuple[int, str]]:
    """Return per-page text, preferring Markdown export for richer chunks.

    Docling's ``export_to_markdown()`` preserves tables, headings, lists
    and figure captions; the raw ``page.text`` / ``page.content`` loses
    that structure. For RAG-grade retrieval we want the Markdown first
    and fall back to plain text only if the Markdown export is empty
    or unavailable (e.g. older Docling versions).
    """

    document = getattr(result, "document", None)
    if document is None:
        return []

    page_items = getattr(document, "pages", None)
    if page_items:
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(page_items, start=1):
            text = ""
            if hasattr(page, "export_to_markdown"):
                try:
                    text = str(page.export_to_markdown() or "").strip()
                except Exception:
                    text = ""
            if not text:
                for candidate in ("text", "content"):
                    value = getattr(page, candidate, None)
                    if value:
                        text = str(value).strip()
                        break
            if text:
                pages.append((index, text))
        if pages:
            return pages

    markdown = ""
    if hasattr(document, "export_to_markdown"):
        markdown = str(document.export_to_markdown() or "").strip()
    if not markdown:
        return []
    return [(1, markdown)]


class DoclingBackend:
    """Thin adapter around Docling ``DocumentConverter``."""

    SUPPORTED_MIME = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",
        "text/markdown",
        "text/html",
    }

    def parse(self, filename: str, raw_bytes: bytes) -> list[tuple[int, str]]:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Docling import failed: {exc}") from exc

        options = _options_from_settings()
        converter = DocumentConverter()

        convert_error: Exception | None = None
        result: Any | None = None
        for source in (io.BytesIO(raw_bytes), raw_bytes):
            try:
                result = converter.convert(source=source, timeout=options.timeout)
                break
            except Exception as exc:
                convert_error = exc

        if result is None:
            suffix = Path(filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
                temp.write(raw_bytes)
                temp.flush()
                try:
                    result = converter.convert(source=temp.name, timeout=options.timeout)
                except Exception as exc:
                    convert_error = exc

        if result is None:
            raise RuntimeError(f"Docling conversion failed: {convert_error}")

        pages = _extract_page_texts(result)
        if options.max_pages and options.max_pages > 0:
            pages = pages[: options.max_pages]
        return pages
