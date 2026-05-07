"""Docling-backed parsing helpers for ingestion."""

from __future__ import annotations

import io
from dataclasses import dataclass

from app.core.config import get_settings


@dataclass(frozen=True)
class DoclingParseOptions:
    timeout: float
    max_pages: int | None


def _options_from_settings() -> DoclingParseOptions:
    settings = get_settings()
    return DoclingParseOptions(
        timeout=float(getattr(settings, "docling_timeout", 60.0)),
        max_pages=getattr(settings, "docling_max_pages", None),
    )


class DoclingBackend:
    """Thin adapter around Docling ``DocumentConverter``."""

    SUPPORTED_MIME = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",
        "text/markdown",
    }

    def parse(self, filename: str, raw_bytes: bytes) -> list[tuple[int, str]]:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Docling import failed: {exc}") from exc

        options = _options_from_settings()
        try:
            converter = DocumentConverter()
            result = converter.convert(source=io.BytesIO(raw_bytes), timeout=options.timeout)
            markdown = result.document.export_to_markdown()
        except Exception as exc:
            raise RuntimeError(f"Docling conversion failed: {exc}") from exc

        lines = [line.strip() for line in str(markdown).splitlines() if line.strip()]
        if options.max_pages and isinstance(options.max_pages, int) and options.max_pages > 0:
            lines = lines[: options.max_pages]
        if not lines:
            return []
        return [(1, "\n".join(lines))]
