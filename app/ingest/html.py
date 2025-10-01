"""Helpers for converting HTML content into plain text chunks."""

from __future__ import annotations

import html
import re
from typing import List

from app.core.config import Settings

try:  # pragma: no cover - optional dependency
    import html2text
except Exception:  # pragma: no cover - graceful fallback when unavailable
    html2text = None  # type: ignore[assignment]

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)(?:\\s[^>]*)?>.*?</\\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_BREAK_RE = re.compile(
    r"</(?:p|div|h[1-6]|section|article|li|tr|table|blockquote|pre|br|hr)>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_UNDERLINE_RE = re.compile(r"^\\s*=+\\s*$", re.MULTILINE)
_LEADING_HASH_RE = re.compile(r"^\\s*#+\\s*", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\\s+")


def _clean(text: str) -> str:
    """Normalise whitespace inside *text* and trim the ends."""

    return _WHITESPACE_RE.sub(" ", text).strip()


def _apply_converter_settings(converter: object, settings: Settings) -> None:
    """Update *converter* attributes based on ``html2text`` settings."""

    setattr(converter, "body_width", max(int(settings.html2text_bodywidth), 0))
    setattr(converter, "ignore_links", not bool(settings.html2text_links))
    setattr(converter, "ignore_images", bool(settings.html2text_ignore_images))
    setattr(converter, "ignore_emphasis", bool(settings.html2text_ignore_emphasis))
    setattr(converter, "inline_links", bool(settings.html2text_inline_links))
    setattr(converter, "single_line_break", bool(settings.html2text_single_line_break))
    setattr(converter, "wrap_links", bool(settings.html2text_wrap_links))
    setattr(converter, "unicode_snob", bool(settings.html2text_unicode_snob))


def _strip_tags_fallback(html_content: str) -> str:
    """Fallback plain text extraction when ``html2text`` is unavailable."""

    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html_content)
    normalised = _BLOCK_BREAK_RE.sub("\n\n", without_scripts)
    stripped = _TAG_RE.sub(" ", normalised)
    return stripped


def _post_process_text(text: str) -> str:
    text = _HEADING_UNDERLINE_RE.sub(" ", text)
    text = _LEADING_HASH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return html.unescape(text)


def _split_sections(text: str) -> List[str]:
    sections: List[str] = []
    buffer: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if buffer:
                merged = _clean(" ".join(buffer))
                if merged:
                    sections.append(merged)
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        merged = _clean(" ".join(buffer))
        if merged:
            sections.append(merged)
    return sections


def html_to_text_sections(html_content: str, *, settings: Settings | None = None) -> List[str]:
    """Convert *html_content* into cleaned plain-text sections."""

    if not html_content:
        return []

    config = settings or Settings()

    if html2text is not None:
        converter = html2text.HTML2Text()
        _apply_converter_settings(converter, config)
        text = converter.handle(html_content)
    else:  # pragma: no cover - executed only when dependency missing
        text = _strip_tags_fallback(html_content)

    processed = _post_process_text(text)
    return _split_sections(processed)


def html_to_plain_text(html_content: str, *, settings: Settings | None = None) -> str:
    """Convert *html_content* into a single plain text string."""

    sections = html_to_text_sections(html_content, settings=settings)
    if not sections:
        return ""
    return "\n\n".join(sections)


__all__ = [
    "html_to_plain_text",
    "html_to_text_sections",
]

