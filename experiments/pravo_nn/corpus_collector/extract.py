"""Turn a raw fetched document into clean `Article`s.

`strip_to_text` is the only source-specific piece (HTML tag-strip below); the
splitter + normalizer are source-independent. Output is the contract every
downstream consumer depends on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTICLE_RE = re.compile(r"(Статья\s+\d+(?:\.\d+)?[^\n]*)")
_PAGE_NUM_RE = re.compile(r"^\d+$")
_WS_RE = re.compile(r"[ \t ]+")
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"'}


@dataclass(frozen=True)
class Article:
    code: str
    article: str
    text: str
    source_url: str
    date: str


def normalize_whitespace(text: str) -> str:
    """Collapse intra-line whitespace, drop standalone page-number lines and
    blank lines."""
    out: list[str] = []
    for line in text.splitlines():
        line = _WS_RE.sub(" ", line).strip()
        if not line or _PAGE_NUM_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def strip_to_text(raw: str) -> str:
    """Source-specific shim. HTML source: drop tags + unescape common entities.
    (Plain-text source: return `raw`. JSON source: return the parsed text field.)"""
    text = _TAG_RE.sub("\n", raw)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return text


def split_articles(text: str, *, code: str, source_url: str, date: str) -> list[Article]:
    """Split normalized text on `Статья N` markers into Articles (marker = the
    `article` field, the following body = `text`)."""
    parts = _ARTICLE_RE.split(text)
    articles: list[Article] = []
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        articles.append(
            Article(code=code, article=marker, text=body, source_url=source_url, date=date)
        )
    return articles


def extract_articles(raw: str, *, code: str, source_url: str, date: str) -> list[Article]:
    text = normalize_whitespace(strip_to_text(raw))
    return split_articles(text, code=code, source_url=source_url, date=date)
