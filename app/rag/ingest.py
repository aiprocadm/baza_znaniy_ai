from __future__ import annotations

import io
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Protocol

from docx import Document
from pypdf import PdfReader

try:  # pragma: no cover - dependency provided in production image
    import tiktoken
except ImportError:  # pragma: no cover - fallback used in tests
    tiktoken = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: List[int]) -> str:
        ...


class _CharTokenizer:
    """Very small tokenizer used as a last-resort fallback."""

    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(token) for token in tokens)


_TOKENIZER: Optional[_Tokenizer] = None


def _load_tiktoken(name: str) -> Optional[_Tokenizer]:
    if tiktoken is None:  # pragma: no cover - handled in tests
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to load tokenizer '%s': %s", name, exc)
        try:
            return tiktoken.encoding_for_model(name)
        except Exception:  # pragma: no cover - final fallback
            return None


@lru_cache(maxsize=1)
def _default_tokenizer() -> _Tokenizer:
    name = os.getenv("RAG_TOKENIZER_NAME", "cl100k_base")
    tokenizer = _load_tiktoken(name)
    if tokenizer is None and name != "text-embedding-3-small":
        tokenizer = _load_tiktoken("text-embedding-3-small")
    return tokenizer or _CharTokenizer()


def _get_tokenizer() -> _Tokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _default_tokenizer()
    return _TOKENIZER


def _clean(text: str) -> str:
    """Collapse whitespace and trim the provided *text*."""

    return re.sub(r"\s+", " ", text).strip()


def _chunk(
    text: str,
    *,
    chunk: int = 900,
    overlap: int = 140,
    encoder: Optional[_Tokenizer] = None,
) -> List[str]:
    """Split *text* into token aware windows."""

    if not text:
        return []

    tokenizer = encoder or _get_tokenizer()
    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    chunk = max(int(chunk), 1)
    overlap = max(int(overlap), 0)
    if overlap >= chunk:
        overlap = chunk - 1 if chunk > 1 else 0

    pieces: List[str] = []
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + chunk, total)
        window_tokens = tokens[start:end]
        pieces.append(tokenizer.decode(window_tokens))
        if end >= total:
            break
        start = max(end - overlap, 0)

    return pieces


def _parse_pdf(data: bytes) -> List[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    pages: List[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - defensive for exotic PDFs
            text = ""
        pages.append((index, _clean(text)))
    return pages


def _parse_docx(data: bytes) -> List[tuple[int, str]]:
    document = Document(io.BytesIO(data))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return [(1, _clean(text))]


def _parse_txt(data: bytes) -> List[tuple[int, str]]:
    text = data.decode("utf-8", errors="ignore")
    return [(1, _clean(text))]


def parse_and_chunk(filename: str, data: bytes) -> List[Dict[str, object]]:
    """Parse *data* originating from *filename* and return chunk payloads."""

    name = (filename or "").strip()
    if not name:
        return []
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext == "pdf":
        pages = _parse_pdf(data)
    elif ext == "docx":
        pages = _parse_docx(data)
    elif ext == "txt":
        pages = _parse_txt(data)
    else:
        return []

    chunk_size = int(os.getenv("RAG_CHUNK", "900"))
    overlap = int(os.getenv("RAG_OVERLAP", "140"))
    tokenizer = _get_tokenizer()

    chunks: List[Dict[str, object]] = []
    for page_number, page_text in pages:
        if not page_text:
            continue
        for piece in _chunk(page_text, chunk=chunk_size, overlap=overlap, encoder=tokenizer):
            chunks.append({"file": name, "page": page_number, "text": piece})
    return chunks


__all__ = ["_chunk", "_clean", "_get_tokenizer", "parse_and_chunk"]
