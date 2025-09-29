"""Document ingestion helpers for parsing and chunking content."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Protocol

from docx import Document
from pypdf import PdfReader

try:  # pragma: no cover - tokenizer optional in some environments
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
    """Fallback tokenizer operating on raw characters."""

    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(token) for token in tokens)


_TOKENIZER: Optional[_Tokenizer] = None


def _load_tiktoken(name: str) -> Optional[_Tokenizer]:
    if tiktoken is None:  # pragma: no cover - handled during tests
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


def _normalise_window_size(value: int, minimum: int = 1) -> int:
    value = int(value)
    return minimum if value < minimum else value


def _normalise_overlap(chunk: int, overlap: int) -> int:
    overlap = 0 if overlap < 0 else int(overlap)
    if chunk <= 1:
        return 0
    return min(overlap, chunk - 1)


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
    """Split ``text`` into overlapping windows based on token counts."""

    if not text:
        return []

    tokenizer = encoder or _get_tokenizer()
    window = _normalise_window_size(chunk)

    token_ids = tokenizer.encode(text)
    if not token_ids:
        return []

    total = len(token_ids)
    if total <= window:
        return [text]

    if window <= 1:
        fallback = _CharTokenizer()
        token_ids = fallback.encode(text)
        if not token_ids:
            return []
        tokenizer = fallback
        total = len(token_ids)
        window = _normalise_window_size(min(window, total))

    step_overlap = _normalise_overlap(window, overlap)

    pieces = []
    start = 0
    while start < total:
        end = min(start + window, total)
        tokens = token_ids[start:end]
        pieces.append(tokenizer.decode(tokens))
        if end >= total:
            break
        next_start = end - step_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return pieces


def _iter_pdf_text(data: bytes) -> Iterable[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - pypdf quirks
            text = ""
        cleaned = _clean(text)
        if cleaned:
            yield page_number, cleaned


def _iter_docx_text(data: bytes) -> Iterable[tuple[int, str]]:
    document = Document(io.BytesIO(data))
    text = _clean("\n".join(paragraph.text for paragraph in document.paragraphs))
    if text:
        yield 1, text


def _iter_txt_text(data: bytes) -> Iterable[tuple[int, str]]:
    text = _clean(data.decode("utf-8", errors="ignore"))
    if text:
        yield 1, text


def _hash_chunk(file: str, page: int, text: str) -> str:
    payload = f"{file}\u0000{page}\u0000{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_and_chunk(filename: str, data: bytes) -> List[dict[str, object]]:
    """Parse ``data`` according to ``filename`` extension and chunk the text."""

    name = (filename or "").strip()
    if not name or "." not in name:
        return []

    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        pages = list(_iter_pdf_text(data))
    elif ext == "docx":
        pages = list(_iter_docx_text(data))
    elif ext == "txt":
        pages = list(_iter_txt_text(data))
    else:
        return []

    if not pages:
        return []

    chunk_size = _normalise_window_size(int(os.getenv("RAG_CHUNK", "900")))
    overlap = _normalise_overlap(chunk_size, int(os.getenv("RAG_OVERLAP", "140")))
    tokenizer = _get_tokenizer()

    chunks: List[dict[str, object]] = []
    for page_number, page_text in pages:
        for piece in _chunk(page_text, chunk=chunk_size, overlap=overlap, encoder=tokenizer):
            sha = _hash_chunk(name, page_number, piece)
            chunks.append(
                {
                    "file": name,
                    "page": page_number,
                    "sha256": sha,
                    "text": piece,
                }
            )
    return chunks


__all__ = ["_chunk", "_clean", "_get_tokenizer", "parse_and_chunk"]
