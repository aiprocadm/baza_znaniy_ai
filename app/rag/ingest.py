from __future__ import annotations

import io
import logging
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Protocol

try:  # pragma: no cover - dependency provided via requirements
    import tiktoken
except ImportError:  # pragma: no cover - used in tests when dependency missing
    tiktoken = None  # type: ignore

from docx import Document
from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    """Subset of the tiktoken ``Encoding`` interface used by the tests."""

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: List[int]) -> str:
        ...


class _CharTokenizer:
    """Fallback tokenizer that operates on individual characters."""

    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(token) for token in tokens)


_TOKENIZER: Optional[_Tokenizer] = None


@lru_cache(maxsize=1)
def _byte_fallback() -> "tiktoken.core.Encoding":  # type: ignore[name-defined]
    """Return a byte-level tokenizer compatible with the tiktoken API."""

    mergeable_ranks = {bytes([i]): i for i in range(256)}
    return tiktoken.Encoding(  # type: ignore[call-arg]
        name="byte_fallback",
        pat_str=r"(?s:.)",
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


def _get_tokenizer() -> _Tokenizer:
    """Return the tokenizer used to measure token lengths."""

    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    if tiktoken is None:
        _TOKENIZER = _CharTokenizer()
        return _TOKENIZER

    name = os.getenv("RAG_TOKENIZER_NAME", "cl100k_base")
    try:
        _TOKENIZER = tiktoken.get_encoding(name)
    except Exception:  # pragma: no cover - defensive fall-back
        try:
            _TOKENIZER = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:  # pragma: no cover - fallback for unusual environments
            _TOKENIZER = _byte_fallback()
    return _TOKENIZER


def _clean(text: str) -> str:
    """Normalise whitespace extracted from source documents."""

    return re.sub(r"\s+", " ", text).strip()


def _normalise_window_size(value: int, minimum: int = 1) -> int:
    value = int(value)
    return minimum if value < minimum else value


def _normalise_overlap(chunk: int, overlap: int) -> int:
    overlap = 0 if overlap < 0 else int(overlap)
    if chunk <= 1:
        return 0
    return min(overlap, chunk - 1)


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
    token_ids = tokenizer.encode(text)
    if not token_ids:
        return []

    window = _normalise_window_size(chunk)
    step_overlap = _normalise_overlap(window, overlap)

    pieces: List[str] = []
    start = 0
    total = len(token_ids)

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


def parse_and_chunk(filename: str, data: bytes) -> List[Dict[str, object]]:
    """Parse ``data`` according to ``filename`` extension and chunk the text."""

    ext = filename.rsplit(".", 1)[-1].lower()
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

    chunks: List[Dict[str, object]] = []
    for page, text in pages:
        for piece in _chunk(text, chunk=chunk_size, overlap=overlap, encoder=tokenizer):
            chunks.append({"file": filename, "page": page, "text": piece})
    return chunks


__all__ = ["_chunk", "_clean", "_get_tokenizer", "parse_and_chunk"]
