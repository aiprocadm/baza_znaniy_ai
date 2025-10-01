"""Document ingestion helpers for parsing and chunking content."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import time
from functools import lru_cache
from typing import BinaryIO, Iterable, Iterator, List, NamedTuple, Optional, Protocol, Union

from docx import Document
from pypdf import PdfReader

try:  # pragma: no cover - optional dependency used for spreadsheets
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - fallback for minimal environments
    load_workbook = None  # type: ignore[assignment]

import zipfile
from xml.etree import ElementTree as ET

try:  # pragma: no cover - tokenizer optional in some environments
    import tiktoken
except ImportError:  # pragma: no cover - fallback used in tests
    tiktoken = None  # type: ignore[assignment]

from app.observability.metrics import record_document_parse

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


def _ensure_binary_stream(data: Union[bytes, bytearray, BinaryIO]) -> BinaryIO:
    if isinstance(data, (bytes, bytearray)):
        return io.BytesIO(data)
    if hasattr(data, "read"):
        stream = data  # type: ignore[assignment]
        try:  # pragma: no cover - not all streams are seekable
            stream.seek(0)
        except Exception:
            pass
        return stream
    raise TypeError("Unsupported binary payload type")


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
    use_tiktoken = os.getenv("RAG_USE_TIKTOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not use_tiktoken:
        return _CharTokenizer()

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


class _WindowPlan(NamedTuple):
    token_ids: List[int]
    tokenizer: _Tokenizer


def _iterate_windows(
    token_ids: List[int], *, window: int, overlap: int, tokenizer: _Tokenizer
) -> List[str]:
    total = len(token_ids)
    if total == 0:
        return []

    step_overlap = _normalise_overlap(window, overlap)
    pieces: List[str] = []
    start = 0
    while start < total:
        end = min(start + window, total)
        pieces.append(tokenizer.decode(token_ids[start:end]))
        if end >= total:
            break
        next_start = end - step_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return pieces


def _handle_small_token_window(
    text: str,
    token_ids: List[int],
    *,
    window: int,
    overlap: int,
    tokenizer: _Tokenizer,
) -> Optional[_WindowPlan]:
    decoded_text = tokenizer.decode(token_ids)

    if window <= 1:
        fallback_text = decoded_text or text
        char_tokenizer = _CharTokenizer()
        char_token_ids = (
            char_tokenizer.encode(fallback_text) if fallback_text else []
        )
        return _WindowPlan(char_token_ids, char_tokenizer)

    if len(token_ids) > window:
        return None

    if decoded_text and len(decoded_text) > window:
        char_tokenizer = _CharTokenizer()
        char_token_ids = char_tokenizer.encode(decoded_text)
        return _WindowPlan(char_token_ids, char_tokenizer)

    if len(token_ids) == 1:
        fallback_text = decoded_text or text
        if fallback_text:
            char_tokenizer = _CharTokenizer()
            char_token_ids = char_tokenizer.encode(fallback_text)
            if char_token_ids:
                return _WindowPlan(char_token_ids, char_tokenizer)
        return _WindowPlan(token_ids, tokenizer)

    if decoded_text:
        try:
            reencoded = tokenizer.encode(decoded_text)
        except Exception:  # pragma: no cover - defensive fallback
            reencoded = []
        if reencoded:
            if len(reencoded) <= window and len(decoded_text) <= window:
                if reencoded == token_ids:
                    return _WindowPlan(token_ids, tokenizer)
                return _WindowPlan(reencoded, tokenizer)
            fallback_text = decoded_text
        else:
            fallback_text = decoded_text
    else:
        fallback_text = text

    if not fallback_text:
        return _WindowPlan(token_ids, tokenizer)
    char_tokenizer = _CharTokenizer()
    char_token_ids = char_tokenizer.encode(fallback_text)
    if not char_token_ids:
        return _WindowPlan(token_ids, tokenizer)

    return _WindowPlan(char_token_ids, char_tokenizer)


def _chunk(
    text: str,
    *,
    chunk: int = 900,
    overlap: int = 140,
    encoder: Optional[_Tokenizer] = None,
    token_ids: Optional[List[int]] = None,
) -> List[str]:
    """Split ``text`` into overlapping windows based on token counts."""

    if not text:
        return []

    chunk = max(int(chunk), 1)
    try:
        overlap_value = int(overlap)
    except Exception:
        overlap_value = 0
    if overlap_value < 0:
        overlap_value = 0
    if overlap_value >= chunk:
        overlap_value = chunk - 1
    overlap = overlap_value
    step = chunk - overlap or 1

    if encoder is None:
        encoder = _get_tokenizer()

    tokens: List[int] = []
    if encoder is not None:
        if token_ids is not None:
            tokens = list(token_ids)
        else:
            try:
                tokens = list(encoder.encode(text))
            except Exception:  # pragma: no cover - defensive fallback
                tokens = []

    decoded_full = ""
    if tokens:
        try:
            decoded_full = encoder.decode(tokens)
        except Exception:  # pragma: no cover - fallback to character chunking
            decoded_full = ""

        if not decoded_full or (len(tokens) > chunk or len(decoded_full) <= chunk):
            try:
                pieces: List[str] = []
                index = 0
                total = len(tokens)
                while index < total:
                    window_tokens = tokens[index : index + chunk]
                    if not window_tokens:
                        break
                    pieces.append(encoder.decode(window_tokens))
                    index += step
                return pieces
            except Exception:  # pragma: no cover - fallback to character chunking
                pass

    source_text = decoded_full or text
    pieces: List[str] = []
    index = 0
    total = len(source_text)
    while index < total:
        window_text = source_text[index : index + chunk]
        if not window_text:
            break
        pieces.append(window_text)
        index += step
    return pieces


def _iter_pdf_text(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    reader = PdfReader(handle)
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - pypdf quirks
            text = ""
        cleaned = _clean(text)
        if cleaned:
            yield page_number, cleaned


def _iter_docx_text(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    document = Document(handle)
    buffer = io.StringIO()
    page_number = 1
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if buffer.tell():
            buffer.write("\n")
        buffer.write(text)
        if buffer.tell() > 8000:
            cleaned = _clean(buffer.getvalue())
            if cleaned:
                yield page_number, cleaned
                page_number += 1
            buffer = io.StringIO()
    remaining = _clean(buffer.getvalue())
    if remaining:
        yield page_number, remaining


def _iter_txt_text(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    wrapper = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
    try:
        buffer = io.StringIO()
        page_number = 1
        for line in wrapper:
            stripped = line.strip()
            if not stripped:
                continue
            if buffer.tell():
                buffer.write(" ")
            buffer.write(stripped)
            if buffer.tell() > 8000:
                text = _clean(buffer.getvalue())
                if text:
                    yield page_number, text
                    page_number += 1
                buffer = io.StringIO()
        remaining = _clean(buffer.getvalue())
        if remaining:
            yield page_number, remaining
    finally:
        try:  # pragma: no cover - best effort detach
            wrapper.detach()
        except Exception:
            pass


def _iter_markdown_text(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    wrapper = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
    try:
        buffer = io.StringIO()
        page_number = 1
        for line in wrapper:
            stripped = line.rstrip()
            if stripped.startswith("#") and buffer.tell():
                text = _clean(buffer.getvalue())
                if text:
                    yield page_number, text
                    page_number += 1
                buffer = io.StringIO()
            if stripped:
                if buffer.tell():
                    buffer.write(" ")
                buffer.write(stripped)
        final = _clean(buffer.getvalue())
        if final:
            yield page_number, final
    finally:
        try:
            wrapper.detach()
        except Exception:
            pass


def _iter_pptx_text(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    with zipfile.ZipFile(handle) as archive:
        slide_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for index, name in enumerate(slide_names, start=1):
            with archive.open(name) as slide:
                text_parts: List[str] = []
                for event, element in ET.iterparse(slide, events=("end",)):
                    if event == "end" and element.tag.endswith("}t"):
                        if element.text:
                            text_parts.append(element.text.strip())
                        element.clear()
                cleaned = _clean(" ".join(text_parts))
                if cleaned:
                    yield index, cleaned


def _iter_xlsx_text(handle: BinaryIO) -> Iterator[tuple[int, str]]:
    if load_workbook is None:  # pragma: no cover - dependency missing
        raise RuntimeError("openpyxl is required to parse XLSX files")

    workbook = load_workbook(handle, read_only=True, data_only=True)
    for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
        buffer = io.StringIO()
        for row in sheet.iter_rows(values_only=True):
            values = [str(cell).strip() for cell in row if cell not in (None, "")]
            if not values:
                continue
            if buffer.tell():
                buffer.write("\n")
            buffer.write(" ".join(values))
        text = _clean(buffer.getvalue())
        if text:
            yield sheet_index, text


def _hash_chunk(file: str, page: int, text: str) -> str:
    payload = f"{file}\u0000{page}\u0000{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_document_pages(
    filename: str, data: Union[bytes, bytearray, BinaryIO]
) -> Iterator[tuple[int, str]]:
    name = (filename or "").strip()
    if not name or "." not in name:
        return iter(())

    ext = name.rsplit(".", 1)[-1].lower()
    stream = _ensure_binary_stream(data)

    if ext == "pdf":
        return _iter_pdf_text(stream)
    if ext == "docx":
        return _iter_docx_text(stream)
    if ext == "txt":
        return _iter_txt_text(stream)
    if ext in {"md", "markdown"}:
        return _iter_markdown_text(stream)
    if ext == "pptx":
        return _iter_pptx_text(stream)
    if ext == "xlsx":
        return _iter_xlsx_text(stream)
    return iter(())


def parse_and_chunk(filename: str, data: Union[bytes, bytearray, BinaryIO]) -> List[dict[str, object]]:
    """Parse ``data`` according to ``filename`` extension and chunk the text."""

    name = (filename or "").strip()
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    start = time.perf_counter()
    status = "success"
    chunks: List[dict[str, object]] = []

    try:
        pages = list(iter_document_pages(name, data))
        if pages:
            chunk_size = _normalise_window_size(int(os.getenv("RAG_CHUNK", "900")))
            overlap = _normalise_overlap(chunk_size, int(os.getenv("RAG_OVERLAP", "140")))
            tokenizer = _get_tokenizer()

            for page_number, page_text in pages:
                for piece in _chunk(
                    page_text, chunk=chunk_size, overlap=overlap, encoder=tokenizer
                ):
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
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        record_document_parse(extension, status, len(chunks), duration)


__all__ = [
    "_chunk",
    "_clean",
    "_get_tokenizer",
    "iter_document_pages",
    "parse_and_chunk",
]
