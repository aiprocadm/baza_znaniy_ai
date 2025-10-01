"""Tests for the document chunking utilities."""

from __future__ import annotations

import hashlib
import io
import sys
import types
import zipfile
from typing import List

import pytest
from openpyxl import Workbook

if "app.ingest.service" not in sys.modules:
    service_stub = types.ModuleType("app.ingest.service")

    class IngestJob:
        def __init__(self, file_record: object | None = None, *, attempt: int = 0) -> None:
            self.file_record = file_record
            self.attempt = attempt
            self.job_record_id: int | None = None

    class IngestWorker:
        def __init__(self, service: "IngestService") -> None:
            self.service = service
            self._task = None

        def ensure_started(self) -> None:  # pragma: no cover - no-op for tests
            return None

    class IngestService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.worker: IngestWorker | None = None

        def set_worker(self, worker: IngestWorker) -> None:
            self.worker = worker

        def ensure_background_worker(self) -> None:  # pragma: no cover - no-op
            return None

        async def stop_background_worker(self) -> None:  # pragma: no cover - no-op
            return None

    service_stub.IngestJob = IngestJob
    service_stub.IngestService = IngestService
    service_stub.IngestWorker = IngestWorker
    sys.modules["app.ingest.service"] = service_stub

import app.ingest.chunking as ingest

_chunk = ingest._chunk
_clean = ingest._clean
_get_tokenizer = ingest._get_tokenizer
parse_and_chunk = ingest.parse_and_chunk
iter_document_pages = ingest.iter_document_pages

TOKENIZER = _get_tokenizer()


def _make_text_with_tokens(count: int, token_id: int = 100) -> tuple[str, List[int]]:
    token_ids = [token_id] * count
    text = TOKENIZER.decode(token_ids)
    return text, token_ids


def _expected_windows(
    text: str, chunk: int, overlap: int, *, tokenizer=TOKENIZER
) -> List[List[int]]:
    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    window = max(int(chunk), 1)
    step_overlap = max(min(int(overlap), window - 1), 0)

    windows: List[List[int]] = []
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + window, total)
        windows.append(tokens[start:end])
        if end >= total:
            break
        next_start = max(end - step_overlap, start + 1)
        start = next_start
    return windows


def test_chunk_handles_zero_and_single_window_sizes() -> None:
    text = "hello"

    assert _chunk(text, chunk=0, overlap=2) == list(text)
    assert _chunk(text, chunk=1, overlap=2) == list(text)



def test_chunk_progress_with_high_overlap() -> None:
    text = "abcdef"
    chunks = _chunk(text, chunk=2, overlap=5)
    assert "".join(chunks).startswith(text[:2])
    assert chunks[-1]
    assert sum(len(chunk) for chunk in chunks) >= len(text)


def test_chunk_respects_token_window_size() -> None:
    text, original_tokens = _make_text_with_tokens(1800)
    chunks = _chunk(text, chunk=900, overlap=140, encoder=TOKENIZER)
    encoded_chunks = [TOKENIZER.encode(chunk) for chunk in chunks]

    assert len(chunks) == 3
    assert encoded_chunks[0] == original_tokens[:900]
    assert encoded_chunks[1] == original_tokens[760:1660]
    assert encoded_chunks[2] == original_tokens[1520:]
    assert all(len(tokens) <= 900 for tokens in encoded_chunks)


def test_chunk_fallback_respects_token_window_slices() -> None:
    class ExpandingTokenizer:
        def encode(self, text: str) -> List[int]:
            if text == "<expand>":
                return [1]
            return [2] * len(text)

        def decode(self, tokens: List[int]) -> str:
            if tokens == [1]:
                return "x" * 1800
            return "x" * len(tokens)

    tokenizer = ExpandingTokenizer()
    chunk = 900
    overlap = 140
    text = "<expand>"

    expanded_text = tokenizer.decode(tokenizer.encode(text))
    chunks = _chunk(text, chunk=chunk, overlap=overlap, encoder=tokenizer)

    assert len(chunks) == 3
    assert chunks[0] == expanded_text[:chunk]
    assert chunks[1] == expanded_text[chunk - overlap : chunk - overlap + chunk]
    assert chunks[2] == expanded_text[(2 * chunk) - (2 * overlap) :]

    encoded_chunks = [tokenizer.encode(chunk) for chunk in chunks]
    expected = _expected_windows(expanded_text, chunk, overlap, tokenizer=tokenizer)

    assert encoded_chunks == expected
    for current, nxt in zip(encoded_chunks, encoded_chunks[1:]):
        assert current[-overlap:] == nxt[:overlap]


def test_chunk_overlap_consistency() -> None:
    text, _ = _make_text_with_tokens(1500, token_id=101)
    chunks = _chunk(text, chunk=900, overlap=140, encoder=TOKENIZER)
    encoded_chunks = [TOKENIZER.encode(chunk) for chunk in chunks]

    for current, nxt in zip(encoded_chunks, encoded_chunks[1:]):
        overlap = min(140, len(current), len(nxt))
        assert current[-overlap:] == nxt[:overlap]


def test_chunk_respects_token_boundaries() -> None:
    text = "hello world " * 10
    chunk = 15
    overlap = 4

    expected = _expected_windows(text, chunk, overlap)
    chunks = _chunk(text, chunk=chunk, overlap=overlap, encoder=TOKENIZER)
    encoded_chunks = [TOKENIZER.encode(piece) for piece in chunks]

    assert encoded_chunks == expected
    assert all(len(window) <= chunk for window in encoded_chunks)


def test_chunk_overlap_adjusts_when_chunk_is_small() -> None:
    text = "a" * 12
    chunks = _chunk(text, chunk=1, overlap=5)
    encoded = [TOKENIZER.encode(piece) for piece in chunks]

    assert len(encoded) == len(TOKENIZER.encode(text))
    assert all(len(tokens) == 1 for tokens in encoded)


def test_chunk_handles_single_token_multi_char_text() -> None:
    class SingleTokenTokenizer:
        def encode(self, text: str) -> List[int]:
            return [1] if text else []

        def decode(self, tokens: List[int]) -> str:
            return "window" if tokens else ""

    tokenizer = SingleTokenTokenizer()
    text = "window"

    chunks = _chunk(text, chunk=1, overlap=0, encoder=tokenizer)

    assert chunks == list(text)


def test_chunk_small_window_char_tokenizer_fallback() -> None:
    class ExpandingTokenizer:
        def encode(self, text: str) -> List[int]:
            return [1] if text else []

        def decode(self, tokens: List[int]) -> str:
            return "expansion" if tokens else ""

    tokenizer = ExpandingTokenizer()
    chunk = 5
    overlap = 2
    text = "trigger"

    pieces = _chunk(text, chunk=chunk, overlap=overlap, encoder=tokenizer)

    expanded = tokenizer.decode(tokenizer.encode(text))
    char_tokenizer = ingest._CharTokenizer()
    expected = _expected_windows(expanded, chunk, overlap, tokenizer=char_tokenizer)
    encoded_pieces = [char_tokenizer.encode(piece) for piece in pieces]

    assert encoded_pieces == expected


def test_chunk_small_window_reencoded_branch() -> None:
    class ReencodingTokenizer:
        def encode(self, text: str) -> List[int]:
            if text == "seed":
                return [1, 2]
            if text == "ab":
                return [3, 4]
            if not text:
                return []
            return [9] * len(text)

        def decode(self, tokens: List[int]) -> str:
            if tokens in ([1, 2], [3, 4]):
                return "ab"
            if not tokens:
                return ""
            return "x" * len(tokens)

    tokenizer = ReencodingTokenizer()
    text = "seed"

    pieces = _chunk(text, chunk=5, overlap=0, encoder=tokenizer)

    assert pieces == ["ab"]
    assert tokenizer.encode(pieces[0]) == [3, 4]


def test_chunk_small_window_returns_original_tokens_when_fallback_empty() -> None:
    class FlakyStr(str):
        def __new__(cls, value: str):
            obj = super().__new__(cls, value)
            obj._first = True
            return obj

        def __bool__(self) -> bool:
            if getattr(self, "_first", False):
                object.__setattr__(self, "_first", False)
                return True
            return False

    class EmptyFallbackTokenizer:
        def encode(self, text: str) -> List[int]:
            return [7] if text == "" else [ord(ch) for ch in text]

        def decode(self, tokens: List[int]) -> str:
            return "" if tokens else ""

    text = FlakyStr("")
    tokenizer = EmptyFallbackTokenizer()

    pieces = _chunk(text, chunk=5, overlap=0, encoder=tokenizer)

    assert pieces == [""]
    assert tokenizer.encode(pieces[0]) == [7]


def test_chunk_with_tiny_window_uses_characters_and_handles_empty_tokens() -> None:
    assert _chunk("hello", chunk=0, overlap=2) == list("hello")

    class TruthyEmptyStr(str):
        def __new__(cls, value: str):
            obj = super().__new__(cls, value)
            return obj

        def __bool__(self) -> bool:
            return True

    empty_text = TruthyEmptyStr("")

    assert _chunk(empty_text, chunk=1, overlap=0) == []


def test_parse_and_chunk_preserves_metadata_and_tokens() -> None:
    text = "page content " * 20
    payload = text.encode("utf-8")

    chunks = parse_and_chunk("example.txt", payload)
    encoded = [TOKENIZER.encode(chunk["text"]) for chunk in chunks]
    expected = _expected_windows(_clean(text), 900, 140)

    assert encoded == expected
    assert all(chunk["file"] == "example.txt" for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)


def test_parse_and_chunk_requires_extension() -> None:
    payload = b"contents"

    assert parse_and_chunk("", payload) == []
    assert parse_and_chunk("no_extension", payload) == []


def test_parse_and_chunk_rejects_unsupported_extension() -> None:
    payload = b"contents"

    assert parse_and_chunk("example.csv", payload) == []


def _expected_sha(file: str, page: int, text: str) -> str:
    payload = f"{file}\u0000{page}\u0000{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_pptx_bytes(slides: List[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index, text in enumerate(slides, start=1):
            payload = (
                "<p:sld xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
                "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">"
                "<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>"
                + text
                + "</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
            )
            archive.writestr(f"ppt/slides/slide{index}.xml", payload)
    buffer.seek(0)
    return buffer.getvalue()



def _make_xlsx_bytes(sheets: List[List[List[str]]]) -> bytes:
    workbook = Workbook()
    first = True
    for index, rows in enumerate(sheets, start=1):
        if first:
            sheet = workbook.active
            sheet.title = f"Sheet{index}"
            first = False
        else:
            sheet = workbook.create_sheet(title=f"Sheet{index}")
        for row in rows:
            sheet.append(row)
    stream = io.BytesIO()
    workbook.save(stream)
    stream.seek(0)
    return stream.getvalue()


def test_parse_and_chunk_pdf_uses_mock_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["First page text", "Second page text"]

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdfReader:
        def __init__(self, _stream: object) -> None:
            self.pages = [FakePage(text) for text in texts]

    encode_calls: List[str] = []

    class RecordingTokenizer:
        def encode(self, text: str) -> List[int]:
            encode_calls.append(text)
            return [ord(ch) for ch in text]

        def decode(self, tokens: List[int]) -> str:
            return "".join(chr(token) for token in tokens)

    tokenizer = RecordingTokenizer()
    tokenizer_calls = {"count": 0}

    def fake_get_tokenizer() -> RecordingTokenizer:
        tokenizer_calls["count"] += 1
        return tokenizer

    monkeypatch.setattr(ingest, "PdfReader", FakePdfReader)
    monkeypatch.setattr(ingest, "_get_tokenizer", fake_get_tokenizer)
    monkeypatch.setenv("RAG_CHUNK", "50")
    monkeypatch.setenv("RAG_OVERLAP", "0")

    filename = "file.pdf"
    chunks = parse_and_chunk(filename, b"binary-pdf")

    cleaned_texts = [_clean(text) for text in texts]
    expected = [
        {
            "file": filename,
            "page": index + 1,
            "sha256": _expected_sha(filename, index + 1, text),
            "text": text,
        }
        for index, text in enumerate(cleaned_texts)
    ]

    assert chunks == expected
    assert tokenizer_calls["count"] == 1
    for text in cleaned_texts:
        assert text in encode_calls


def test_parse_and_chunk_docx_uses_mock_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    paragraphs = ["First   paragraph", "Second paragraph"]

    class FakeParagraph:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeDocument:
        def __init__(self, _stream: object) -> None:
            self.paragraphs = [FakeParagraph(text) for text in paragraphs]

    encode_calls: List[str] = []

    class RecordingTokenizer:
        def encode(self, text: str) -> List[int]:
            encode_calls.append(text)
            return [ord(ch) for ch in text]

        def decode(self, tokens: List[int]) -> str:
            return "".join(chr(token) for token in tokens)

    tokenizer = RecordingTokenizer()
    tokenizer_calls = {"count": 0}

    def fake_get_tokenizer() -> RecordingTokenizer:
        tokenizer_calls["count"] += 1
        return tokenizer

    monkeypatch.setattr(ingest, "Document", FakeDocument)
    monkeypatch.setattr(ingest, "_get_tokenizer", fake_get_tokenizer)
    monkeypatch.setenv("RAG_CHUNK", "50")
    monkeypatch.setenv("RAG_OVERLAP", "0")

    filename = "report.docx"
    chunks = parse_and_chunk(filename, b"binary-docx")

    cleaned_text = _clean("\n".join(paragraphs))
    expected = [
        {
            "file": filename,
            "page": 1,
            "sha256": _expected_sha(filename, 1, cleaned_text),
            "text": cleaned_text,
        }
    ]

    assert chunks == expected
    assert tokenizer_calls["count"] == 1
    assert cleaned_text in encode_calls


def test_iter_document_pages_splits_markdown_sections() -> None:
    payload = b"# Intro\nFirst paragraph\n# Next\nSecond part"
    pages = list(iter_document_pages("notes.md", payload))

    assert [number for number, _ in pages] == [1, 2]
    assert "Intro" in pages[0][1]
    assert "Second" in pages[1][1]


def test_iter_document_pages_reads_pptx_slides() -> None:
    slides = ["Slide One", "Slide Two"]
    pages = list(iter_document_pages("deck.pptx", _make_pptx_bytes(slides)))

    assert len(pages) == 2
    assert [text for _, text in pages] == slides


def test_iter_document_pages_reads_xlsx_sheets() -> None:
    sheets = [["Header", "Value"], ["Row", "1"]]
    more = [["Another", "Sheet"]]
    pages = list(iter_document_pages("book.xlsx", _make_xlsx_bytes([sheets, more])))

    assert len(pages) == 2
    assert "Header Value" in pages[0][1]
    assert "Another Sheet" in pages[1][1]


def test_parse_and_chunk_accepts_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK", "200")
    monkeypatch.setenv("RAG_OVERLAP", "0")
    chunks = parse_and_chunk("notes.md", b"# Title\nContent")

    assert chunks
    assert all(chunk["file"] == "notes.md" for chunk in chunks)


def test_parse_and_chunk_accepts_pptx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK", "200")
    monkeypatch.setenv("RAG_OVERLAP", "0")
    slides = ["Slide A", "Slide B"]
    payload = _make_pptx_bytes(slides)
    chunks = parse_and_chunk("slides.pptx", payload)

    assert len(chunks) == len(slides)
    assert [chunk["page"] for chunk in chunks] == [1, 2]


def test_parse_and_chunk_accepts_xlsx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK", "200")
    monkeypatch.setenv("RAG_OVERLAP", "0")
    sheets = [["Header", "Value"], ["Row", "1"]]
    payload = _make_xlsx_bytes([sheets])
    chunks = parse_and_chunk("workbook.xlsx", payload)

    assert chunks
    assert chunks[0]["page"] == 1
