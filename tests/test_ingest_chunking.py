"""Tests for the document chunking utilities."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import List

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE_ROOT = ROOT / "srv" / "projects" / "kb" / "app"


def _load_ingest():
    package_name = "kb_service_ingest"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    ingest_spec = importlib.util.spec_from_file_location(
        f"{package_name}.ingest", SERVICE_ROOT / "ingest.py"
    )
    assert ingest_spec and ingest_spec.loader
    ingest_module = importlib.util.module_from_spec(ingest_spec)
    sys.modules[ingest_spec.name] = ingest_module
    ingest_spec.loader.exec_module(ingest_module)
    return ingest_module


ingest = _load_ingest()
_chunk = ingest._chunk
_clean = ingest._clean
_get_tokenizer = ingest._get_tokenizer
parse_and_chunk = ingest.parse_and_chunk

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


        codex/update-tokenization-logic-in-ingest.py
def test_chunk_handles_zero_and_single_window_sizes() -> None:
    text = "hello"

    assert _chunk(text, chunk=0, overlap=2) == list(text)
    assert _chunk(text, chunk=1, overlap=2) == list(text)



        main
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
