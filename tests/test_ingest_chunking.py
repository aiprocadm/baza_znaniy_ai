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


def _expected_windows(text: str, chunk: int, overlap: int) -> List[List[int]]:
    tokens = TOKENIZER.encode(text)
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


def test_chunk_returns_single_char_windows_when_chunk_is_one() -> None:
    assert _chunk("abcd", chunk=1, overlap=1) == ["a", "b", "c", "d"]


def test_chunk_returns_single_char_windows_when_chunk_is_zero() -> None:
    assert _chunk("hello", chunk=0, overlap=2) == list("hello")


def test_chunk_single_character_windows_return_characters() -> None:
    assert _chunk("abcd", chunk=1, overlap=1) == list("abcd")
    assert _chunk("hello", chunk=0, overlap=2) == list("hello")


def test_chunk_single_character_windows_return_characters() -> None:
    assert _chunk("abcd", chunk=1, overlap=1) == list("abcd")
    assert _chunk("hello", chunk=0, overlap=2) == list("hello")


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


def test_parse_and_chunk_preserves_metadata_and_tokens() -> None:
    text = "page content " * 20
    payload = text.encode("utf-8")

    chunks = parse_and_chunk("example.txt", payload)
    encoded = [TOKENIZER.encode(chunk["text"]) for chunk in chunks]
    expected = _expected_windows(_clean(text), 900, 140)

    assert encoded == expected
    assert all(chunk["file"] == "example.txt" for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)
