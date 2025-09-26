from __future__ import annotations

import importlib
import pathlib
import sys
from typing import List

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ingest_module = importlib.import_module("app.rag.ingest")
_chunk = ingest_module._chunk
_clean = ingest_module._clean
_get_tokenizer = ingest_module._get_tokenizer
parse_and_chunk = ingest_module.parse_and_chunk


def _make_text_with_tokens(count: int, token_id: int = 100):
    encoder = _get_tokenizer()
    token_ids = [token_id] * count
    text = encoder.decode(token_ids)
    return text, encoder, token_ids


def _token_windows(text: str, chunk: int, overlap: int) -> List[List[int]]:
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    windows: List[List[int]] = []
    if not tokens:
        return windows

    chunk = max(chunk, 1)
    overlap = max(min(overlap, chunk - 1), 0) if chunk > 1 else 0
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + chunk, total)
        windows.append(tokens[start:end])
        if end >= total:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return windows


@pytest.mark.parametrize(
    "text, chunk, overlap, expected",
    [
        ("abcd", 1, 1, ["a", "b", "c", "d"]),
        ("hello", 0, 2, ["h", "e", "l", "l", "o"]),
    ],
)
def test_chunk_small_sizes(text: str, chunk: int, overlap: int, expected: list[str]):
    assert _chunk(text, chunk=chunk, overlap=overlap) == expected


def test_chunk_progress_with_high_overlap():
    text = "abcdef"
    chunks = _chunk(text, chunk=2, overlap=5)
    assert "".join(chunks).startswith(text[:2])
    assert chunks[-1]
    assert sum(len(c) for c in chunks) >= len(text)


def test_chunk_respects_token_window_size():
    text, encoder, original_tokens = _make_text_with_tokens(1800)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]

    assert len(chunks) == 3
    assert encoded_chunks[0] == original_tokens[:900]
    assert encoded_chunks[1] == original_tokens[760:1660]
    assert encoded_chunks[2] == original_tokens[1520:1800]
    assert all(len(tokens) <= 900 for tokens in encoded_chunks)


def test_chunk_overlap_consistency():
    text, encoder, _ = _make_text_with_tokens(1500, token_id=101)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]

    for current, nxt in zip(encoded_chunks, encoded_chunks[1:]):
        overlap = min(140, len(current), len(nxt))
        assert current[-overlap:] == nxt[:overlap]


def test_chunk_respects_token_boundaries():
    text = "hello world " * 10
    chunk = 15
    overlap = 4

    tokenizer = _get_tokenizer()
    expected = _token_windows(text, chunk, overlap)
    chunks = _chunk(text, chunk=chunk, overlap=overlap)
    encoded_chunks = [tokenizer.encode(piece) for piece in chunks]

    assert encoded_chunks == expected
    assert all(len(window) <= chunk for window in encoded_chunks)


def test_chunk_overlap_adjusts_when_chunk_is_small():
    text = "a" * 12
    tokenizer = _get_tokenizer()

    chunks = _chunk(text, chunk=1, overlap=5)
    encoded_chunks = [tokenizer.encode(piece) for piece in chunks]

    assert len(encoded_chunks) == len(tokenizer.encode(text))
    assert all(len(tokens) == 1 for tokens in encoded_chunks)


def test_parse_and_chunk_preserves_metadata_and_tokens():
    text = "page content " * 20
    payload = text.encode("utf-8")

    chunks = parse_and_chunk("example.txt", payload)

    tokenizer = _get_tokenizer()
    reconstructed = [tokenizer.encode(chunk["text"]) for chunk in chunks]
    expected = _token_windows(_clean(text), 900, 140)

    assert reconstructed == expected
    assert all(chunk["file"] == "example.txt" for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)
