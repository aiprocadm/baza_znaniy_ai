from __future__ import annotations

import importlib
import pathlib
import sys
from typing import Iterable, List

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_ingest():
    module = importlib.import_module("app.rag.ingest")
    return importlib.reload(module)


def _token_windows(text: str, chunk: int, overlap: int) -> List[List[int]]:
    ingest = _reload_ingest()
    tokenizer = ingest._get_tokenizer()
    tokens = tokenizer.encode(text)
    if not tokens:
        return []
    chunk = max(int(chunk), 1)
    overlap = max(0, min(int(overlap), chunk - 1))

    windows: List[List[int]] = []
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + chunk, total)
        windows.append(tokens[start:end])
        if end >= total:
            break
        start = max(end - overlap, 0)
    return windows


@pytest.mark.parametrize(
    "text, chunk, overlap, expected",
    [
        ("abcd", 1, 1, ["a", "b", "c", "d"]),
        ("hello", 0, 2, ["h", "e", "l", "l", "o"]),
    ],
)
def test_chunk_small_sizes(text: str, chunk: int, overlap: int, expected: List[str]) -> None:
    ingest = _reload_ingest()
    assert ingest._chunk(text, chunk=chunk, overlap=overlap) == expected


def _make_text_with_tokens(count: int, token_id: int = 100) -> tuple[str, Iterable[int]]:
    ingest = _reload_ingest()
    tokenizer = ingest._get_tokenizer()
    token_ids = [token_id] * count
    return tokenizer.decode(token_ids), token_ids


def test_chunk_progress_with_high_overlap() -> None:
    ingest = _reload_ingest()
    text = "abcdef"
    chunks = ingest._chunk(text, chunk=2, overlap=5)
    assert "".join(chunks).startswith(text[:2])
    assert chunks[-1]
    assert sum(len(c) for c in chunks) >= len(text)


def test_chunk_respects_token_window_size() -> None:
    ingest = _reload_ingest()
    text, original_tokens = _make_text_with_tokens(1800)

    chunks = ingest._chunk(text, chunk=900, overlap=140)
    tokenizer = ingest._get_tokenizer()
    encoded_chunks = [tokenizer.encode(chunk) for chunk in chunks]

    assert len(chunks) == 3
    assert encoded_chunks[0] == original_tokens[:900]
    assert encoded_chunks[1] == original_tokens[760:1660]
    assert encoded_chunks[2] == original_tokens[1520:1800]
    assert all(len(tokens) <= 900 for tokens in encoded_chunks)


def test_chunk_overlap_consistency() -> None:
    ingest = _reload_ingest()
    text, _ = _make_text_with_tokens(1500, token_id=101)

    chunks = ingest._chunk(text, chunk=900, overlap=140)
    tokenizer = ingest._get_tokenizer()
    encoded_chunks = [tokenizer.encode(chunk) for chunk in chunks]

    for current, nxt in zip(encoded_chunks, encoded_chunks[1:]):
        overlap = min(140, len(current), len(nxt))
        assert current[-overlap:] == nxt[:overlap]


def test_chunk_respects_token_boundaries() -> None:
    ingest = _reload_ingest()
    text = "hello world " * 10
    chunk = 15
    overlap = 4

    tokenizer = ingest._get_tokenizer()
    expected = _token_windows(text, chunk, overlap)
    chunks = ingest._chunk(text, chunk=chunk, overlap=overlap)
    encoded_chunks = [tokenizer.encode(piece) for piece in chunks]

    assert encoded_chunks == expected
    assert all(len(window) <= chunk for window in encoded_chunks)


def test_chunk_overlap_adjusts_when_chunk_is_small() -> None:
    ingest = _reload_ingest()
    text = "a" * 12
    tokenizer = ingest._get_tokenizer()

    chunks = ingest._chunk(text, chunk=1, overlap=5)
    encoded_chunks = [tokenizer.encode(piece) for piece in chunks]

    assert len(encoded_chunks) == len(tokenizer.encode(text))
    assert all(len(tokens) == 1 for tokens in encoded_chunks)


def test_parse_and_chunk_preserves_metadata_and_tokens() -> None:
    ingest = _reload_ingest()
    text = "page content " * 20
    payload = text.encode("utf-8")

    chunks = ingest.parse_and_chunk("example.txt", payload)

    tokenizer = ingest._get_tokenizer()
    reconstructed = [tokenizer.encode(chunk["text"]) for chunk in chunks]
    expected = _token_windows(ingest._clean(text), 900, 140)

    assert reconstructed == expected
    assert all(chunk["file"] == "example.txt" for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)
