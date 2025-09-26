        codex/introduce-tokenizer-and-rewrite-chunking-logic-ca0dv1
import importlib

        codex/introduce-tokenizer-and-rewrite-chunking-logic
import importlib

      codex/fix-overlapping-chunk-processing-in-ingest.py
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.ingest import _chunk


@pytest.mark.parametrize(
    "text, chunk, overlap, expected",
    [
        ("abcd", 1, 1, ["a", "b", "c", "d"]),
        ("hello", 0, 2, ["h", "e", "l", "l", "o"]),
    ],
)
def test_chunk_small_sizes(text, chunk, overlap, expected):
    assert _chunk(text, chunk=chunk, overlap=overlap) == expected


def test_chunk_progress_with_high_overlap():
    text = "abcdef"
    chunks = _chunk(text, chunk=2, overlap=5)
    assert "".join(chunks).startswith(text[:2])
    assert chunks[-1]
    assert sum(len(c) for c in chunks) >= len(text)

from pathlib import Path
        main
        main
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

        codex/introduce-tokenizer-and-rewrite-chunking-logic-ca0dv1

        codex/introduce-tokenizer-and-rewrite-chunking-logic
        main
ingest_module = importlib.import_module("app.rag.ingest")
_chunk = ingest_module._chunk
_get_tokenizer = ingest_module._get_tokenizer

for name in ["app.rag.ingest", "app.rag", "app"]:
    sys.modules.pop(name, None)

        codex/introduce-tokenizer-and-rewrite-chunking-logic-ca0dv1

def _make_text_with_tokens(count: int, token_id: int = 100):
    encoder = _get_tokenizer()
    token_ids = [token_id] * count
    text = encoder.decode(token_ids)
    return text, encoder, token_ids

        main

def _make_text_with_tokens(count: int, token_id: int = 100):
    encoder = _get_tokenizer()
    token_ids = [token_id] * count
    text = encoder.decode(token_ids)
    return text, encoder, token_ids

        codex/introduce-tokenizer-and-rewrite-chunking-logic-ca0dv1
def test_chunk_respects_token_window_size():
    text, encoder, original_tokens = _make_text_with_tokens(1800)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]


from app.rag.ingest import _chunk, _clean, _get_tokenizer

        main

def _token_windows(text: str, chunk: int, overlap: int):
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    windows = []
    i = 0
    n = len(tokens)
    if n == 0:
        return windows
    overlap = max(min(overlap, chunk - 1), 0)
    while i < n:
        j = min(i + chunk, n)
        windows.append(tokens[i:j])
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return windows

        codex/introduce-tokenizer-and-rewrite-chunking-logic
def test_chunk_respects_token_window_size():
    text, encoder, original_tokens = _make_text_with_tokens(1800)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]

        main
    assert len(chunks) == 3
    assert encoded_chunks[0] == original_tokens[:900]
    assert encoded_chunks[1] == original_tokens[760:1660]
    assert encoded_chunks[2] == original_tokens[1520:1800]
    assert all(len(tokens) <= 900 for tokens in encoded_chunks)
        codex/introduce-tokenizer-and-rewrite-chunking-logic-ca0dv1


def test_chunk_overlap_consistency():
    text, encoder, _ = _make_text_with_tokens(1500, token_id=101)

    chunks = _chunk(text, chunk=900, overlap=140, encoder=encoder)
    encoded_chunks = [encoder.encode(chunk) for chunk in chunks]

    for current, nxt in zip(encoded_chunks, encoded_chunks[1:]):
        overlap = min(140, len(current), len(nxt))
        assert current[-overlap:] == nxt[:overlap]



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
    from app.rag.ingest import parse_and_chunk

    text = "page content " * 20
    payload = text.encode("utf-8")

    chunks = parse_and_chunk("example.txt", payload)

    tokenizer = _get_tokenizer()
    reconstructed = [tokenizer.encode(chunk["text"]) for chunk in chunks]
    expected = _token_windows(_clean(text), 900, 140)

    assert reconstructed == expected
    assert all(chunk["file"] == "example.txt" for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)
        main
        main
        main
