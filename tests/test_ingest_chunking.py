from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.ingest import _chunk, _clean, _get_tokenizer


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
