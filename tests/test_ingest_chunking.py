from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.ingest import _chunk


def test_chunk_with_large_overlap_moves_forward():
    text = "".join(chr(97 + i) for i in range(26))

    chunks = _chunk(text, chunk=10, overlap=12)

    assert len(chunks) == len(text) - 10 + 1
    assert chunks[0] == text[:10]
    assert chunks[-1] == text[-10:]


def test_chunk_overlap_adjusts_when_chunk_is_small():
    text = "abcdefghij"

    chunks = _chunk(text, chunk=1, overlap=5)

    assert len(chunks) == len(text)
    assert all(len(chunk) == 1 for chunk in chunks[:-1])
    assert chunks[-1] == text[-1]
