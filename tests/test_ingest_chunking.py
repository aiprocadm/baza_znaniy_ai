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
