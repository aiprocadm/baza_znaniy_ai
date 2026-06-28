"""Characterization tests for the pure seams of ``_chunk``.

Lives in its own module (no openpyxl/pptx ``importorskip``) so the windowing
algorithm and parameter math are verifiable without the document-parser deps
that gate ``test_ingest_chunking.py``. These pin the helpers lifted out of the
~94-line ``_chunk`` so the split is provably behaviour-preserving.
"""

from __future__ import annotations

import app.ingest.chunking as ingest

_window_and_step = ingest._window_and_step
_collect_windows = ingest._collect_windows
_chunk = ingest._chunk


def test_window_and_step_normal() -> None:
    assert _window_and_step(900, 140) == (900, 760)


def test_window_and_step_clamps_overlap_to_window() -> None:
    # overlap >= window collapses to a step of 1 so progress is always made.
    assert _window_and_step(4, 10) == (4, 1)


def test_window_and_step_coerces_bad_chunk_to_one() -> None:
    assert _window_and_step("oops", 2) == (1, 1)
    assert _window_and_step(0, 2) == (1, 1)


def test_collect_windows_emits_overlapping_pieces_with_tail() -> None:
    source = "abcdefghij"  # length 10
    pieces = _collect_windows(10, 4, 2, lambda start, end: source[start:end])
    assert pieces[0] == "abcd"
    assert pieces[-1].endswith("j")
    assert all(len(p) <= 4 for p in pieces)


def test_collect_windows_empty_for_nonpositive_length() -> None:
    assert _collect_windows(0, 4, 2, lambda start, end: "x") == []


# A handful of end-to-end _chunk characterizations mirrored here so the refactor
# has a locally-runnable regression net (test_ingest_chunking.py is skipped
# without openpyxl/pptx).
def test_chunk_zero_and_single_window_return_characters() -> None:
    text = "hello world"
    assert _chunk(text, chunk=0, overlap=2) == list(text)
    assert _chunk(text, chunk=1, overlap=2) == list(text)


def test_chunk_empty_and_short_text() -> None:
    assert _chunk("", chunk=5, overlap=2) == []
    assert _chunk("short", chunk=20, overlap=5) == ["short"]


def test_chunk_high_overlap_still_progresses() -> None:
    chunks = _chunk("abcdef", chunk=2, overlap=5)
    assert chunks
    assert "".join(dict.fromkeys("".join(chunks))) == "abcdef"
