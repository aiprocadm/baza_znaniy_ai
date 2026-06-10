"""Validate the committed public golden set (structure, not retrieval quality)."""

from __future__ import annotations

from pathlib import Path

from app.eval.dataset import load_golden, read_signature

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "data" / "eval" / "golden_public.jsonl"
CORPUS = REPO / "data" / "eval" / "corpus_public"


def test_public_golden_loads_and_uses_composite_keys() -> None:
    items = load_golden(GOLDEN)
    assert len(items) >= 30
    for item in items:
        for key in item.relevant_chunks:
            assert ":" in key, f"non-composite key {key!r} in {item.question!r}"


def test_public_golden_has_refusals_and_curated() -> None:
    items = load_golden(GOLDEN)
    refusals = [it for it in items if it.expect_refusal]
    assert len(refusals) >= 3
    assert all(it.relevant_chunks == () for it in refusals)
    assert sum(1 for it in items if it.source == "curated") >= 15


def test_public_golden_keys_point_at_committed_corpus_files() -> None:
    corpus_files = {p.name for p in CORPUS.glob("*.md")}
    assert len(corpus_files) >= 9
    items = load_golden(GOLDEN)
    for item in items:
        for key in item.relevant_chunks:
            fname = key.rsplit(":", 1)[0]
            assert fname in corpus_files, f"{key!r} references a file outside corpus_public/"


def test_public_golden_signature_pins_st_1024() -> None:
    sig = read_signature(GOLDEN)
    assert sig is not None
    data = sig.to_dict()
    assert data["dim"] == 1024
    assert "st" in str(data["embedder_name"])
