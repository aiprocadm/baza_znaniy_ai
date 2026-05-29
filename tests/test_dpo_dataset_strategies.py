"""Tests for the DPOPairBuilder orchestrator + apportionment."""

from __future__ import annotations

from collections import Counter

import pytest

from app.services.synthetic_qa import QAPair


def _make_seeds(n: int) -> list[QAPair]:
    return [
        QAPair(
            instruction=f"Вопрос {i}?",
            input="",
            output=f"Ответ {i}. [doc_chunk:{i}]",
            source_chunk_id=i,
        )
        for i in range(1, n + 1)
    ]


def test_default_synthetic_proportions_are_40_30_30() -> None:
    from app.services.dpo_dataset import RejectStrategy, default_synthetic_proportions

    p = default_synthetic_proportions()
    assert pytest.approx(p[RejectStrategy.NO_CITATION], 1e-6) == 0.40
    assert pytest.approx(p[RejectStrategy.GENERIC], 1e-6) == 0.30
    assert pytest.approx(p[RejectStrategy.HALLUCINATION], 1e-6) == 0.30


def test_builder_respects_apportionment() -> None:
    from app.services.dpo_dataset import DPOPairBuilder

    builder = DPOPairBuilder(teacher=lambda _p: "Fake teacher answer.")
    pairs = list(builder.build(_make_seeds(10), total=10))
    assert len(pairs) == 10
    counts = Counter(p.strategy.value for p in pairs)
    assert counts["no_citation"] == 4
    assert counts["generic"] == 3
    assert counts["hallucination"] == 3


def test_builder_skips_when_no_citation_marker() -> None:
    """NO_CITATION quota is re-allocated when the seed has no marker."""
    from app.services.dpo_dataset import DPOPairBuilder

    seeds = [
        QAPair(instruction="Q1?", input="", output="No marker here.", source_chunk_id=1),
        QAPair(instruction="Q2?", input="", output="Still no marker.", source_chunk_id=2),
    ]
    builder = DPOPairBuilder(teacher=lambda _p: "Fake teacher answer.")
    pairs = list(builder.build(seeds, total=2))
    for p in pairs:
        assert p.strategy.value != "no_citation"


def test_builder_under_delivers_when_seeds_exhausted() -> None:
    from app.services.dpo_dataset import DPOPairBuilder

    builder = DPOPairBuilder(teacher=lambda _p: "Generic.")
    pairs = list(builder.build(_make_seeds(3), total=10))
    assert len(pairs) <= 3


def test_builder_deterministic_across_runs() -> None:
    """Same seeds + same teacher produce identical strategy assignment."""
    from app.services.dpo_dataset import DPOPairBuilder

    seeds = _make_seeds(20)
    a = list(DPOPairBuilder(teacher=lambda _p: "x").build(seeds, total=20))
    b = list(DPOPairBuilder(teacher=lambda _p: "x").build(seeds, total=20))
    assert [p.strategy.value for p in a] == [p.strategy.value for p in b]
