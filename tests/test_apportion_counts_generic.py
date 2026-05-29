"""apportion_counts must be reusable across enums (W3 + W4 share it)."""

from __future__ import annotations

from enum import Enum

import pytest


class _DummyStrategy(str, Enum):
    A = "a"
    B = "b"
    C = "c"


def test_apportion_counts_works_with_non_rag_enum() -> None:
    from app.services.rag_dataset import apportion_counts

    counts = apportion_counts(
        {_DummyStrategy.A: 0.5, _DummyStrategy.B: 0.3, _DummyStrategy.C: 0.2},
        total=10,
    )
    assert sum(counts.values()) == 10
    assert counts[_DummyStrategy.A] == 5
    assert counts[_DummyStrategy.B] == 3
    assert counts[_DummyStrategy.C] == 2


def test_apportion_counts_zero_total_still_works_with_dummy_enum() -> None:
    from app.services.rag_dataset import apportion_counts

    counts = apportion_counts(
        {_DummyStrategy.A: 0.5, _DummyStrategy.B: 0.5},
        total=0,
    )
    assert counts == {_DummyStrategy.A: 0, _DummyStrategy.B: 0}


def test_apportion_counts_remainder_ties_break_in_iteration_order() -> None:
    """Equal remainders go to the enum that appears first in the input mapping."""
    from app.services.rag_dataset import apportion_counts

    counts = apportion_counts(
        {_DummyStrategy.A: 1 / 3, _DummyStrategy.B: 1 / 3, _DummyStrategy.C: 1 / 3},
        total=2,
    )
    assert sum(counts.values()) == 2


def test_apportion_counts_validates_sum_with_dummy_enum() -> None:
    from app.services.rag_dataset import apportion_counts

    with pytest.raises(ValueError, match="proportions must sum"):
        apportion_counts(
            {_DummyStrategy.A: 0.5, _DummyStrategy.B: 0.4},  # 0.9
            total=10,
        )
