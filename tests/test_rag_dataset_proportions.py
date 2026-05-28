"""Hamilton apportionment for the RAG variant distribution."""

from __future__ import annotations

import pytest


def test_proportion_spec_default_70_15_10_5() -> None:
    """Defaults match the spec — 70 / 15 / 10 / 5."""
    from app.services.rag_dataset import RAGVariant, default_proportions

    p = default_proportions()
    assert p[RAGVariant.RELEVANT] == 0.70
    assert p[RAGVariant.IRRELEVANT] == 0.15
    assert p[RAGVariant.PARTIAL] == 0.10
    assert p[RAGVariant.EMPTY] == 0.05


def test_apportion_sum_matches_total() -> None:
    """Apportionment never loses or invents samples."""
    from app.services.rag_dataset import apportion_counts, default_proportions

    counts = apportion_counts(default_proportions(), total=100)
    assert sum(counts.values()) == 100


@pytest.mark.parametrize("total", [1, 7, 23, 100, 257])
def test_apportion_total_invariant(total: int) -> None:
    from app.services.rag_dataset import apportion_counts, default_proportions

    counts = apportion_counts(default_proportions(), total=total)
    assert sum(counts.values()) == total


def test_apportion_zero_total_yields_zeros() -> None:
    from app.services.rag_dataset import RAGVariant, apportion_counts, default_proportions

    counts = apportion_counts(default_proportions(), total=0)
    assert all(counts[v] == 0 for v in RAGVariant)


def test_custom_proportions_validated() -> None:
    """Proportions must sum to 1.0 within float tolerance."""
    from app.services.rag_dataset import RAGVariant, apportion_counts

    with pytest.raises(ValueError, match="proportions must sum"):
        apportion_counts(
            {RAGVariant.RELEVANT: 0.5, RAGVariant.IRRELEVANT: 0.4},  # 0.9
            total=10,
        )
