from app.eval.dataset import GoldenItem
from scripts.build_pravo_golden import (
    build_golden_items,
    heading_to_query,
    select_heldout,
)


def test_heading_to_query_strips_statya_prefix():
    assert (
        heading_to_query("Статья 12. Способы защиты гражданских прав")
        == "Способы защиты гражданских прав"
    )
    assert heading_to_query("Статья 1. X") == "X"
    assert heading_to_query("  Просто тема  ") == "Просто тема"


def test_select_heldout_takes_every_stride():
    docs = list(range(10))
    assert select_heldout(docs, stride=3) == [0, 3, 6, 9]
    assert select_heldout(docs, stride=1) == docs


def test_build_golden_items_uses_all_chunks_as_relevant_and_skips_empty():
    heldout = [
        ("ГК_РФ_ч.1__a00012", "Статья 12. Способы защиты", [0, 1]),
        ("ГК_РФ_ч.1__a00099", "Статья 99.", []),
    ]
    items = build_golden_items(heldout)
    assert len(items) == 1
    it = items[0]
    assert isinstance(it, GoldenItem)
    assert it.question == "Способы защиты"
    assert it.relevant_chunks == ("ГК_РФ_ч.1__a00012:0", "ГК_РФ_ч.1__a00012:1")
    assert it.source == "auto"
