"""Юнит-тесты чистых хелперов свип-скрипта (без модели/стора)."""

from __future__ import annotations

from scripts.sweep_rerank_candidates import _fixture_items, _parse_ks


def test_parse_ks_splits_and_ints() -> None:
    assert _parse_ks("1,2,3,5,8,10,12,16,20") == [1, 2, 3, 5, 8, 10, 12, 16, 20]
    assert _parse_ks("5") == [5]
    assert _parse_ks("1, ,2 ,") == [1, 2]  # пустые токены игнорируются


def test_fixture_items_drops_question_and_texts() -> None:
    items = [
        {
            "question": "q",
            "relevant": ["g1"],
            "shortlist_keys": ["a", "g1"],
            "teacher_scores": [0.1, 0.9],
            "_texts": ["ta", "tg1"],
        }
    ]
    out = _fixture_items(items)
    assert out == [
        {"relevant": ["g1"], "shortlist_keys": ["a", "g1"], "teacher_scores": [0.1, 0.9]}
    ]
    assert "question" not in out[0] and "_texts" not in out[0]
