"""Юнит-тесты чистой логики свипа кандидатов (без модели, детерминированно)."""

from __future__ import annotations

from app.eval.candidate_sweep import base_topk, rerank_topk, sweep_quality


def test_base_topk_truncates_in_order() -> None:
    keys = ["a", "b", "c", "d"]
    assert base_topk(keys, 2) == ["a", "b"]
    assert base_topk(keys, 10) == ["a", "b", "c", "d"]  # k>len → весь список
    assert base_topk(keys, 0) == []


def test_rerank_topk_sorts_by_teacher_score_desc() -> None:
    # би-энкодер: a,b,c ; teacher поднимает c (0.9) над a (0.5), b (0.1)
    keys = ["a", "b", "c"]
    scores = [0.5, 0.1, 0.9]
    assert rerank_topk(keys, scores, 3) == ["c", "a", "b"]
    # реранк только top-2 → c не в шорт-листе, порядок среди {a,b} по скору
    assert rerank_topk(keys, scores, 2) == ["a", "b"]
    # k=1 → ровно один ключ (одиночку реранк не меняет)
    assert rerank_topk(keys, scores, 1) == ["a"]


def test_rerank_topk_tie_break_by_original_position() -> None:
    # равные скоры → меньший исходный индекс раньше (зеркалит rerank.py:141)
    keys = ["a", "b", "c"]
    scores = [0.5, 0.5, 0.5]
    assert rerank_topk(keys, scores, 3) == ["a", "b", "c"]


def test_sweep_quality_shows_teacher_lift_at_full_k() -> None:
    # 2 вопроса; base ставит золото на позицию 2, teacher — на 1
    items = [
        {"relevant": ["g1"], "shortlist_keys": ["x", "g1", "y"], "teacher_scores": [0.1, 0.9, 0.2]},
        {"relevant": ["g2"], "shortlist_keys": ["z", "g2", "w"], "teacher_scores": [0.1, 0.9, 0.2]},
    ]
    table = sweep_quality(items, [1, 3])
    # k=3: teacher поднял золото на 1 → hit@1 teacher=1.0, base=0.0
    assert table[3]["teacher"]["hit@1"] == 1.0
    assert table[3]["base"]["hit@1"] == 0.0
    # k=1: шорт-лист = [x]/[z] — золота нет вообще → оба hit@1 = 0.0
    assert table[1]["teacher"]["hit@1"] == 0.0
    assert table[1]["base"]["hit@1"] == 0.0
