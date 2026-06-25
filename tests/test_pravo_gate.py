"""Юнит-тесты чистой логики frozen-гейта pravo (без модели, детерминированно)."""

from __future__ import annotations

from app.eval.pravo_gate import aggregate_side, gate_failures


# Два вопроса, по три релевантных ключа. base ставит релевант на 2-ю позицию,
# teacher — на 1-ю, поэтому hit@1 teacher = 1.0, base = 0.0.
_ITEMS = [
    {
        "relevant": ["a:0"],
        "base_ranked": ["x:0", "a:0", "y:0"],
        "teacher_ranked": ["a:0", "x:0", "y:0"],
    },
    {
        "relevant": ["b:0"],
        "base_ranked": ["z:0", "b:0", "w:0"],
        "teacher_ranked": ["b:0", "z:0", "w:0"],
    },
]


def test_aggregate_side_computes_hit_at_1() -> None:
    base = aggregate_side(_ITEMS, "base_ranked")
    teacher = aggregate_side(_ITEMS, "teacher_ranked")
    assert base["hit@1"] == 0.0
    assert teacher["hit@1"] == 1.0
    # base нашёл релевант на позиции 2 → mrr@5 = 0.5; teacher на 1 → 1.0
    assert base["mrr@5"] == 0.5
    assert teacher["mrr@5"] == 1.0


def test_gate_passes_when_floors_and_deltas_met() -> None:
    base = {"hit@1": 0.0, "mrr@5": 0.5}
    teacher = {"hit@1": 1.0, "mrr@5": 1.0}
    thresholds = {
        "teacher_floors": {"hit@1": 0.84, "mrr@5": 0.86},
        "min_delta_over_base": {"hit@1": 0.05, "mrr@5": 0.04},
    }
    assert gate_failures(base, teacher, thresholds) == []


def test_gate_flags_floor_violation() -> None:
    base = {"hit@1": 0.0}
    teacher = {"hit@1": 0.80}  # ниже floor 0.84
    thresholds = {"teacher_floors": {"hit@1": 0.84}, "min_delta_over_base": {}}
    failures = gate_failures(base, teacher, thresholds)
    assert len(failures) == 1 and "floor" in failures[0]


def test_gate_flags_delta_violation() -> None:
    base = {"hit@1": 0.85}
    teacher = {"hit@1": 0.87}  # выше floor, но дельта 0.02 < 0.05
    thresholds = {
        "teacher_floors": {"hit@1": 0.84},
        "min_delta_over_base": {"hit@1": 0.05},
    }
    failures = gate_failures(base, teacher, thresholds)
    assert len(failures) == 1 and "delta" in failures[0]
