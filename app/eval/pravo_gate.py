"""Чистая логика frozen-гейта качества reranker'а на корпусе права.

Без I/O и без модели: работает по уже замороженным спискам ранжированных
chunk-ключей, поэтому гейт исполняется детерминированно в CI без загрузки
``bge-reranker-v2-m3``. Две проверки: метрики teacher не ниже абсолютных floors
и превосходство teacher над base не меньше зафиксированных дельт.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.eval.metrics import RETRIEVAL_KS, aggregate, score_item

# Допуск на float-сравнения (зеркалит публичный гейт в test_eval_frozen.py).
EPS = 1e-9


def aggregate_side(
    items: Sequence[Mapping[str, Any]],
    side: str,
    ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[str, float]:
    """Агрегировать метрики ретривала для одной замороженной стороны.

    ``side`` — ключ списка ранжирования в каждом item ("base_ranked" /
    "teacher_ranked"); ``item["relevant"]`` — релевантные chunk-ключи.
    """
    rows = [score_item(it["relevant"], it[side], ks) for it in items]
    return aggregate(rows)


def gate_failures(
    base: Mapping[str, float],
    teacher: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Вернуть список человекочитаемых нарушений; пустой список = гейт пройден."""
    failures: list[str] = []
    for metric, floor in thresholds.get("teacher_floors", {}).items():
        got = float(teacher.get(metric, 0.0))
        if got + EPS < float(floor):
            failures.append(f"teacher {metric}={got:.4f} below floor {floor}")
    for metric, dmin in thresholds.get("min_delta_over_base", {}).items():
        delta = float(teacher.get(metric, 0.0)) - float(base.get(metric, 0.0))
        if delta + EPS < float(dmin):
            failures.append(f"teacher-over-base {metric} delta={delta:.4f} below min {dmin}")
    return failures


def student_gate(
    base: Mapping[str, float],
    student: Mapping[str, float],
    *,
    min_delta: float = 0.05,
) -> dict[str, Any]:
    """GO/NO-GO ученика-reranker'а против base (Phase 1 §4).

    GO ⟺ ученик обходит base хотя бы по одной топ-ранговой метрике
    (``mrr@5`` ИЛИ ``hit@1``) на ``min_delta`` И не регрессирует по покрытию
    (``recall@5`` не ниже base). Возвращает структурированный вердикт для
    самоотчёта оркестратора: ``{"passed", "reasons", "deltas"}`` — пустой
    ``reasons`` означает GO.
    """
    deltas = {
        metric: round(float(student.get(metric, 0.0)) - float(base.get(metric, 0.0)), 4)
        for metric in ("hit@1", "mrr@5", "recall@5")
    }
    reasons: list[str] = []
    beats = deltas["mrr@5"] + EPS >= min_delta or deltas["hit@1"] + EPS >= min_delta
    if not beats:
        reasons.append(
            f"student beats base by neither mrr@5 ({deltas['mrr@5']:+.4f}) nor "
            f"hit@1 ({deltas['hit@1']:+.4f}); below min +{min_delta}"
        )
    if deltas["recall@5"] + EPS < 0.0:
        reasons.append(f"student recall@5 regressed vs base (delta {deltas['recall@5']:+.4f})")
    return {"passed": not reasons, "reasons": reasons, "deltas": deltas}
