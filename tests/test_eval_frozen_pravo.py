"""Офлайн frozen-гейт качества reranker'а на корпусе права.

Детерминированный, БЕЗ модели: грузит замороженные ранжирования base/teacher,
пересчитывает метрики и ассертит абсолютные floors + дельты teacher-over-base из
data/eval/ci_thresholds_pravo.json. Зеркалит публичный гейт в test_eval_frozen.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.dataset import read_signature
from app.eval.pravo_gate import aggregate_side, gate_failures

FROZEN = Path("data/eval/frozen_pravo_natural.json")
GOLDEN = Path("data/eval/golden_pravo_natural.jsonl")
THRESHOLDS = Path("data/eval/ci_thresholds_pravo.json")


def _load_frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_frozen_sig_matches_golden() -> None:
    """Заморозка должна быть построена против того же корпуса, что и golden."""
    frozen = _load_frozen()
    gold_sig = read_signature(GOLDEN)
    assert gold_sig is not None, "golden_pravo_natural has no .sig.json"
    assert (
        frozen["_sig"] == gold_sig.to_dict()
    ), "frozen fixture sig drift — re-run scripts/freeze_pravo_eval.py"


def test_pravo_teacher_meets_floors_and_beats_base() -> None:
    """ГЕЙТ: teacher ≥ floors И teacher − base ≥ min deltas на замороженном наборе."""
    frozen = _load_frozen()
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    items = frozen["items"]
    assert items, "frozen fixture has no items"

    base = aggregate_side(items, "base_ranked")
    teacher = aggregate_side(items, "teacher_ranked")
    failures = gate_failures(base, teacher, thresholds)
    assert not failures, f"pravo reranker gate failed: {failures}"
