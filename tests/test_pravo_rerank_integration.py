"""Живой integration-чек reranker'а права (вне дефолтного CI).

Гоняет настоящий bge-reranker-v2-m3 по реальному стору и ассертит те же floors +
дельты, что и frozen-гейт. Это то, что запускают перед рефризом фикстуры:
    KB_PRAVO_LIVE=1 KB_EMBEDDINGS_BACKEND=st py -3.13 -m pytest -m integration -k pravo_rerank
Полный прогон занимает ~30 мин на CPU, поэтому по умолчанию тест skip'ается, пока
не выставлен явный opt-in KB_PRAVO_LIVE. Также громко skip'ается, если стор/эмбеддер
недоступны.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.eval.dataset import load_golden
from app.eval.pravo_gate import aggregate_side, gate_failures

STORE = Path("var/data/pravo_public.sqlite")
GOLDEN = Path("data/eval/golden_pravo_natural.jsonl")
THRESHOLDS = Path("data/eval/ci_thresholds_pravo.json")


@pytest.mark.integration
def test_pravo_teacher_beats_base_live() -> None:
    # Явный opt-in: без KB_PRAVO_LIVE тяжёлый (~30 мин) прогон не запускается даже
    # при обычном `pytest tests/` — дефолтный backend "st" сам по себе не защищает.
    if not os.environ.get("KB_PRAVO_LIVE"):
        pytest.skip("set KB_PRAVO_LIVE=1 to run the ~30-min live pravo rerank check")
    if not STORE.exists():
        pytest.skip(f"pravo store missing: {STORE}")
    if os.environ.get("KB_EMBEDDINGS_BACKEND", "st") != "st":
        pytest.skip("KB_EMBEDDINGS_BACKEND is not 'st' — would score with hashing embedder")

    os.environ["KB_MVP_DB_PATH"] = str(STORE)
    try:
        from app.services.kb_store import get_store, reset_default_store

        reset_default_store()
        store = get_store()
        from app.eval.adapter import make_mvp_reranking_retriever, make_mvp_retriever
    except Exception as exc:  # noqa: BLE001 — любая проблема загрузки = громкий skip
        pytest.skip(f"eval stack unavailable: {exc}")

    golden = load_golden(GOLDEN)
    assert golden, "golden_pravo_natural is empty"

    base_r = make_mvp_retriever(store)
    teacher_r = make_mvp_reranking_retriever(store)
    items = [
        {
            "relevant": list(it.relevant_chunks),
            "base_ranked": [h.chunk_key for h in base_r(it.question, 10)],
            "teacher_ranked": [h.chunk_key for h in teacher_r(it.question, 10)],
        }
        for it in golden
    ]

    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    base = aggregate_side(items, "base_ranked")
    teacher = aggregate_side(items, "teacher_ranked")
    failures = gate_failures(base, teacher, thresholds)
    assert not failures, f"LIVE pravo reranker gate failed: {failures}"
