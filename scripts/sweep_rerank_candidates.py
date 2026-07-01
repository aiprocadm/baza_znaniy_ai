"""Свип «число кандидатов reranker'а ↔ качество/латентность» на корпусе права.

Один инструментированный проход захватывает top-N шорт-лист би-энкодера и teacher-скор
(bge) на каждого кандидата → пишет коммитимый фикстур ``data/eval/rerank_sweep_pravo.json``.
Качество «реранк top-k» реконструируется офлайн (``app/eval/candidate_sweep.py``). С
флагом ``--latency`` дополнительно мерит p50/p95 реального bge на CPU по каждому k,
single-process, с прогревом (переиспользуя ``scripts.bench_reranker``).

Запуск (Windows; эмбеддер st обязателен — хэш-эмбеддер даёт мусор):
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.sweep_rerank_candidates
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.sweep_rerank_candidates --latency
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from app.eval.candidate_sweep import sweep_quality

DEFAULT_STORE = "var/data/pravo_public.sqlite"
DEFAULT_GOLDEN = "data/eval/golden_pravo_natural.jsonl"
DEFAULT_OUT = "data/eval/rerank_sweep_pravo.json"
DEFAULT_KS = "1,2,3,5,8,10,12,16,20"
DEFAULT_SHORTLIST = 20
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_HEADLINE = ("hit@1", "hit@3", "hit@5", "mrr@5", "recall@10")


def _parse_ks(text: str) -> list[int]:
    """'1,2, ,5' -> [1, 2, 5] (пустые токены игнорируются)."""
    return [int(tok) for tok in text.split(",") if tok.strip()]


def _fixture_items(items: Sequence[dict]) -> list[dict]:
    """Проекция для коммита: только поля, нужные для офлайн-качества."""
    return [
        {
            "relevant": it["relevant"],
            "shortlist_keys": it["shortlist_keys"],
            "teacher_scores": it["teacher_scores"],
        }
        for it in items
    ]


def capture_items(store: Any, golden: Sequence[Any], model: Any, shortlist: int) -> list[dict]:
    """Один проход модели: на вопрос — top-N шорт-лист (ключи+тексты) + teacher-скор.

    Тексты остаются в items в памяти (для ``--latency``), но в фикстур не пишутся.
    """
    from app.eval.adapter import make_mvp_retriever

    base = make_mvp_retriever(store)
    items: list[dict] = []
    for it in golden:
        hits = base(it.question, shortlist)
        texts = [h.text for h in hits]
        scores = [float(s) for s in model.predict([(it.question, t) for t in texts])]
        items.append(
            {
                "question": it.question,
                "relevant": list(it.relevant_chunks),
                "shortlist_keys": [h.chunk_key for h in hits],
                "teacher_scores": scores,
                "_texts": texts,
            }
        )
    return items


def print_quality(items: Sequence[dict], candidate_ks: Sequence[int]) -> None:
    table = sweep_quality(items, candidate_ks)
    print(f"\nquality sweep (n={len(items)}):")
    print("  " + f"{'k':>4} | " + " ".join(f"{m:>10}" for m in _HEADLINE))
    for k in candidate_ks:
        t = table[k]["teacher"]
        print("  " + f"{k:>4} | " + " ".join(f"{t[m]:>10.3f}" for m in _HEADLINE))
    base_row = table[candidate_ks[-1]]["base"]
    print("  " + f"{'base':>4} | " + " ".join(f"{base_row[m]:>10.3f}" for m in _HEADLINE))


def print_latency(
    items: Sequence[dict],
    candidate_ks: Sequence[int],
    model: Any,
    budget_ms: float,
    warmup: int,
) -> None:
    from scripts.bench_reranker import measure, percentile

    queries = [(it["question"], it["_texts"]) for it in items if it["_texts"]]
    warm = queries[:warmup]
    timed = queries[warmup:] or queries
    print(f"\nlatency sweep (n={len(timed)}, budget {budget_ms:.0f}ms, single-process):")
    print("  " + f"{'k':>4} | {'p50(ms)':>9} {'p95(ms)':>9}  verdict")
    for k in candidate_ks:
        if warm:
            measure(model.predict, warm, candidates=k)  # прогрев, отбрасывается
        timings = measure(model.predict, timed, candidates=k)
        p50 = percentile(timings, 0.50)
        p95 = percentile(timings, 0.95)
        verdict = "PASS" if p95 <= budget_ms else "FAIL"
        print("  " + f"{k:>4} | {p50:>9.0f} {p95:>9.0f}  {verdict}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.environ.get("KB_RERANK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--ks", default=DEFAULT_KS)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--budget-ms", type=float, default=200.0)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)

    candidate_ks = _parse_ks(args.ks)

    # Стор выбирается через env до первого get_store(); сбрасываем кэш стора.
    os.environ["KB_MVP_DB_PATH"] = str(Path(args.store))
    from app.services.kb_store import get_store, reset_default_store

    reset_default_store()
    store = get_store()

    from app.eval.adapter import compute_signature
    from app.eval.dataset import load_golden

    golden = load_golden(Path(args.golden))
    if not golden:
        raise SystemExit(f"empty golden: {args.golden}")

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.model)

    items = capture_items(store, golden, model, args.shortlist)

    fixture = {
        "_sig": compute_signature(store).to_dict(),
        "_measured": {"shortlist": args.shortlist, "ks": candidate_ks},
        "items": _fixture_items(items),
    }
    Path(args.out).write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}  n={len(items)}  store={args.store}")

    print_quality(items, candidate_ks)
    if args.latency:
        print_latency(items, candidate_ks, model, args.budget_ms, args.warmup)


if __name__ == "__main__":
    main()
