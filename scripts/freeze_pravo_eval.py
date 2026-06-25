"""Заморозить ранжирования base и teacher на golden_pravo_natural для офлайн-гейта.

Гоняет ОБА пайплайна один раз реальными моделями и пишет коммитимый фикстур с
ранжированными chunk-ключами на каждый вопрос, чтобы tests/test_eval_frozen_pravo.py
пересчитывал метрики детерминированно, не загружая bge-reranker-v2-m3 в CI.

Запуск (Windows, эмбеддер st обязателен — иначе хэш-эмбеддер даёт мусор):
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.freeze_pravo_eval
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_STORE = "var/data/pravo_public.sqlite"
DEFAULT_GOLDEN = "data/eval/golden_pravo_natural.jsonl"
DEFAULT_OUT = "data/eval/frozen_pravo_natural.json"
_HEADLINE = ("hit@1", "hit@3", "mrr@5", "recall@10")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    # Стор выбирается через env до первого get_store(); сбрасываем кэш стора.
    os.environ["KB_MVP_DB_PATH"] = str(Path(args.store))
    from app.services.kb_store import get_store, reset_default_store

    reset_default_store()
    store = get_store()

    from app.eval.adapter import (
        compute_signature,
        make_mvp_reranking_retriever,
        make_mvp_retriever,
    )
    from app.eval.dataset import load_golden
    from app.eval.pravo_gate import aggregate_side

    golden = load_golden(Path(args.golden))
    if not golden:
        raise SystemExit(f"empty golden: {args.golden}")

    base = make_mvp_retriever(store)
    teacher = make_mvp_reranking_retriever(store)  # enabled форсится, модель = bge по умолчанию

    items: list[dict[str, object]] = []
    for it in golden:
        items.append(
            {
                "relevant": list(it.relevant_chunks),
                "base_ranked": [h.chunk_key for h in base(it.question, args.top_k)],
                "teacher_ranked": [h.chunk_key for h in teacher(it.question, args.top_k)],
            }
        )

    measured = {
        "base": aggregate_side(items, "base_ranked"),
        "teacher": aggregate_side(items, "teacher_ranked"),
    }
    fixture = {"_sig": compute_signature(store).to_dict(), "_measured": measured, "items": items}
    Path(args.out).write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {args.out}  n={len(items)}  store={args.store}")
    for metric in _HEADLINE:
        b = measured["base"][metric]
        t = measured["teacher"][metric]
        print(f"  {metric:10s} base={b:.3f} teacher={t:.3f} delta={t - b:+.3f}")


if __name__ == "__main__":
    main()
