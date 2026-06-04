"""Sweep retrieval depth (top_k) and tabulate the Phase-2 gate metrics.

For each ``--values`` top_k, run the MVP eval and print recall / completeness /
faithfulness, so the argmax (per the gate: ``completeness`` up without
``faithfulness`` down) is chosen by evidence rather than guessed. Requires a
real embedder (and, with ``--judge``, an LLM) — same loud guard as
``eval_rag run``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.eval import generation_eval, guards, retrieval_eval
from app.eval.adapter import compute_signature, make_mvp_retriever
from app.eval.dataset import load_golden
from app.services.kb_store import get_store
from scripts.eval_rag import _gen_provider, _judge_provider

_COLUMNS = ("top_k", "recall@5", "recall@10", "completeness", "faithfulness")


def _parse_values(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise SystemExit("--values must list at least one integer, e.g. 5,8,10,12")
    return values


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "—"


def format_sweep_table(rows: list[dict]) -> str:
    """Render one markdown row per top_k. Pure (no I/O) → unit-testable."""
    out = [" | ".join(_COLUMNS), " | ".join("---" for _ in _COLUMNS)]
    for row in rows:
        cells = [
            str(row.get("top_k", "")) if col == "top_k" else _fmt(row.get(col)) for col in _COLUMNS
        ]
        out.append(" | ".join(cells))
    return "\n".join(out)


def cmd_sweep(args: argparse.Namespace) -> None:
    store = get_store()
    guards.ensure_real_embedder(compute_signature(store), allow_hashing=args.allow_hashing)
    golden = load_golden(Path(args.golden))
    if not golden:
        raise SystemExit(f"Golden set is empty: {args.golden}")
    retriever = make_mvp_retriever(store)

    rows: list[dict] = []
    for top_k in _parse_values(args.values):
        retrieval_agg = retrieval_eval.evaluate(golden, retriever, top_k=top_k)["aggregate"]
        row: dict[str, object] = {
            "top_k": top_k,
            "recall@5": retrieval_agg.get("recall@5"),  # type: ignore[union-attr]
            "recall@10": retrieval_agg.get("recall@10"),  # type: ignore[union-attr]
        }
        if args.judge:
            gen_agg = generation_eval.evaluate_generation(
                golden,
                retriever,
                gen_provider=_gen_provider(),
                judge_provider=_judge_provider(),
                top_k=top_k,
            )["aggregate"]
            row["completeness"] = gen_agg.get("completeness")  # type: ignore[union-attr]
            row["faithfulness"] = gen_agg.get("faithfulness")  # type: ignore[union-attr]
        rows.append(row)
        print(f"  swept top_k={top_k}", flush=True)

    print(format_sweep_table(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_sweep", description="Sweep top_k and tabulate gate metrics."
    )
    parser.add_argument("--golden", default="var/data/eval/golden_auto.jsonl")
    parser.add_argument("--values", default="5,8,10,12", help="comma-separated top_k values")
    parser.add_argument("--judge", action="store_true", help="also score generation (needs LLM)")
    parser.add_argument("--allow-hashing", action="store_true")
    parser.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
