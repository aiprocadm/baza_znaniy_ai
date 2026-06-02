"""CLI for the RAG answer-quality eval harness.

Subcommands:
  run      — score retrieval (and, with --judge, generation) on the MVP surface
  generate — build a golden set from the corpus (added in a later task)
  compare  — diff two saved run JSONs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.eval import report as report_mod
from app.eval import retrieval_eval
from app.eval.adapter import compute_signature, make_mvp_retriever
from app.eval.dataset import load_golden, read_signature
from app.services.kb_store import get_store


def cmd_run(args: argparse.Namespace) -> None:
    store = get_store()
    sig = compute_signature(store)
    if sig.embedder_name == "hash" and not args.allow_hashing:
        raise SystemExit(
            "Refusing to produce a baseline on the hashing embedder (near-random "
            "results). Configure KB_EMBEDDINGS_BACKEND=ollama|api (+ model/base), or "
            "pass --allow-hashing for a throwaway smoke run."
        )
    golden_path = Path(args.golden)
    golden = load_golden(golden_path)
    gold_sig = read_signature(golden_path)
    if gold_sig is not None and gold_sig != sig:
        raise SystemExit(
            f"Corpus signature mismatch — golden was built against {gold_sig.to_dict()} "
            f"but the live corpus is {sig.to_dict()}. Regenerate the golden set."
        )
    retriever = make_mvp_retriever(store)
    retrieval = retrieval_eval.evaluate(golden, retriever)
    rep = report_mod.build_report(surface="mvp", signature=sig.to_dict(), retrieval=retrieval)
    report_mod.save_report(Path(args.out), rep)
    print(report_mod.to_markdown(rep))


def cmd_compare(args: argparse.Namespace) -> None:
    run_a = json.loads(Path(args.run_a).read_text(encoding="utf-8"))
    run_b = json.loads(Path(args.run_b).read_text(encoding="utf-8"))
    print(report_mod.compare(run_a, run_b))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval_rag")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score retrieval on the MVP surface")
    run.add_argument("--golden", default="data/eval/golden_curated.jsonl")
    run.add_argument("--out", default="var/data/eval/run.json")
    run.add_argument("--allow-hashing", action="store_true")
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="diff two run JSONs")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
