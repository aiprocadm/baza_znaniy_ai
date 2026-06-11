"""CPU-latency gate for the distilled reranker (spec 2026-06-10 §4.2).

Reranking 20 candidates must fit the tier-B budget (default 200 ms p95).
Measures end-to-end ``CrossEncoder.predict`` wall time per query over real
pairs from the training set. Exit code 1 on budget violation, so the run is
recordable as a pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_PAIRS = Path("var/data/rerank/pairs.jsonl")
DEFAULT_MODEL = "var/models/kbai-reranker-ru"

ScoreFn = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


def group_queries(pairs_path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(row["query"], []).append(row["text"])
    return grouped


def measure(
    score_fn: ScoreFn, queries: Sequence[tuple[str, list[str]]], *, candidates: int
) -> list[float]:
    timings: list[float] = []
    for query, texts in queries:
        batch = [(query, text) for text in texts[:candidates]]
        started = time.perf_counter()
        score_fn(batch)
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def percentile(timings: Sequence[float], q: float) -> float:
    ordered = sorted(timings)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bench_reranker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    args = parser.parse_args(argv)

    grouped = group_queries(Path(args.pairs))
    sample = [(q, t) for q, t in grouped.items() if len(t) >= args.candidates][: args.queries]
    if not sample:
        raise SystemExit("No queries with enough candidates in the pairs file.")
    if len(sample) < 10:
        print(f"WARNING: only {len(sample)} qualifying queries — p95 will be unreliable")

    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(args.model, max_length=384)
    warmup_set = sample[:2]
    timed_set = sample[2:] or sample  # tiny corpora: better biased than empty
    measure(encoder.predict, warmup_set, candidates=args.candidates)
    timings = measure(encoder.predict, timed_set, candidates=args.candidates)
    p50 = percentile(timings, 0.50)
    p95 = percentile(timings, 0.95)
    print(
        f"rerank {args.candidates} candidates x {len(timings)} queries: "
        f"p50={p50:.0f}ms p95={p95:.0f}ms (budget {args.budget_ms:.0f}ms)"
    )
    if p95 > args.budget_ms:
        raise SystemExit(f"FAIL: p95 {p95:.0f}ms > budget {args.budget_ms:.0f}ms")
    print("PASS")


if __name__ == "__main__":
    main()
