"""Build the stage-1 reranker pre-train set from Russian mr-TyDi (spec Phase 1 §3.1).

Stream ``castorini/mr-tydi`` (russian) -> ``{query, text, teacher_score}`` jsonl with
synthetic binary labels (positive=1.0, negative=0.0). Each record carries 1 positive
and ~30 pre-mined hard negatives, so no teacher pass and no own negative-mining is
needed — the pairwise loss only needs within-query ordering. ``datasets`` is imported
lazily so stub-backed unit tests never load it. Requires ``datasets==3.6.0`` +
``trust_remote_code=True`` (mr-TyDi is a script dataset; datasets 4.0+ dropped it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS_OUT = Path("var/data/rerank/mrtydi_pairs.jsonl")
MRTYDI_DATASET = "castorini/mr-tydi"
MRTYDI_CONFIG = "russian"


def to_pairs(query: str, positive: str, negatives: list[str]) -> list[dict]:
    """One record -> scored rows: positive=1.0, each non-blank negative=0.0.
    A blank query or positive yields nothing (no usable ordering signal)."""
    if not (query.strip() and positive.strip()):
        return []
    rows = [{"query": query, "text": positive, "teacher_score": 1.0}]
    for neg in negatives:
        if neg.strip():
            rows.append({"query": query, "text": neg, "teacher_score": 0.0})
    return rows


def record_to_texts(record: dict, *, max_negs: int) -> tuple[str, str, list[str]]:
    """Pull (query, positive_text, [negative_texts]) from a mr-TyDi record.
    Uses the first positive passage; caps negatives at ``max_negs``."""
    query = record["query"]
    positives = record.get("positive_passages") or []
    negatives = record.get("negative_passages") or []
    positive = positives[0]["text"] if positives else ""
    neg_texts = [n["text"] for n in negatives[:max_negs]]
    return query, positive, neg_texts


from typing import Iterable, Iterator


def take_first(records: Iterable, limit: int) -> Iterator:
    """Yield the first ``limit`` items of an iterable (deterministic subsample of
    a streaming dataset). ``limit <= 0`` yields nothing."""
    for i, record in enumerate(records):
        if i >= limit:
            break
        yield record


def iter_records(limit: int, *, max_negs: int):
    """Yield up to ``limit`` (query, positive, [negatives]) tuples from streamed
    Russian mr-TyDi. Lazy ``datasets`` import keeps unit tests ML-free."""
    from datasets import load_dataset

    ds = load_dataset(
        MRTYDI_DATASET,
        MRTYDI_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    for record in take_first(ds, limit):
        yield record_to_texts(record, max_negs=max_negs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build_mrtydi_pairs")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="number of queries to keep (dataset has ~5k; >size = all)",
    )
    parser.add_argument(
        "--negs", type=int, default=10, help="hard negatives kept per query (mr-TyDi has ~30)"
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stream into a temp file and promote it to ``out`` only on full success, so a
    # mid-stream kill leaves ``out`` absent (not a truncated file the turnkey
    # runner would trust as "already mined" and skip — training stage-1 on a
    # partial set). The single-pass write makes atomic rename the right tool here;
    # build_pravo_pairs keeps its incremental append because it resumes across kills.
    tmp = out.with_name(out.name + ".tmp")
    n_rows = 0
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for query, positive, negatives in iter_records(args.limit, max_negs=args.negs):
                for row in to_pairs(query, positive, negatives):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
    except BaseException:
        # Kill/interrupt/error mid-stream: drop the partial temp, never touch out.
        tmp.unlink(missing_ok=True)
        raise
    if n_rows == 0:
        # Empty stream (network/version issue): don't promote a 0-row file — a
        # resume run would trust it as "already mined" and fail at stage-1 on empty
        # input. Fail loud and clean instead (mirrors build_pravo_pairs).
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"No mr-TyDi rows written to {out} — the dataset stream was empty. "
            "Needs datasets==3.6.0 + trust_remote_code and HF network access "
            "(datasets 4.0+ dropped the mr-TyDi script)."
        )
    tmp.replace(out)  # atomic promote: out exists only once fully written
    print(f"Wrote {n_rows} rows to {out}")


if __name__ == "__main__":
    main()
