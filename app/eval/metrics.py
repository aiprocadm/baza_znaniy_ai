"""Pure retrieval-quality metrics over chunk keys. No I/O, no env, no globals."""

from __future__ import annotations

from typing import Collection, Sequence

RETRIEVAL_KS: tuple[int, ...] = (1, 3, 5, 10)


def hit_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def recall_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    topk = set(retrieved[:k])
    return sum(1 for cid in rel if cid in topk) / len(rel)


def mrr_at_k(relevant: Collection[str], retrieved: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def score_item(
    relevant: Collection[str],
    retrieved: Sequence[str],
    ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(relevant, retrieved, k)
        out[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        out[f"mrr@{k}"] = mrr_at_k(relevant, retrieved, k)
    return out


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {key: sum(r[key] for r in rows) / n for key in rows[0]}
