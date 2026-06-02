"""Run a retriever over golden items and score retrieval quality."""
from __future__ import annotations

from typing import Sequence

from app.eval.adapter import Retriever
from app.eval.dataset import GoldenItem
from app.eval.metrics import RETRIEVAL_KS, aggregate, score_item


def evaluate_item(item: GoldenItem, retriever: Retriever, *, max_k: int) -> dict[str, float]:
    hits = retriever(item.question, max_k)
    retrieved_ids = [h.chunk_id for h in hits]
    return score_item(item.relevant_chunk_ids, retrieved_ids)


def evaluate(
    items: Sequence[GoldenItem],
    retriever: Retriever,
    ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[str, object]:
    max_k = max(ks) if ks else 0
    per_item = [evaluate_item(it, retriever, max_k=max_k) for it in items]
    return {"n": len(items), "per_item": per_item, "aggregate": aggregate(per_item)}
