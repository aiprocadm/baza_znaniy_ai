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
