"""Build the stage-2 reranker fine-tune set: structural pravo pairs (spec Phase 1 §3.2).

Heading topic -> query; the article is the positive; hard negatives are the
bi-encoder's top-k confusable neighbours from the pravo store; teacher scores
come from bge-reranker-v2-m3. No LLM — this removes v1/v2's CPU query-generation
bottleneck. Reuses ``heading_to_query`` (build_pravo_golden) and ``build_pairs`` /
``normalize_question`` (build_rerank_dataset). Heavy imports (store, teacher) are
lazy so stub-backed unit tests stay ML-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_pravo_golden import heading_to_query

PAIRS_OUT = Path("var/data/rerank/pravo_pairs.jsonl")
GOLDEN_PRAVO = Path("data/eval/golden_pravo.jsonl")
DEFAULT_TEACHER = "BAAI/bge-reranker-v2-m3"


def articles_to_queries(docs) -> list[tuple[str, str]]:
    """``(filename, title, [chunk_index, ...])`` rows -> ``(query, source_key)``.

    Query = heading topic (the «Статья N.» prefix stripped); source_key = the
    article's filename (threads through build_pairs for resume bookkeeping).
    Rows whose heading has no topic are dropped — they cannot be a query.
    """
    out: list[tuple[str, str]] = []
    for filename, title, _indices in docs:
        query = heading_to_query(title)
        if query:
            out.append((query, filename))
    return out
