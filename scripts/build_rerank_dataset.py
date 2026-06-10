"""Build the reranker-distillation training set (spec 2026-06-10).

Pipeline: synthetic queries per chunk (W1 generator) -> candidate mining via
the production bi-encoder (``store.search``) -> teacher scores
(bge-reranker-v2-m3). Output: ``var/data/rerank/pairs.jsonl`` + ``.meta.json``
sidecar. Queries colliding with the public golden are excluded (anti-leak,
spec §3.4) — enforced in code, with an assert as backstop.

Heavy imports (sentence_transformers, the LLM provider) are lazy: importing
this module must stay cheap so stub-backed unit tests never touch ML deps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

LOGGER = logging.getLogger(__name__)

PAIRS_OUT = Path("var/data/rerank/pairs.jsonl")
GOLDEN_PUBLIC = Path("data/eval/golden_public.jsonl")
DEFAULT_TEACHER = "BAAI/bge-reranker-v2-m3"

# (query, k) -> [(chunk_key, text), ...]
Retrieve = Callable[[str, int], Sequence[tuple[str, str]]]


@dataclass(frozen=True)
class Pair:
    query: str
    chunk_key: str
    text: str


def normalize_question(q: str) -> str:
    """Collapse whitespace/case/trailing punctuation for leak comparison."""
    return " ".join(q.lower().split()).rstrip("?!.… ")


def build_pairs(
    queries: Sequence[tuple[str, str]],
    retrieve: Retrieve,
    golden_questions: frozenset[str],
    *,
    k: int = 20,
) -> list[Pair]:
    """Mine top-*k* candidates per query, dropping golden-colliding queries."""
    banned = {normalize_question(q) for q in golden_questions}
    out: list[Pair] = []
    for query, _source_key in queries:
        if normalize_question(query) in banned:
            continue
        for chunk_key, text in retrieve(query, k):
            out.append(Pair(query=query, chunk_key=chunk_key, text=text))
    leaked = {normalize_question(p.query) for p in out} & banned
    assert not leaked, f"golden leak into training pairs: {sorted(leaked)[:3]}"
    return out


def write_pairs(path: Path, pairs: Sequence[Pair], scores: Sequence[float], meta: dict) -> None:
    if len(pairs) != len(scores):
        raise ValueError(f"pairs/scores length mismatch: {len(pairs)} != {len(scores)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair, score in zip(pairs, scores, strict=True):
            fh.write(
                json.dumps(
                    {
                        "query": pair.query,
                        "chunk_key": pair.chunk_key,
                        "text": pair.text,
                        "teacher_score": float(score),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
