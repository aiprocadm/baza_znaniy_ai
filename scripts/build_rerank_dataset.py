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

import argparse
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


def as_retrieve(eval_retriever) -> Retrieve:
    """Adapt an ``app.eval.adapter`` Retriever (EvalHit) to (chunk_key, text)."""

    def _retrieve(query: str, k: int) -> list[tuple[str, str]]:
        return [(h.chunk_key, h.text) for h in eval_retriever(query, k)]

    return _retrieve


def dedupe_queries(queries: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for query, source_key in queries:
        norm = normalize_question(query)
        if norm in seen:
            continue
        seen.add(norm)
        out.append((query, source_key))
    return out


def select_chunks(chunks: list, *, stride: int = 1, offset: int = 0, limit: int = 0) -> list:
    """Evenly sample every *stride*-th chunk starting at *offset*, then cap at *limit* (0 = no cap).

    When offset > 0 or stride > 1, slices as chunks[offset::stride]. This allows disjoint
    subsets across successive runs (e.g., offset=0 stride=3 → [0,3,6], offset=1 stride=3 → [1,4,7]).
    """
    if stride > 1 or offset:
        chunks = chunks[offset::stride]
    if limit:
        chunks = chunks[:limit]
    return chunks


def generate_queries(
    store,
    provider,
    *,
    rounds: int,
    limit_chunks: int = 0,
    stride: int = 1,
    offset: int = 0,
    self_consistency: bool = True,
) -> list[tuple[str, str]]:
    """Synthetic (query, source_chunk_key) via the W1 generator. LLM-slow."""
    from app.eval.adapter import build_global_id_key_map
    from app.services import synthetic_qa as sq

    generator = sq.SyntheticQAGenerator(provider=provider, check_self_consistency=self_consistency)
    key_map = build_global_id_key_map(store)
    chunks = select_chunks(
        list(sq.iter_chunks(store)), stride=stride, offset=offset, limit=limit_chunks
    )
    queries: list[tuple[str, str]] = []
    dropped = 0
    for round_no in range(rounds):
        for chunk_id, text in chunks:
            for qa in generator.generate_for_chunk(
                chunks=[text], chunk_ids=[chunk_id], mode=sq.GenerationMode.SINGLE
            ):
                key = key_map.get(qa.source_chunk_id)
                if key is None:
                    dropped += 1
                    continue
                queries.append((qa.instruction, key))
        LOGGER.info(
            "round %d/%d: %d queries so far, %d dropped (unknown chunk_id)",
            round_no + 1,
            rounds,
            len(queries),
            dropped,
        )
    return dedupe_queries(queries)


def teacher_scores(pairs: Sequence[Pair], *, model_name: str, batch_size: int) -> list[float]:
    """Score (query, text) with the teacher cross-encoder. CPU-slow."""
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(model_name, max_length=512)
    scores = encoder.predict(
        [(p.query, p.text) for p in pairs], batch_size=batch_size, show_progress_bar=True
    )
    return [float(s) for s in scores]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build_rerank_dataset")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument("--rounds", type=int, default=3, help="QA-generation passes per chunk")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit-chunks", type=int, default=0, help="smoke runs (0 = all)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="sample every Nth chunk (even coverage across docs)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="start index for stride sampling (cover a different subset per run)",
    )
    parser.add_argument(
        "--no-self-consistency",
        action="store_true",
        help="single LLM call per chunk; noisy queries are fine for distillation (teacher labels them)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.eval.adapter import make_mvp_retriever
    from app.eval.dataset import load_golden
    from app.services.kb_store import get_store
    from scripts.eval_rag import _gen_provider

    store = get_store()
    queries = generate_queries(
        store,
        _gen_provider(),
        rounds=args.rounds,
        limit_chunks=args.limit_chunks,
        stride=args.stride,
        offset=args.offset,
        self_consistency=not args.no_self_consistency,
    )
    golden_questions = frozenset(item.question for item in load_golden(GOLDEN_PUBLIC))
    pairs = build_pairs(
        queries, as_retrieve(make_mvp_retriever(store)), golden_questions, k=args.candidates
    )
    if not pairs:
        raise SystemExit(
            f"No pairs generated (queries={len(queries)}). "
            "Check KB_MVP_DB_PATH and that the store is ingested."
        )
    LOGGER.info("scoring %d pairs with teacher %s", len(pairs), args.teacher)
    scores = teacher_scores(pairs, model_name=args.teacher, batch_size=args.batch)
    write_pairs(
        Path(args.out),
        pairs,
        scores,
        meta={
            "teacher": args.teacher,
            "rounds": args.rounds,
            "candidates": args.candidates,
            "stride": args.stride,
            "offset": args.offset,
            "self_consistency": not args.no_self_consistency,
            "n_queries": len(queries),
            "n_pairs": len(pairs),
            "golden_excluded": str(GOLDEN_PUBLIC),
        },
    )
    print(f"Wrote {len(pairs)} pairs ({len(queries)} queries) to {args.out}")


if __name__ == "__main__":
    main()
