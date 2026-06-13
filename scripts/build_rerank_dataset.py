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
import os
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
    source_key: str = ""


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
    for query, source_key in queries:
        if normalize_question(query) in banned:
            continue
        for chunk_key, text in retrieve(query, k):
            out.append(Pair(query=query, chunk_key=chunk_key, text=text, source_key=source_key))
    leaked = {normalize_question(p.query) for p in out} & banned
    assert not leaked, f"golden leak into training pairs: {sorted(leaked)[:3]}"
    return out


def _pair_row(pair: Pair, score: float) -> dict:
    """Serialize one scored pair to its on-disk JSON record.

    ``source_key`` is the *source* chunk a query was generated from (distinct
    from ``chunk_key``, the mined candidate). It is the resume marker: a row's
    presence means that source chunk is fully generated, mined, and scored.
    """
    return {
        "query": pair.query,
        "chunk_key": pair.chunk_key,
        "text": pair.text,
        "teacher_score": float(score),
        "source_key": pair.source_key,
    }


def append_rows(path: Path, pairs: Sequence[Pair], scores: Sequence[float]) -> None:
    """Append scored pairs to ``path`` (JSONL), creating it if absent.

    Used on the incremental/per-chunk path so a kill mid-run keeps everything
    flushed so far. Flushes + fsyncs each batch so a hard kill cannot leave a
    torn final line.
    """
    if len(pairs) != len(scores):
        raise ValueError(f"pairs/scores length mismatch: {len(pairs)} != {len(scores)}")
    if not pairs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for pair, score in zip(pairs, scores, strict=True):
            fh.write(json.dumps(_pair_row(pair, score), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_meta(path: Path, meta: dict) -> None:
    """Write/refresh the ``.meta.json`` sidecar next to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def write_pairs(path: Path, pairs: Sequence[Pair], scores: Sequence[float], meta: dict) -> None:
    """Overwrite ``path`` with all pairs + write the meta sidecar (batch path)."""
    if len(pairs) != len(scores):
        raise ValueError(f"pairs/scores length mismatch: {len(pairs)} != {len(scores)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair, score in zip(pairs, scores, strict=True):
            fh.write(json.dumps(_pair_row(pair, score), ensure_ascii=False) + "\n")
    write_meta(path, meta)


def completed_source_keys(path: Path) -> set[str]:
    """Return the set of source chunk keys already represented in ``path``.

    A source key on disk means that chunk's queries were generated, mined, and
    teacher-scored — so resume can skip it. Tolerates a torn final line (a
    partial write from a hard kill) and rows missing ``source_key`` (legacy
    format, which simply contribute nothing to skip).
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line from a hard kill
        key = row.get("source_key")
        if key:
            done.add(key)
    return done


def filter_done_queries(
    queries: Sequence[tuple[str, str]], done_source_keys: set[str]
) -> list[tuple[str, str]]:
    """Drop queries whose source chunk is already completed on disk."""
    return [(q, src) for q, src in queries if src not in done_source_keys]


def count_rows(path: Path) -> int:
    """Count non-blank JSONL rows in ``path`` (0 if absent)."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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


def group_by_source(queries: Sequence[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    """Group ``(query, source_key)`` tuples by source key, preserving first-seen order.

    The unit of resumable work is one source chunk: all its queries are mined +
    teacher-scored + flushed together so a row's presence on disk means that
    source chunk is fully done.
    """
    order: list[str] = []
    grouped: dict[str, list[str]] = {}
    for query, source_key in queries:
        if source_key not in grouped:
            grouped[source_key] = []
            order.append(source_key)
        grouped[source_key].append(query)
    return [(src, grouped[src]) for src in order]


def teacher_scores(pairs: Sequence[Pair], *, model_name: str, batch_size: int) -> list[float]:
    """Score (query, text) with the teacher cross-encoder. CPU-slow."""
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(model_name, max_length=512)
    scores = encoder.predict(
        [(p.query, p.text) for p in pairs], batch_size=batch_size, show_progress_bar=True
    )
    return [float(s) for s in scores]


def score_and_flush_by_chunk(
    queries: Sequence[tuple[str, str]],
    retrieve: Retrieve,
    golden_questions: frozenset[str],
    score_fn: Callable[[Sequence[Pair]], list[float]],
    *,
    out: Path,
    k: int = 20,
) -> int:
    """Per-source-chunk: mine -> score -> append, flushing after each chunk.

    Returns the number of pairs newly appended. ``score_fn`` maps a chunk's
    candidate pairs to teacher scores (kept as a parameter so unit tests inject
    a pure stub instead of the cross-encoder). The anti-leak filter runs per
    chunk via :func:`build_pairs`, so the guarantee holds on this path too.
    """
    new_pairs = 0
    for source_key, group in group_by_source(queries):
        chunk_pairs = build_pairs([(q, source_key) for q in group], retrieve, golden_questions, k=k)
        if not chunk_pairs:
            continue  # all queries for this chunk were golden-filtered
        scores = score_fn(chunk_pairs)
        append_rows(out, chunk_pairs, scores)
        new_pairs += len(chunk_pairs)
        LOGGER.info(
            "source chunk %s: +%d pairs (%d total appended this run)",
            source_key,
            len(chunk_pairs),
            new_pairs,
        )
    return new_pairs


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to --out, skipping source chunks already completed on disk "
        "(a killed run loses at most the in-flight chunk)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.eval.adapter import make_mvp_retriever
    from app.eval.dataset import load_golden
    from app.services.kb_store import get_store
    from scripts.eval_rag import _gen_provider

    out_path = Path(args.out)
    if not args.resume and out_path.exists():
        # Fresh run: start clean so resume bookkeeping (source_key markers) is
        # never mixed with a previous run's rows.
        out_path.unlink()

    done = completed_source_keys(out_path) if args.resume else set()
    if done:
        LOGGER.info("resume: %d source chunks already on disk in %s", len(done), out_path)

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
    pending = filter_done_queries(queries, done)
    LOGGER.info(
        "%d queries generated, %d pending after resume-skip (%d source chunks done)",
        len(queries),
        len(pending),
        len(done),
    )

    golden_questions = frozenset(item.question for item in load_golden(GOLDEN_PUBLIC))
    retrieve = as_retrieve(make_mvp_retriever(store))

    def _score(chunk_pairs: Sequence[Pair]) -> list[float]:
        return teacher_scores(chunk_pairs, model_name=args.teacher, batch_size=args.batch)

    new_pairs = score_and_flush_by_chunk(
        pending, retrieve, golden_questions, _score, out=out_path, k=args.candidates
    )

    total_pairs = count_rows(out_path)
    if total_pairs == 0:
        raise SystemExit(
            f"No pairs generated (queries={len(queries)}). "
            "Check KB_MVP_DB_PATH and that the store is ingested."
        )
    write_meta(
        out_path,
        meta={
            "teacher": args.teacher,
            "rounds": args.rounds,
            "candidates": args.candidates,
            "stride": args.stride,
            "offset": args.offset,
            "self_consistency": not args.no_self_consistency,
            "resume": args.resume,
            "n_source_chunks_done": len(completed_source_keys(out_path)),
            "n_pairs": total_pairs,
            "golden_excluded": str(GOLDEN_PUBLIC),
        },
    )
    print(
        f"Wrote {new_pairs} new pairs this run; {total_pairs} pairs total "
        f"({len(pending)} queries processed) to {args.out}"
    )


if __name__ == "__main__":
    main()
