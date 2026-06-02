#!/usr/bin/env python3
"""Generate a RAG-aware SFT dataset by composing W1 seeds with retrieval.

This is the CLI wrapper for Workstream 3 of the Pack B++ ML
strengthening plan. The pure logic lives in
``app.services.rag_dataset``; this module handles argument parsing,
seed loading, retriever wiring, streaming JSONL writes, and resume.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("scripts.generate_rag_dataset")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a RAG-aware SFT dataset from W1 seeds + KB retrieval."
    )
    parser.add_argument("--corpus", required=True, type=Path, help="Path to KB SQLite file.")
    parser.add_argument(
        "--seeds",
        required=True,
        type=Path,
        help="Path to W1-generated synthetic Q&A JSONL (input seeds).",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--target-pairs",
        type=int,
        required=True,
        help="Total number of RAG samples to emit (apportioned across variants).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k chunks to retrieve per question (default: 3).",
    )
    parser.add_argument(
        "--negative-document-id",
        type=int,
        default=None,
        help=(
            "If set, draw IRRELEVANT/PARTIAL pool chunks from this document id. "
            "Otherwise, pool is sampled from any document other than the seed's source."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seed chunk ids already represented in --output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_seeds(path: Path):
    from app.services.synthetic_qa import QAPair

    seeds: list[QAPair] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                seeds.append(QAPair.from_jsonl_line(raw))
            except (ValueError, KeyError) as exc:
                LOGGER.warning("Skipping malformed seed line: %s", exc)
    return seeds


def _negative_pool(
    store,
    *,
    exclude_document_id: int | None,
    negative_document_id: int | None,
):
    # EvalHit carries the global kb_chunks.id under ``.chunk_id`` — the identity
    # the builder matches/cites on (shared with the eval harness adapter).
    from app.eval.adapter import EvalHit

    pool: list[EvalHit] = []
    with store._connect() as conn:  # noqa: SLF001
        sql = "SELECT id, text FROM kb_chunks"
        params: tuple = ()
        clauses: list[str] = []
        if negative_document_id is not None:
            clauses.append("document_id = ?")
            params = (int(negative_document_id),)
        elif exclude_document_id is not None:
            clauses.append("document_id != ?")
            params = (int(exclude_document_id),)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id ASC LIMIT 50"
        for row in conn.execute(sql, params):
            pool.append(EvalHit(chunk_id=int(row[0]), text=str(row[1] or "")))
    return pool


def _resume_seed_ids(path: Path) -> set[int]:
    import json

    seen: set[int] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            meta = data.get("meta") if isinstance(data, dict) else None
            if isinstance(meta, dict):
                try:
                    seen.add(int(meta.get("source_chunk_id")))
                except (TypeError, ValueError):
                    continue
    return seen


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.services.kb_store import KnowledgeBaseStore
    from app.services.rag_dataset import RAGSampleBuilder, default_proportions

    if not args.corpus.is_file():
        raise SystemExit(f"Corpus file not found: {args.corpus}")
    if not args.seeds.is_file():
        raise SystemExit(f"Seeds file not found: {args.seeds}")

    store = KnowledgeBaseStore(db_path=args.corpus)
    seeds = _load_seeds(args.seeds)
    if not seeds:
        LOGGER.warning("No seeds loaded from %s; nothing to do.", args.seeds)
        return 0

    if args.resume:
        already = _resume_seed_ids(args.output)
        before = len(seeds)
        seeds = [s for s in seeds if s.source_chunk_id not in already]
        LOGGER.info("Resume: skipping %d seeds already in output.", before - len(seeds))

    # Resolve each SearchHit's (document_id, chunk_index) back to its global
    # kb_chunks.id so the builder matches the seed's source_chunk_id (also a
    # global id). Shared resolver with the eval harness — see app/eval/adapter.py
    # and the app.services.rag_dataset module docstring for why.
    from app.eval.adapter import make_mvp_retriever

    retriever = make_mvp_retriever(store)

    pool = _negative_pool(
        store,
        exclude_document_id=None,
        negative_document_id=args.negative_document_id,
    )

    builder = RAGSampleBuilder(
        retriever=retriever,
        negative_pool=pool,
        distractor_pool=pool,
        proportions=default_proportions(),
        top_k=args.top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    with args.output.open(open_mode, encoding="utf-8") as fh:
        for sample in builder.build(seeds, total=args.target_pairs):
            fh.write(sample.to_jsonl_line())
            fh.flush()
            written += 1

    LOGGER.info("Done: %d RAG samples written to %s", written, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
