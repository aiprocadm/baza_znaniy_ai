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
from app.eval.dataset import load_golden

PAIRS_OUT = Path("var/data/rerank/pravo_pairs.jsonl")
GOLDEN_PRAVO = Path("data/eval/golden_pravo.jsonl")
GOLDEN_PRAVO_NATURAL = Path("data/eval/golden_pravo_natural.jsonl")
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


def load_golden_questions(path: Path) -> frozenset[str]:
    """Held-out golden questions to exclude from mined training pairs (anti-leak,
    spec §3.2). Missing file => empty set (golden not built yet is not an error
    here; the leak assert in build_pairs is the real backstop). Reads the canonical
    GoldenItem JSONL (``instruction`` field) via ``load_golden`` — NOT a hand-rolled
    key, which previously crashed on the real format."""
    if not path.exists():
        return frozenset()
    return frozenset(item.question for item in load_golden(path))


def main(argv: list[str] | None = None) -> None:
    import logging

    parser = argparse.ArgumentParser(prog="build_pravo_pairs")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument("--golden", default=str(GOLDEN_PRAVO))
    parser.add_argument("--golden-natural", default=str(GOLDEN_PRAVO_NATURAL))
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--k", type=int, default=20, help="hard negatives mined per query")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.eval.adapter import make_mvp_retriever
    from app.services.kb_store import get_store
    from scripts.build_pravo_golden import documents_with_chunks
    from scripts.build_rerank_dataset import as_retrieve, count_rows, score_and_flush_by_chunk
    from sentence_transformers import CrossEncoder

    store = get_store()
    docs = documents_with_chunks(store)
    if not docs:
        raise SystemExit("Store is empty — run scripts.ingest_pravo first (check KB_MVP_DB_PATH).")

    queries = articles_to_queries(docs)
    golden = load_golden_questions(Path(args.golden)) | load_golden_questions(Path(args.golden_natural))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # fresh run — keep source_key resume markers clean

    retrieve = as_retrieve(make_mvp_retriever(store))
    encoder = CrossEncoder(args.teacher, max_length=512)

    def _score(chunk_pairs) -> list[float]:
        scores = encoder.predict([(p.query, p.text) for p in chunk_pairs], batch_size=args.batch)
        return [float(s) for s in scores]

    new_pairs = score_and_flush_by_chunk(queries, retrieve, golden, _score, out=out, k=args.k)
    if count_rows(out) == 0:
        raise SystemExit("No pairs mined — check the corpus and golden exclusion.")
    print(f"Wrote {new_pairs} teacher-scored pairs to {out}")


if __name__ == "__main__":
    main()
