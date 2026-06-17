"""Ingest the pravo corpus (one article = one document) into an MVP store for
the reranker headroom probe (Phase 0, spec 2026-06-15).

Heavy imports (the store + the ST embedder) are lazy so importing this module
stays cheap and the pure helpers are unit-testable without ML deps.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger(__name__)

CORPUS = Path("experiments/pravo_nn/data/corpus/corpus.jsonl")


def iter_articles(path: Path) -> Iterator[dict]:
    """Yield each article record from the corpus JSONL, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def article_slug(code: str, index: int) -> str:
    """Stable, globally-unique document filename for chunk keys.

    The enumeration ``index`` guarantees uniqueness even though article numbers
    repeat across codes (every code has its own «Статья 1»). Whitespace in the
    code name is collapsed so the slug is a single token.

    Note: uniqueness is per-ingest pass only.  ``enumerate`` restarts at 0 on
    a re-ingest, so calling :func:`ingest_articles` twice into the same store
    will produce duplicate slugs and overwrite earlier documents.
    """
    base = re.sub(r"\s+", "_", code.strip())
    return f"{base}__a{index:05d}"


def ingest_articles(store, articles, *, existing_slugs: set[str] | None = None) -> tuple[int, int]:
    """Add each article as one document. Returns (ingested, skipped).

    Skips (a) articles whose slug is already in ``existing_slugs`` (resume — a
    prior run already stored them) and (b) empty/over-length bodies
    (``add_document`` raises ``ValueError``). Every skip is counted; resume
    skips are silent (expected), value-errors are logged.
    """
    existing = existing_slugs or set()
    n_ok = n_skip = 0
    for idx, art in enumerate(articles):
        slug = article_slug(art["code"], idx)
        if slug in existing:
            n_skip += 1
            continue
        try:
            store.add_document(
                title=art["article"],
                text=art["text"],
                filename=slug,
                source="pravo",
            )
            n_ok += 1
        except ValueError as exc:
            n_skip += 1
            LOGGER.warning("skipped article %d (%s): %s", idx, art.get("article", "?"), exc)
        if (idx + 1) % 200 == 0:
            LOGGER.info("ingested %d articles (%d skipped) so far", n_ok, n_skip)
    return n_ok, n_skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ingest_pravo")
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--limit", type=int, default=0, help="ingest only the first N (smoke runs)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip articles already in the store (re-run after a kill to continue)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.services.kb_store import get_store

    articles = iter_articles(Path(args.corpus))
    if args.limit:
        articles = (a for i, a in enumerate(articles) if i < args.limit)

    store = get_store()

    existing_slugs: set[str] = set()
    if args.resume:
        with store._connect() as conn:  # noqa: SLF001
            existing_slugs = {
                row[0] for row in conn.execute("SELECT filename FROM kb_documents").fetchall()
            }
        LOGGER.info("resume: %d articles already in store, will skip", len(existing_slugs))

    n_ok, n_skip = ingest_articles(store, articles, existing_slugs=existing_slugs)
    print(f"Ingested {n_ok} articles ({n_skip} skipped)")


if __name__ == "__main__":
    main()
