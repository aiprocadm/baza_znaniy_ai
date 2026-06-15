"""Build the structural held-out golden for the pravo headroom probe (Phase 0).

Each held-out article's heading topic becomes a query; the article's own chunks
are the relevant set (retrieving ANY of them counts as a hit). No LLM — this is
the structural-golden choice from the spec. Heavy imports (the store) are lazy;
the pure helpers below are unit-testable without ML deps.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from app.eval.dataset import GoldenItem, save_golden, write_signature

LOGGER = logging.getLogger(__name__)

GOLDEN_OUT = Path("data/eval/golden_pravo.jsonl")
_HEADING_RE = re.compile(r"^\s*Статья\s+[\d.]+\.?\s*")


def heading_to_query(article: str) -> str:
    """Strip the «Статья N.» prefix, leaving the topic phrase as the query."""
    return _HEADING_RE.sub("", article).strip()


def select_heldout(docs, *, stride: int):
    """Every *stride*-th document — even coverage across codes."""
    return docs[::stride] if stride > 1 else list(docs)


def build_golden_items(heldout) -> list[GoldenItem]:
    """``(filename, title, [chunk_index, ...])`` rows -> GoldenItems.

    Relevant set = every chunk of the article. Rows with an empty query or no
    chunks are skipped (cannot be a usable eval item).
    """
    items: list[GoldenItem] = []
    for filename, title, indices in heldout:
        query = heading_to_query(title)
        if not query or not indices:
            continue
        keys = tuple(f"{filename}:{i}" for i in indices)
        items.append(GoldenItem(question=query, relevant_chunks=keys, source="auto"))
    return items


def documents_with_chunks(store):
    """Return ``[(filename, title, [chunk_index, ...]), ...]`` grouped by document,
    preserving DB order. Reads only metadata (no embedding)."""
    with store._connect() as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT d.filename, d.title, c.chunk_index "
            "FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id "
            "ORDER BY d.id, c.chunk_index"
        ).fetchall()
    grouped: list[tuple[str, str, list[int]]] = []
    index_by_file: dict[str, int] = {}
    for filename, title, chunk_index in rows:
        if filename not in index_by_file:
            index_by_file[filename] = len(grouped)
            grouped.append((filename, title, []))
        grouped[index_by_file[filename]][2].append(int(chunk_index))
    return grouped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build_pravo_golden")
    parser.add_argument("--out", default=str(GOLDEN_OUT))
    parser.add_argument(
        "--stride",
        type=int,
        default=80,
        help="hold out every Nth article as an eval query (~6141/80 ≈ 77 queries)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.eval.adapter import compute_signature
    from app.services.kb_store import get_store

    store = get_store()
    docs = documents_with_chunks(store)
    if not docs:
        raise SystemExit("Store is empty — run scripts.ingest_pravo first (check KB_MVP_DB_PATH).")
    heldout = select_heldout(docs, stride=args.stride)
    items = build_golden_items(heldout)
    if not items:
        raise SystemExit("No golden items produced — check the corpus headings.")

    out = Path(args.out)
    save_golden(out, items)
    write_signature(out, compute_signature(store))
    print(f"Wrote {len(items)} golden items + signature to {out}")


if __name__ == "__main__":
    main()
