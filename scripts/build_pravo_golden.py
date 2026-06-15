"""Build the structural held-out golden for the pravo headroom probe (Phase 0).

Each held-out article's heading topic becomes a query; the article's own chunks
are the relevant set (retrieving ANY of them counts as a hit). No LLM — this is
the structural-golden choice from the spec. Heavy imports (the store) are lazy;
the pure helpers below are unit-testable without ML deps.
"""

from __future__ import annotations

import argparse  # noqa: F401 — used in main() added in Increment B
import logging
import re
from pathlib import Path

from app.eval.dataset import (
    GoldenItem,
    save_golden,  # noqa: F401 — used in main() added in Increment B
    write_signature,  # noqa: F401 — used in main() added in Increment B
)

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
