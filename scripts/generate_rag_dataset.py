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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Stub: CLI not yet wired. args=%s", args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
