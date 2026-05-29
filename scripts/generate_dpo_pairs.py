#!/usr/bin/env python3
"""Generate a synthetic DPO preference dataset.

CLI wrapper for Workstream 4. Pure logic lives in
``app.services.dpo_dataset``; this module handles argument parsing,
seed loading, teacher-provider wiring, streaming JSONL writes,
budget guard, and resume.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("scripts.generate_dpo_pairs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a synthetic DPO preference dataset from W1 seeds."
    )
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
        help="Total number of DPO pairs to emit (40/30/30 across strategies).",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=1.0,
        help="Abort if estimated teacher-call cost exceeds this (default $1.00).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost-confirmation prompt (override the cost guard).",
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
