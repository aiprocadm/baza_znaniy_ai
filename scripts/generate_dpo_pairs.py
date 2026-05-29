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
                chunk_id = meta.get("source_chunk_id")
                if chunk_id is not None:
                    try:
                        seen.add(int(chunk_id))
                    except (TypeError, ValueError):
                        continue
    return seen


def _estimate_cost(target_pairs: int, proportions) -> float:
    """Estimate teacher-call cost: ~$0.0005 per call (DeepSeek-V3 baseline)."""

    from app.services.dpo_dataset import RejectStrategy

    paid_share = sum(
        share
        for strategy, share in proportions.items()
        if strategy != RejectStrategy.NO_CITATION
    )
    teacher_calls = int(round(target_pairs * paid_share))
    return teacher_calls * 0.0005


def _make_teacher(args):
    """Build the teacher callable from configured LLM provider env vars.

    Test hooks monkeypatch this function to inject a fake.
    """

    from app.services.kb_llm import create_llm_provider

    provider = create_llm_provider()

    def teacher(prompt: str) -> str:
        try:
            return provider.complete(prompt, max_tokens=512)
        except Exception as exc:
            LOGGER.warning("Teacher call failed: %s", exc)
            return ""

    return teacher


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.services.dpo_dataset import DPOPairBuilder, default_synthetic_proportions

    if not args.seeds.is_file():
        raise SystemExit(f"Seeds file not found: {args.seeds}")

    proportions = default_synthetic_proportions()
    cost = _estimate_cost(args.target_pairs, proportions)
    if cost > args.max_cost_usd and not args.yes:
        raise SystemExit(
            f"Estimated ${cost:.2f} > budget ${args.max_cost_usd:.2f}. "
            "Pass --yes to override."
        )

    seeds = _load_seeds(args.seeds)
    if not seeds:
        LOGGER.warning("No seeds loaded from %s; nothing to do.", args.seeds)
        return 0

    if args.resume:
        already = _resume_seed_ids(args.output)
        before = len(seeds)
        seeds = [s for s in seeds if s.source_chunk_id not in already]
        LOGGER.info("Resume: skipping %d seeds already in output.", before - len(seeds))

    teacher = _make_teacher(args)
    builder = DPOPairBuilder(teacher=teacher, proportions=proportions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    with args.output.open(open_mode, encoding="utf-8") as fh:
        for pair in builder.build(seeds, total=args.target_pairs):
            fh.write(pair.to_jsonl_line())
            fh.flush()
            written += 1

    LOGGER.info("Done: %d DPO pairs written to %s", written, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
