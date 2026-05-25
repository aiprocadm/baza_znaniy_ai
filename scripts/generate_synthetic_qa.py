#!/usr/bin/env python3
"""Generate a synthetic Q&A dataset from a KB corpus.

This is the CLI wrapper for Workstream 1 of the Pack B++ ML
strengthening plan. The pure logic lives in
``app.services.synthetic_qa``; this module only handles argument
parsing, provider/store wiring, the streaming JSONL writer, the
budget guard and resume support.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from app.services.kb_llm import (
    LLMUnavailable,
    OpenAICompatibleProvider,
    build_provider,
    select_provider,
)
from app.services.kb_store import KnowledgeBaseStore
from app.services.synthetic_qa import (
    GenerationMode,
    SyntheticQAGenerator,
    estimate_total_cost_usd,
    iter_chunks,
    load_processed_chunk_ids,
)

LOGGER = logging.getLogger("scripts.generate_synthetic_qa")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Q&A dataset from a KB corpus via a teacher LLM."
    )
    parser.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to KB SQLite file (e.g. var/data/kb_mvp.sqlite).",
    )
    parser.add_argument(
        "--provider", default=None,
        help="Teacher LLM provider name (deepseek, groq, openrouter, openai, ollama, custom). "
             "Defaults to KB_LLM_PROVIDER env or auto-selection.",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in GenerationMode],
        default=GenerationMode.SINGLE.value,
        help="Generation strategy (default: single).",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSONL file (created or appended to).",
    )
    parser.add_argument(
        "--document-id", type=int, default=None,
        help="Restrict generation to one document id (default: all chunks).",
    )
    parser.add_argument(
        "--multi-hop-chunks", type=int, default=3,
        help="How many chunks to combine when mode=multi-hop (default 3).",
    )
    parser.add_argument(
        "--max-budget-usd", type=float, default=5.0,
        help="Abort if estimated cost exceeds this many USD (default 5.0).",
    )
    parser.add_argument(
        "--no-budget-guard", action="store_true",
        help="Disable the budget guard entirely (use with care).",
    )
    parser.add_argument(
        "--no-self-consistency", action="store_true",
        help="Disable the second-generation self-consistency check.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip chunks already represented in the output JSONL.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _load_store(args: argparse.Namespace) -> KnowledgeBaseStore:
    """Open the KnowledgeBaseStore pointed at by --corpus."""

    if not args.corpus.is_file():
        raise SystemExit(f"Corpus file not found: {args.corpus}")
    return KnowledgeBaseStore(db_path=args.corpus)


def _load_provider(args: argparse.Namespace) -> OpenAICompatibleProvider:
    """Build the teacher provider from --provider or env autoselection."""

    if args.provider:
        try:
            return build_provider(args.provider)
        except LLMUnavailable as exc:
            raise SystemExit(f"LLM provider unusable: {exc}")
    selected = select_provider()
    if selected is None:
        raise SystemExit(
            "No LLM provider configured. Set KB_LLM_PROVIDER or one of "
            "DEEPSEEK_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY."
        )
    return selected


def _enforce_budget(
    args: argparse.Namespace,
    provider: OpenAICompatibleProvider,
    chunk_chars: list[int],
) -> None:
    if args.no_budget_guard:
        LOGGER.info("Budget guard disabled by --no-budget-guard")
        return

    mode = GenerationMode(args.mode)
    estimate = estimate_total_cost_usd(
        provider=provider.name,
        model=provider.model,
        mode=mode,
        chunk_chars=chunk_chars,
    )
    if estimate is None:
        LOGGER.warning(
            "No pricing data for (%s, %s); budget guard disabled.",
            provider.name, provider.model,
        )
        return

    LOGGER.info("Estimated cost: $%.4f (budget cap $%.4f)", estimate, args.max_budget_usd)
    if estimate > args.max_budget_usd:
        raise SystemExit(
            f"Estimated cost ${estimate:.4f} exceeds budget cap ${args.max_budget_usd:.4f}. "
            "Increase --max-budget-usd or trim the corpus."
        )


def _select_chunk_batches(
    chunks: list[tuple[int, str]],
    mode: GenerationMode,
    multi_hop_size: int,
) -> list[list[tuple[int, str]]]:
    if mode is not GenerationMode.MULTI_HOP:
        return [[chunk] for chunk in chunks]
    if multi_hop_size < 2:
        raise SystemExit("--multi-hop-chunks must be >= 2 for multi-hop mode")
    return [
        chunks[i : i + multi_hop_size]
        for i in range(0, len(chunks), multi_hop_size)
        if len(chunks[i : i + multi_hop_size]) >= 2
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.log_level)

    store = _load_store(args)
    provider = _load_provider(args)
    mode = GenerationMode(args.mode)

    LOGGER.info(
        "Generator config: provider=%s model=%s mode=%s",
        provider.name, provider.model, mode.value,
    )

    all_chunks = list(iter_chunks(store, document_id=args.document_id))
    if not all_chunks:
        LOGGER.warning("No chunks found in corpus; nothing to do.")
        return 0

    processed: set[int] = set()
    if args.resume:
        processed = load_processed_chunk_ids(args.output)
        LOGGER.info("Resume: %d chunks already in %s", len(processed), args.output)

    remaining = [(cid, text) for cid, text in all_chunks if cid not in processed]
    if not remaining:
        LOGGER.info("All chunks already processed; exiting cleanly.")
        return 0

    _enforce_budget(args, provider, [len(text) for _, text in remaining])

    generator = SyntheticQAGenerator(
        provider=provider,
        check_self_consistency=not args.no_self_consistency,
    )

    batches = _select_chunk_batches(remaining, mode, args.multi_hop_chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"

    written = 0
    with args.output.open(open_mode, encoding="utf-8") as handle:
        for batch in batches:
            chunk_ids = [cid for cid, _ in batch]
            chunk_texts = [text for _, text in batch]
            try:
                pairs = generator.generate_for_chunk(
                    chunks=chunk_texts,
                    chunk_ids=chunk_ids,
                    mode=mode,
                )
            except Exception:
                LOGGER.exception("Generation failed for chunks %s", chunk_ids)
                continue

            for pair in pairs:
                handle.write(pair.to_jsonl_line())
                handle.flush()
                written += 1

            LOGGER.info(
                "Batch chunks=%s kept=%d total=%d",
                chunk_ids, len(pairs), written,
            )

    LOGGER.info("Done: %d Q&A pairs written to %s", written, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
