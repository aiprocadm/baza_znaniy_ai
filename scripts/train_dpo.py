#!/usr/bin/env python3
"""Train a DPO adapter on top of the W3 SFT adapter.

Lightweight CLI wrapping :class:`trl.DPOTrainer`. Local TDD uses
the stub under ``tests/stubs/trl`` (no real ML deps required);
CI / production install real ``trl~=0.11`` and run the same
script unmodified.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("scripts.train_dpo")


# Mirrors scripts.train_lora.PROMPT_TEMPLATE_RAG verbatim so the W3 SFT
# adapter and the W4 DPO adapter both see the same prefix at training time.
# Inlined here (rather than imported) because train_lora pulls in torch at
# module level, and DPO dataset generation should run without torch.
_PROMPT_TEMPLATE_RAG = (
    "<s>[INST] <<SYS>>\n"
    "Ответь на вопрос, используя контекст и свои знания. Если контекст "
    "релевантен — приоритизируй его. Указывай источник цитаты в "
    "формате [doc_chunk:X].\n"
    "<</SYS>>\n\n"
    "Контекст:\n{retrieved_context}\n\n"
    "Вопрос: {instruction} [/INST]\n"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO adapter from a JSONL dataset.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--sft-adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-mode", choices=["generic", "rag"], default="rag")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Train dataset not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed line: %s", exc)
    if not rows:
        LOGGER.warning("Empty dataset at %s; trainer will record an empty run.", path)
    return rows


def _apply_prompt_mode(rows: list[dict], prompt_mode: str) -> list[dict]:
    """Optionally re-format prompts using the W3 RAG template.

    When prompt_mode='rag', wraps each row's prompt in the same
    template the production inference pipeline uses, so the trainer
    sees identical prefixes to inference.
    """

    if prompt_mode == "generic":
        return rows

    out: list[dict] = []
    for row in rows:
        prompt = row.get("prompt", "")
        rewritten = _PROMPT_TEMPLATE_RAG.format(
            instruction=prompt,
            retrieved_context="",
        )
        out.append({**row, "prompt": rewritten})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        import trl
    except ImportError as exc:  # pragma: no cover - depends on env
        raise SystemExit(f"trl is required: {exc}. Install with `pip install trl~=0.11`.")

    rows = _load_dataset(args.train)
    rows = _apply_prompt_mode(rows, args.prompt_mode)

    cfg = trl.DPOConfig(
        output_dir=str(args.output),
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_steps=args.max_steps,
    )

    trainer = trl.DPOTrainer(
        model=args.base_model,
        args=cfg,
        train_dataset=rows,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    LOGGER.info("DPO adapter saved to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
