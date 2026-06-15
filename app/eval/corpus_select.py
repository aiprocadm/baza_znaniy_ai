"""Resolve which eval corpus a run targets: public (committed) vs private (local).

The eval harness has two halves (spec 2026-06-06 §7):

* ``public``  — the committed synthetic corpus + golden. Deterministic, shipped
  in-tree, the input to the offline CI gate (``app/eval/frozen.py``).
* ``private`` — the operator's real documents, kept entirely local for
  trustworthy absolute numbers and LLM-judge scoring. Their paths come from the
  environment and are **never committed**.

``KB_EVAL_CORPUS`` selects the half (default ``public``). For ``private`` both
``KB_EVAL_PRIVATE_DB`` and ``KB_EVAL_PRIVATE_GOLDEN`` must be set — otherwise the
run fails loudly via :class:`CorpusSelectionError` rather than silently scoring
the public corpus and reporting it as private results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Public-half defaults. The store path mirrors the PR2 ingest convention
# (``KB_MVP_DB_PATH=var/data/kb_public.sqlite``); the golden is the committed set.
PUBLIC_DB = Path("var/data/kb_public.sqlite")
PUBLIC_GOLDEN = Path("data/eval/golden_public.jsonl")


class CorpusSelectionError(RuntimeError):
    """Raised when the requested corpus cannot be resolved (bad name / missing env)."""


@dataclass(frozen=True)
class CorpusPaths:
    label: str  # "public" | "private"
    db_path: Path
    golden_path: Path


def resolve_corpus(env: Mapping[str, str] | None = None) -> CorpusPaths:
    """Resolve ``(db_path, golden_path)`` for the selected corpus half.

    *env* defaults to ``os.environ``; passing a dict keeps the resolver pure and
    unit-testable.
    """
    env = os.environ if env is None else env
    choice = (env.get("KB_EVAL_CORPUS") or "public").strip().lower()

    if choice == "public":
        return CorpusPaths(label="public", db_path=PUBLIC_DB, golden_path=PUBLIC_GOLDEN)

    if choice == "private":
        db = (env.get("KB_EVAL_PRIVATE_DB") or "").strip()
        golden = (env.get("KB_EVAL_PRIVATE_GOLDEN") or "").strip()
        missing = [
            name
            for name, value in (
                ("KB_EVAL_PRIVATE_DB", db),
                ("KB_EVAL_PRIVATE_GOLDEN", golden),
            )
            if not value
        ]
        if missing:
            raise CorpusSelectionError(
                "KB_EVAL_CORPUS=private requires " + ", ".join(missing) + " to be set"
            )
        return CorpusPaths(label="private", db_path=Path(db), golden_path=Path(golden))

    raise CorpusSelectionError(
        f"unknown KB_EVAL_CORPUS={choice!r} (expected 'public' or 'private')"
    )
