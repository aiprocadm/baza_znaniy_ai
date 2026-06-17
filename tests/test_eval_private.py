"""Private-corpus eval smoke test — runs only when a local private corpus exists.

The private half of the eval (real documents + golden) is never committed; its
paths come from ``KB_EVAL_PRIVATE_DB`` / ``KB_EVAL_PRIVATE_GOLDEN``. This test
**skips loudly** (with a reason that names what is missing) when that corpus is
not configured, so the coverage gap is visible rather than a silent pass. When
the corpus IS present it confirms the wiring end-to-end: the selector resolves
to the private paths and the golden loads from there.

It does NOT measure retrieval/judge quality — those gate B/D numbers are an
operator step on a real corpus (often a GPU box), recorded in the runbook, not
asserted here against fabricated thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.eval.corpus_select import resolve_corpus
from app.eval.dataset import load_golden


@pytest.mark.integration
def test_private_corpus_resolves_and_golden_loads() -> None:
    db = os.environ.get("KB_EVAL_PRIVATE_DB", "").strip()
    golden = os.environ.get("KB_EVAL_PRIVATE_GOLDEN", "").strip()
    if not db or not golden:
        pytest.skip(
            "private corpus not configured — set KB_EVAL_PRIVATE_DB and "
            "KB_EVAL_PRIVATE_GOLDEN to run the private-half smoke test"
        )
    if not Path(db).exists():
        pytest.skip(f"KB_EVAL_PRIVATE_DB points at a missing file: {db}")
    if not Path(golden).exists():
        pytest.skip(f"KB_EVAL_PRIVATE_GOLDEN points at a missing file: {golden}")

    paths = resolve_corpus(
        {
            "KB_EVAL_CORPUS": "private",
            "KB_EVAL_PRIVATE_DB": db,
            "KB_EVAL_PRIVATE_GOLDEN": golden,
        }
    )
    assert paths.label == "private"
    assert paths.db_path == Path(db)

    items = load_golden(paths.golden_path)
    assert items, "private golden set is empty"
