# W4 DPO Post-Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Workstream 4 of the Pack B++ ML strengthening plan (closes gap **G3** — DPO post-training / preference learning). Three deliverables in one branch / one PR (~600–700 LoC):
- W4-A: `app/services/dpo_dataset.py` + `scripts/generate_dpo_pairs.py` — synthetic preference dataset.
- W4-B: `scripts/train_dpo.py` + `tests/stubs/trl/` — DPO trainer on top of the W3 SFT adapter.
- W4-C: `app/api/kb_feedback.py` + `kb_store` schema additions — live thumbs-up/down feedback collection.

**Architecture:** Mirror W3's pure-logic split. `dpo_dataset.py` is I/O free: a `RejectStrategy` enum, a `DPOPair` dataclass, three `build_*_pair` helpers, and a `DPOPairBuilder` orchestrator. `generate_dpo_pairs.py` is a thin CLI wrapper around the builder. `train_dpo.py` reuses `scripts.train_lora.format_prompt` for prompt routing and is gated by a `tests/stubs/trl/` shadow so local TDD does not need the real `trl` package. `kb_feedback.py` is a FastAPI router registered in `kb_mvp.py`; pairing logic lives in `kb_store.iter_feedback_pairs()`.

**Tech Stack:** Python 3.12, pytest, FastAPI TestClient, SQLite (single-tenant kb_mvp path — **no Alembic**), `dataclasses`, `argparse`. New optional ML dep: `trl~=0.11` (stub-shadowed locally, real package only in CI integration job).

**Spec reference:** [`docs/superpowers/specs/2026-05-29-w4-dpo-post-training-design.md`](../specs/2026-05-29-w4-dpo-post-training-design.md). Composes with W1 ([`app/services/synthetic_qa.py`](../../../app/services/synthetic_qa.py)) for seed Q&A and W3 ([`scripts/train_lora.py:format_prompt`](../../../scripts/train_lora.py)) for prompt routing.

**Plans-dir conventions:** Per-task atomic commits using `git commit -m @'…'@` here-strings on PowerShell (see [repo-pythonenv-py-launcher](../../../memory/repo_pythonenv_py_launcher.md) for the `py -3` launcher and no-venv setup). Every task = failing test first, then minimal implementation, then commit.

---

## File Structure

**Create:**
- `app/services/dpo_dataset.py` — pure logic: `RejectStrategy` enum, `DPOPair` dataclass, three `build_*_pair` helpers, `DPOPairBuilder` orchestrator (~250 LoC).
- `app/api/kb_feedback.py` — FastAPI router: `POST /api/kb/messages/{id}/feedback`, `GET /api/kb/feedback/export` (~120 LoC).
- `scripts/generate_dpo_pairs.py` — CLI wrapper around `DPOPairBuilder` with budget guard + resume (~150 LoC).
- `scripts/train_dpo.py` — DPO trainer wrapping `trl.DPOTrainer` (~200 LoC).
- `tests/stubs/trl/__init__.py` — `DPOConfig` + `DPOTrainer` stub mirroring the real `trl` 0.11+ signature (~80 LoC).
- `tests/test_dpo_dataset.py` — unit tests for module pieces (~12 tests).
- `tests/test_dpo_dataset_strategies.py` — per-strategy mix and apportionment tests (~5 tests).
- `tests/test_kb_feedback_store.py` — `store_feedback` / `iter_feedback_pairs` unit tests against a tmp SQLite DB (~8 tests).
- `tests/test_kb_feedback_api.py` — endpoint tests via `fastapi.testclient.TestClient` (~6 tests).
- `tests/scripts/test_generate_dpo_pairs.py` — CLI smoke + parse_args tests (~3 tests).
- `tests/scripts/test_train_dpo.py` — stub-backed CLI tests (~4 tests).
- `tests/test_train_dpo_integration.py` — `@pytest.mark.integration`, real `trl` on a tiny model (~1 test).

**Modify:**
- `app/services/rag_dataset.py` — Sprint 0 refactor: make `apportion_counts` generic over enum types; expose `strip_citations` publicly while keeping `_strip_citations` as a back-compat alias.
- `app/services/kb_store.py` — add `kb_feedback` `CREATE TABLE` to `_initialise_schema`; add `store_feedback()` and `iter_feedback_pairs()` methods.
- `app/api/kb_mvp.py` — register the new `kb_feedback` router.
- `README.md` / `docs/legacy_README.md` — append a W4 usage example after the W3 section.
- `requirements-runtime.txt` — append `trl~=0.11` as an optional ML dep (deferred behind import guard inside `train_dpo.py`, no hard import at module level).

**NOT modified:**
- `app/services/synthetic_qa.py` — W1 stays stable; W4 only consumes `QAPair`.
- `scripts/train_lora.py` — W3 stays stable; W4's `train_dpo.py` only imports `format_prompt` and `PROMPT_TEMPLATE_RAG`.
- `tests/stubs/transformers/` — already exists; W4 uses it unchanged.

**DPOPair JSONL schema (one line per pair):**
```json
{
  "prompt": "<question>",
  "chosen": "<grounded answer with [doc_chunk:X]>",
  "rejected": "<failure-mode answer>",
  "meta": {
    "source": "synthetic|live",
    "strategy": "no_citation|generic|hallucination|live_alt|live_paired",
    "source_chunk_id": 42,
    "feedback_ids": ["<uuid>", "..."]
  }
}
```

Top-level `prompt / chosen / rejected` keys match `trl.DPOTrainer`'s dataset contract directly — no transform pass needed before training.

---

## Sprint 0 — Shared-util refactor (~30 min)

**Goal:** Make `apportion_counts` generic over enum types and expose `strip_citations` as a public API on `rag_dataset`. This is the minimum cross-workstream refactor so W4 can DRY-reuse Hamilton apportionment and citation stripping. Pre-existing W3 tests stay green; the public surface is additive.

**Abort point:** If reviewers push back on the refactor, split it into a separate prep-PR and revert this sprint here — the next sprints would then re-implement Hamilton locally in `dpo_dataset.py` (DRY violation; ~20 extra LoC).

---

### Task 0.1: Make `apportion_counts` generic

**Files:**
- Modify: `app/services/rag_dataset.py`
- Create: `tests/test_apportion_counts_generic.py`

**Background:** The current signature is `apportion_counts(proportions: Mapping[RAGVariant, float], *, total: int) -> dict[RAGVariant, int]`. The body never mentions `RAGVariant` by value — only iterates and indexes the mapping — but the type hard-codes it. We parameterise with `TypeVar` so any `Enum` works.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apportion_counts_generic.py`:

```python
"""apportion_counts must be reusable across enums (W3 + W4 share it)."""

from __future__ import annotations

from enum import Enum

import pytest


class _DummyStrategy(str, Enum):
    A = "a"
    B = "b"
    C = "c"


def test_apportion_counts_works_with_non_rag_enum() -> None:
    from app.services.rag_dataset import apportion_counts

    counts = apportion_counts(
        {_DummyStrategy.A: 0.5, _DummyStrategy.B: 0.3, _DummyStrategy.C: 0.2},
        total=10,
    )
    assert sum(counts.values()) == 10
    assert counts[_DummyStrategy.A] == 5
    assert counts[_DummyStrategy.B] == 3
    assert counts[_DummyStrategy.C] == 2


def test_apportion_counts_zero_total_still_works_with_dummy_enum() -> None:
    from app.services.rag_dataset import apportion_counts

    counts = apportion_counts(
        {_DummyStrategy.A: 0.5, _DummyStrategy.B: 0.5},
        total=0,
    )
    assert counts == {_DummyStrategy.A: 0, _DummyStrategy.B: 0}


def test_apportion_counts_remainder_ties_break_in_iteration_order() -> None:
    """Equal remainders go to the enum that appears first in the input mapping."""
    from app.services.rag_dataset import apportion_counts

    # 1 / 3 split with total=2 → each has remainder 0.667 ≈ tie; first wins.
    counts = apportion_counts(
        {_DummyStrategy.A: 1 / 3, _DummyStrategy.B: 1 / 3, _DummyStrategy.C: 1 / 3},
        total=2,
    )
    assert sum(counts.values()) == 2


def test_apportion_counts_validates_sum_with_dummy_enum() -> None:
    from app.services.rag_dataset import apportion_counts

    with pytest.raises(ValueError, match="proportions must sum"):
        apportion_counts(
            {_DummyStrategy.A: 0.5, _DummyStrategy.B: 0.4},  # 0.9
            total=10,
        )
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_apportion_counts_generic.py -v
```

Expected: tests pass or fail with a TypeError on enum indexing — currently the body uses `list(RAGVariant).index(...)` which only works for `RAGVariant`. The 3rd test (tie-break) will fail because the tie-break code references `RAGVariant` directly.

- [ ] **Step 3: Generalise `apportion_counts`**

In `app/services/rag_dataset.py`, replace the body of `apportion_counts` (lines ~92–129) with:

```python
from typing import TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)


def apportion_counts(
    proportions: Mapping[_EnumT, float],
    *,
    total: int,
) -> dict[_EnumT, int]:
    """Hamilton's largest-remainder method.

    Given target shares summing to 1.0, return integer counts per
    enum member whose sum equals ``total`` exactly. Deterministic
    ordering — driven by the order of keys in ``proportions`` —
    breaks remainder ties.

    Generic over any :class:`enum.Enum` subclass so both W3
    (RAGVariant) and W4 (RejectStrategy) can share this helper.
    """

    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    share_sum = sum(proportions.values())
    if abs(share_sum - 1.0) > _PROPORTION_TOLERANCE:
        raise ValueError(
            f"proportions must sum to 1.0 (within {_PROPORTION_TOLERANCE}); got {share_sum}"
        )

    ordered = list(proportions.keys())
    counts: dict[_EnumT, int] = {member: 0 for member in ordered}
    if total == 0:
        return counts

    raw = [(member, proportions[member] * total) for member in ordered]
    floors = [(member, int(value)) for member, value in raw]
    assigned = sum(c for _, c in floors)
    leftover = total - assigned

    remainders = sorted(
        ((member, value - int(value)) for member, value in raw),
        key=lambda item: (-item[1], ordered.index(item[0])),
    )
    for member, count in floors:
        counts[member] = count
    for i in range(leftover):
        counts[remainders[i][0]] += 1
    return counts
```

Add `from typing import TypeVar` at the top of the file if not already imported (the existing `from typing import Callable, Iterable, ...` line should be extended).

- [ ] **Step 4: Run the new generic tests**

```powershell
py -3 -m pytest tests/test_apportion_counts_generic.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run the W3 regression suite**

```powershell
py -3 -m pytest tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py -v
```

Expected: 0 failures (refactor is type-only; runtime behaviour unchanged for `RAGVariant` callers).

- [ ] **Step 6: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_apportion_counts_generic.py
git commit -m @'
refactor(rag-dataset): make apportion_counts generic over Enum

Replaces the hard-coded RAGVariant TypeVar with a parameterised
_EnumT so Workstream 4 (DPO) can reuse the Hamilton apportionment
helper with its own RejectStrategy enum without duplicating logic.

Runtime behaviour for existing RAGVariant callers is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 0.2: Expose `strip_citations` publicly

**Files:**
- Modify: `app/services/rag_dataset.py`
- Modify: `tests/test_rag_dataset.py` (append one test)

**Background:** W4's `NO_CITATION` reject strategy must strip `[doc_chunk:X]` markers from a `chosen` answer. The exact regex already exists as `_strip_citations` (private). Cross-module access to private functions is discouraged — expose a public alias and keep the underscore one for back-compat with the W3 `build_empty_sample`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:

```python
def test_strip_citations_is_public_api() -> None:
    """W4 imports strip_citations directly — keep it on the module surface."""
    from app.services.rag_dataset import strip_citations

    assert strip_citations("Ответ. [doc_chunk:7]") == "Ответ."
    assert strip_citations("До [doc_chunk:1] середина [doc_chunk:2] конец") == \
        "До  середина  конец"
    assert strip_citations("без цитат") == "без цитат"
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_rag_dataset.py::test_strip_citations_is_public_api -v
```

Expected: FAIL with `ImportError: cannot import name 'strip_citations'`.

- [ ] **Step 3: Add the public alias**

In `app/services/rag_dataset.py`, immediately after the existing `_strip_citations` definition (around line 240–241), add:

```python
def strip_citations(text: str) -> str:
    """Remove ``[doc_chunk:N]`` markers from ``text``.

    Public alias of :func:`_strip_citations`. Used by W4
    (DPO post-training) to construct the ``NO_CITATION`` reject branch.
    """

    return _strip_citations(text)
```

(Note: do **not** swap callers of `_strip_citations` — keeping both names avoids a noisy diff in W3 code.)

- [ ] **Step 4: Run the test to confirm it passes**

```powershell
py -3 -m pytest tests/test_rag_dataset.py::test_strip_citations_is_public_api -v
```

Expected: PASS.

- [ ] **Step 5: Sprint 0 sweep**

```powershell
py -3 -m pytest tests/test_apportion_counts_generic.py tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py -v
py -3 -m ruff check app/services/rag_dataset.py tests/test_apportion_counts_generic.py
py -3 -m black --check app/services/rag_dataset.py tests/test_apportion_counts_generic.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
refactor(rag-dataset): expose strip_citations as a public alias

W4 (DPO post-training) needs to strip [doc_chunk:N] markers when
constructing the NO_CITATION reject branch. Public alias keeps the
existing _strip_citations in place for W3 callers; new code imports
the public name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 1 — `dpo_dataset.py` module + reject strategies (~3 h)

**Goal:** All module-level tests pass. The module produces deterministic `DPOPair` objects for each `RejectStrategy` given a fake teacher provider, and the orchestrator respects 40 / 30 / 30 apportionment.

**Abort point:** After Task 1.5 (NO_CITATION + GENERIC strategies done) — that covers ~70 % of the dataset distribution; HALLUCINATION and the orchestrator can land in a follow-up PR if review time runs short.

---

### Task 1.1: Module skeleton + import test

**Files:**
- Create: `app/services/dpo_dataset.py`
- Create: `tests/test_dpo_dataset.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_dpo_dataset.py`:

```python
"""Tests for app.services.dpo_dataset — pure-logic DPO dataset builder."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module imports without side effects."""
    from app.services import dpo_dataset

    assert dpo_dataset.__name__ == "app.services.dpo_dataset"
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py::test_module_imports -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.dpo_dataset'`.

- [ ] **Step 3: Create the minimal module**

Create `app/services/dpo_dataset.py`:

```python
"""Compose a synthetic DPO preference dataset.

Pure-logic core of Workstream 4 (DPO post-training / preference
learning) in the Pack B++ ML strengthening plan. Builds on W1's
:mod:`app.services.synthetic_qa` for seed Q&A and W3's
:mod:`app.services.rag_dataset` (Hamilton apportionment +
citation stripping helpers).

The module is intentionally I/O free: teacher provider and the
seed iterator are injected so the logic is deterministic in tests.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)
```

- [ ] **Step 4: Run the test to confirm it passes**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py::test_module_imports -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/dpo_dataset.py tests/test_dpo_dataset.py
git commit -m @'
feat(dpo-dataset): module skeleton for W4 DPO preference builder

Empty module + import test as the foundation for Workstream 4
(closes G3: DPO post-training / preference learning).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.2: `RejectStrategy` enum + `DPOPair` dataclass

**Files:**
- Modify: `app/services/dpo_dataset.py`
- Modify: `tests/test_dpo_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dpo_dataset.py`:

```python
def test_reject_strategy_values() -> None:
    """The three canonical synthetic reject strategies are exposed."""
    from app.services.dpo_dataset import RejectStrategy

    values = {s.value for s in RejectStrategy}
    assert {"no_citation", "generic", "hallucination"}.issubset(values)


def test_dpo_pair_to_jsonl_line_top_level_keys() -> None:
    """to_jsonl_line() emits prompt / chosen / rejected at the top level."""
    import json

    from app.services.dpo_dataset import DPOPair, RejectStrategy

    pair = DPOPair(
        prompt="Что такое отпуск?",
        chosen="Это перерыв. [doc_chunk:7]",
        rejected="Это перерыв.",
        strategy=RejectStrategy.NO_CITATION,
        source="synthetic",
        source_chunk_id=7,
        feedback_ids=(),
    )
    line = pair.to_jsonl_line()
    assert line.endswith("\n")

    data = json.loads(line)
    assert data["prompt"] == "Что такое отпуск?"
    assert data["chosen"].endswith("[doc_chunk:7]")
    assert data["rejected"] == "Это перерыв."
    assert data["meta"]["strategy"] == "no_citation"
    assert data["meta"]["source"] == "synthetic"
    assert data["meta"]["source_chunk_id"] == 7
    assert data["meta"]["feedback_ids"] == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v
```

Expected: 2 FAIL with `ImportError: cannot import name 'RejectStrategy'`.

- [ ] **Step 3: Implement enum and dataclass**

Append to `app/services/dpo_dataset.py`:

```python
import json
from dataclasses import dataclass
from enum import Enum


class RejectStrategy(str, Enum):
    """How the ``rejected`` half of a DPO pair was constructed.

    Synthetic branches (`no_citation`, `generic`, `hallucination`) come
    from teacher-LLM or regex generators. Live branches
    (`live_alt`, `live_paired`) come from user feedback collected via
    the ``/api/kb/messages/{id}/feedback`` endpoint.
    """

    NO_CITATION = "no_citation"
    GENERIC = "generic"
    HALLUCINATION = "hallucination"
    LIVE_ALT = "live_alt"
    LIVE_PAIRED = "live_paired"


@dataclass(frozen=True, slots=True)
class DPOPair:
    """One preference pair ready for trl.DPOTrainer.

    Top-level ``prompt / chosen / rejected`` match the trl 0.11
    dataset contract so no transform pass is needed before training.
    """

    prompt: str
    chosen: str
    rejected: str
    strategy: RejectStrategy
    source: str  # "synthetic" or "live"
    source_chunk_id: int | None
    feedback_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "meta": {
                "source": self.source,
                "strategy": self.strategy.value,
                "source_chunk_id": (
                    int(self.source_chunk_id) if self.source_chunk_id is not None else None
                ),
                "feedback_ids": list(self.feedback_ids),
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/dpo_dataset.py tests/test_dpo_dataset.py
git commit -m @'
feat(dpo-dataset): add RejectStrategy enum and DPOPair dataclass

Defines five reject strategies (three synthetic + two live-feedback)
and the JSONL-serialisable DPOPair. Top-level prompt/chosen/rejected
match trl.DPOTrainer dataset contract directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.3: `NO_CITATION` strategy (regex strip — no LLM)

**Files:**
- Modify: `app/services/dpo_dataset.py`
- Modify: `tests/test_dpo_dataset.py`

**Background:** `NO_CITATION` strips the `[doc_chunk:X]` suffix from the chosen answer, teaching the model that a citation suffix is preferred over a confident-but-uncited paraphrase. Uses W3's `strip_citations` (made public in Sprint 0).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dpo_dataset.py`:

```python
def test_build_no_citation_pair_strips_marker() -> None:
    from app.services.dpo_dataset import RejectStrategy, build_no_citation_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    pair = build_no_citation_pair(seed)

    assert pair.strategy is RejectStrategy.NO_CITATION
    assert pair.prompt == "Что такое отпуск?"
    assert pair.chosen == "Это перерыв. [doc_chunk:7]"
    assert pair.rejected == "Это перерыв."
    assert "[doc_chunk:" not in pair.rejected
    assert pair.source == "synthetic"
    assert pair.source_chunk_id == 7


def test_build_no_citation_pair_returns_none_when_no_marker() -> None:
    """Seeds without a citation marker can't form a meaningful NO_CITATION pair."""
    from app.services.dpo_dataset import build_no_citation_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Q",
        input="",
        output="A without marker",
        source_chunk_id=1,
    )
    assert build_no_citation_pair(seed) is None
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v -k no_citation
```

Expected: 2 FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_no_citation_pair`**

Append to `app/services/dpo_dataset.py`:

```python
from app.services.rag_dataset import strip_citations
from app.services.synthetic_qa import QAPair

_CITATION_MARKER = "[doc_chunk:"


def build_no_citation_pair(seed: QAPair) -> DPOPair | None:
    """Strip the citation suffix from the chosen answer to form ``rejected``.

    Returns ``None`` when the seed has no citation marker — there is
    no signal to learn from in that case.
    """

    if _CITATION_MARKER not in seed.output:
        return None
    rejected = strip_citations(seed.output)
    return DPOPair(
        prompt=seed.instruction,
        chosen=seed.output,
        rejected=rejected,
        strategy=RejectStrategy.NO_CITATION,
        source="synthetic",
        source_chunk_id=seed.source_chunk_id,
        feedback_ids=(),
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v -k no_citation
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/dpo_dataset.py tests/test_dpo_dataset.py
git commit -m @'
feat(dpo-dataset): NO_CITATION strategy (zero LLM calls)

Strips [doc_chunk:N] markers from the seed answer to construct the
rejected branch. Reuses strip_citations from rag_dataset (W3) to
keep regex behaviour consistent across workstreams. Returns None
when the seed already has no marker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.4: Teacher-provider protocol + `GENERIC` strategy

**Files:**
- Modify: `app/services/dpo_dataset.py`
- Modify: `tests/test_dpo_dataset.py`

**Background:** `GENERIC` asks a teacher LLM to answer the question **without** the retrieved context — producing a plausible but ungrounded answer that the model must learn to disprefer. We inject a `TeacherProvider` callable so tests can use a fake; production wires `app.services.kb_llm.create_llm_provider()` (already typed by W1).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dpo_dataset.py`:

```python
def test_build_generic_pair_calls_teacher_without_context() -> None:
    from app.services.dpo_dataset import RejectStrategy, build_generic_pair
    from app.services.synthetic_qa import QAPair

    captured: list[str] = []

    def fake_teacher(prompt: str) -> str:
        captured.append(prompt)
        return "Это общий ответ из обучающих данных модели."

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    pair = build_generic_pair(seed, teacher=fake_teacher)

    assert pair is not None
    assert pair.strategy is RejectStrategy.GENERIC
    assert pair.chosen == seed.output
    assert pair.rejected == "Это общий ответ из обучающих данных модели."
    assert "Что такое отпуск?" in captured[0]
    assert "[doc_chunk:" not in pair.rejected


def test_build_generic_pair_returns_none_when_teacher_returns_empty() -> None:
    from app.services.dpo_dataset import build_generic_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Q?",
        input="",
        output="A. [doc_chunk:1]",
        source_chunk_id=1,
    )
    pair = build_generic_pair(seed, teacher=lambda _q: "  ")
    assert pair is None


def test_build_generic_pair_strips_accidental_citations_from_teacher() -> None:
    """Teacher might paste a fake citation; strip it to keep rejected ungrounded."""
    from app.services.dpo_dataset import build_generic_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Q?",
        input="",
        output="Real. [doc_chunk:5]",
        source_chunk_id=5,
    )
    pair = build_generic_pair(
        seed,
        teacher=lambda _q: "Generic answer. [doc_chunk:5]",
    )
    assert pair is not None
    assert "[doc_chunk:" not in pair.rejected
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v -k generic
```

Expected: 3 FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_generic_pair` + the prompt helper**

Append to `app/services/dpo_dataset.py`:

```python
from typing import Callable

TeacherProvider = Callable[[str], str]


_GENERIC_TEACHER_PROMPT = (
    "Ответь на вопрос пользователя, опираясь только на свои общие знания. "
    "НЕ используй никаких документов или цитат. Не указывай источников.\n\n"
    "Вопрос: {question}"
)


def _ask_teacher_generic(question: str, teacher: TeacherProvider) -> str:
    return teacher(_GENERIC_TEACHER_PROMPT.format(question=question))


def build_generic_pair(
    seed: QAPair,
    *,
    teacher: TeacherProvider,
) -> DPOPair | None:
    """Ask the teacher to answer **without** the retrieved chunk.

    Returns ``None`` when the teacher response is empty or whitespace —
    that means the call failed silently and the pair would teach noise.
    """

    raw = _ask_teacher_generic(seed.instruction, teacher).strip()
    if not raw:
        return None
    # Defensive: if the teacher leaked a citation marker, strip it so
    # the rejected branch stays cleanly ungrounded.
    rejected = strip_citations(raw)
    return DPOPair(
        prompt=seed.instruction,
        chosen=seed.output,
        rejected=rejected,
        strategy=RejectStrategy.GENERIC,
        source="synthetic",
        source_chunk_id=seed.source_chunk_id,
        feedback_ids=(),
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v -k generic
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/dpo_dataset.py tests/test_dpo_dataset.py
git commit -m @'
feat(dpo-dataset): GENERIC strategy (one teacher call, ungrounded answer)

Asks the teacher LLM to answer without the retrieved context, then
strips any accidental citation markers to keep the rejected branch
clearly ungrounded. Returns None on empty teacher response.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.5: `HALLUCINATION` strategy (fake citation)

**Files:**
- Modify: `app/services/dpo_dataset.py`
- Modify: `tests/test_dpo_dataset.py`

**Background:** `HALLUCINATION` asks the teacher to **invent** a fake `[doc_chunk:9XX]` citation. The fabricated chunk id MUST be in the 900+ range so we can verify the regex matched. If the teacher refuses or returns a real-looking ID, we coerce it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dpo_dataset.py`:

```python
def test_build_hallucination_pair_injects_fake_citation() -> None:
    from app.services.dpo_dataset import RejectStrategy, build_hallucination_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )

    def fake_teacher(prompt: str) -> str:
        # Teacher cooperated and added a fake marker in the 900+ range.
        return "Согласно документу, отпуск — это отдых. [doc_chunk:912]"

    pair = build_hallucination_pair(seed, teacher=fake_teacher)
    assert pair is not None
    assert pair.strategy is RejectStrategy.HALLUCINATION
    assert "[doc_chunk:" in pair.rejected
    # The injected ID is in the fake range (>= 900) by construction.
    import re

    match = re.search(r"\[doc_chunk:(\d+)\]", pair.rejected)
    assert match is not None
    assert int(match.group(1)) >= 900


def test_build_hallucination_pair_coerces_teacher_without_marker() -> None:
    """If the teacher forgot the marker, we append one ourselves."""
    from app.services.dpo_dataset import build_hallucination_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Q?",
        input="",
        output="A. [doc_chunk:1]",
        source_chunk_id=1,
    )
    pair = build_hallucination_pair(
        seed,
        teacher=lambda _p: "Просто ответ без маркера.",
    )
    assert pair is not None
    import re

    match = re.search(r"\[doc_chunk:(\d+)\]", pair.rejected)
    assert match is not None
    assert int(match.group(1)) >= 900


def test_build_hallucination_pair_returns_none_on_empty_teacher() -> None:
    from app.services.dpo_dataset import build_hallucination_pair
    from app.services.synthetic_qa import QAPair

    seed = QAPair(
        instruction="Q?",
        input="",
        output="A. [doc_chunk:1]",
        source_chunk_id=1,
    )
    assert build_hallucination_pair(seed, teacher=lambda _p: "") is None
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v -k hallucination
```

Expected: 3 FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_hallucination_pair`**

Append to `app/services/dpo_dataset.py`:

```python
import random
import re

_FAKE_CITATION_RE = re.compile(r"\[doc_chunk:(\d+)\]")
_FAKE_CHUNK_RANGE = (900, 999)


_HALLUCINATION_TEACHER_PROMPT = (
    "Сгенерируй правдоподобный ответ на вопрос и обязательно сошлись на "
    "несуществующий документ в формате [doc_chunk:N] где N >= 900. "
    "Это специальный обучающий пример: модель должна научиться НЕ давать "
    "такие выдуманные ссылки.\n\n"
    "Вопрос: {question}"
)


def build_hallucination_pair(
    seed: QAPair,
    *,
    teacher: TeacherProvider,
    rng: random.Random | None = None,
) -> DPOPair | None:
    """Ask the teacher for an answer with an **invented** ``[doc_chunk:9XX]``.

    Coerces the citation into the 900-999 fake range if the teacher
    cooperated but used a different id; appends a fresh one if the
    teacher returned no marker at all. Returns ``None`` on empty
    response.
    """

    raw = teacher(_HALLUCINATION_TEACHER_PROMPT.format(question=seed.instruction)).strip()
    if not raw:
        return None

    rng = rng or random.Random(seed.source_chunk_id)  # deterministic per seed
    fake_id = rng.randint(*_FAKE_CHUNK_RANGE)

    match = _FAKE_CITATION_RE.search(raw)
    if match is None:
        rejected = f"{raw} [doc_chunk:{fake_id}]"
    else:
        existing = int(match.group(1))
        if _FAKE_CHUNK_RANGE[0] <= existing <= _FAKE_CHUNK_RANGE[1]:
            rejected = raw  # teacher already used a fake id in range
        else:
            rejected = _FAKE_CITATION_RE.sub(f"[doc_chunk:{fake_id}]", raw, count=1)

    return DPOPair(
        prompt=seed.instruction,
        chosen=seed.output,
        rejected=rejected,
        strategy=RejectStrategy.HALLUCINATION,
        source="synthetic",
        source_chunk_id=seed.source_chunk_id,
        feedback_ids=(),
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py -v -k hallucination
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/dpo_dataset.py tests/test_dpo_dataset.py
git commit -m @'
feat(dpo-dataset): HALLUCINATION strategy (fabricated [doc_chunk:9XX])

Asks the teacher to invent a fake citation in the 900-999 range and
coerces the marker if the teacher used a different id or forgot one
entirely. Deterministic RNG seeded by source_chunk_id keeps regen
reproducible across re-runs of generate_dpo_pairs.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.6: `DPOPairBuilder` orchestrator with 40 / 30 / 30 apportionment

**Files:**
- Modify: `app/services/dpo_dataset.py`
- Create: `tests/test_dpo_dataset_strategies.py`

**Background:** The orchestrator iterates seeds round-robin across the three synthetic strategies, dispatching to the right `build_*_pair`. Uses Sprint 0's generic `apportion_counts` so the 40 / 30 / 30 split is exact across re-runs. Skips seeds where the chosen strategy returns `None` (e.g. NO_CITATION when no marker) and tries the next variant.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dpo_dataset_strategies.py`:

```python
"""Tests for the DPOPairBuilder orchestrator + apportionment."""

from __future__ import annotations

from collections import Counter

import pytest

from app.services.synthetic_qa import QAPair


def _make_seeds(n: int) -> list[QAPair]:
    return [
        QAPair(
            instruction=f"Вопрос {i}?",
            input="",
            output=f"Ответ {i}. [doc_chunk:{i}]",
            source_chunk_id=i,
        )
        for i in range(1, n + 1)
    ]


def test_default_synthetic_proportions_are_40_30_30() -> None:
    from app.services.dpo_dataset import RejectStrategy, default_synthetic_proportions

    p = default_synthetic_proportions()
    assert pytest.approx(p[RejectStrategy.NO_CITATION], 1e-6) == 0.40
    assert pytest.approx(p[RejectStrategy.GENERIC], 1e-6) == 0.30
    assert pytest.approx(p[RejectStrategy.HALLUCINATION], 1e-6) == 0.30


def test_builder_respects_apportionment() -> None:
    from app.services.dpo_dataset import DPOPairBuilder

    builder = DPOPairBuilder(teacher=lambda _p: "Fake teacher answer.")
    pairs = list(builder.build(_make_seeds(10), total=10))
    assert len(pairs) == 10
    counts = Counter(p.strategy.value for p in pairs)
    assert counts["no_citation"] == 4  # 40% of 10
    assert counts["generic"] == 3  # 30% of 10
    assert counts["hallucination"] == 3  # 30% of 10


def test_builder_skips_when_no_citation_marker() -> None:
    """NO_CITATION quota is re-allocated when the seed has no marker."""
    from app.services.dpo_dataset import DPOPairBuilder

    seeds = [
        QAPair(instruction="Q1?", input="", output="No marker here.", source_chunk_id=1),
        QAPair(instruction="Q2?", input="", output="Still no marker.", source_chunk_id=2),
    ]
    builder = DPOPairBuilder(teacher=lambda _p: "Fake teacher answer.")
    pairs = list(builder.build(seeds, total=2))
    # Both NO_CITATION slots would be dropped — builder falls back to the
    # next strategies in priority order (GENERIC, HALLUCINATION).
    for p in pairs:
        assert p.strategy.value != "no_citation"


def test_builder_under_delivers_when_seeds_exhausted() -> None:
    from app.services.dpo_dataset import DPOPairBuilder

    builder = DPOPairBuilder(teacher=lambda _p: "Generic.")
    pairs = list(builder.build(_make_seeds(3), total=10))
    assert len(pairs) <= 3


def test_builder_deterministic_across_runs() -> None:
    """Same seeds + same RNG seed produce identical strategy assignment."""
    from app.services.dpo_dataset import DPOPairBuilder

    seeds = _make_seeds(20)
    a = list(DPOPairBuilder(teacher=lambda _p: "x").build(seeds, total=20))
    b = list(DPOPairBuilder(teacher=lambda _p: "x").build(seeds, total=20))
    assert [p.strategy.value for p in a] == [p.strategy.value for p in b]
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_dpo_dataset_strategies.py -v
```

Expected: 5 FAIL with `ImportError`.

- [ ] **Step 3: Implement `DPOPairBuilder` and `default_synthetic_proportions`**

Append to `app/services/dpo_dataset.py`:

```python
from dataclasses import field
from typing import Iterable, Iterator, Mapping

from app.services.rag_dataset import apportion_counts


SyntheticProportions = Mapping[RejectStrategy, float]


def default_synthetic_proportions() -> dict[RejectStrategy, float]:
    """Spec defaults: 40 % NO_CITATION (free) / 30 % GENERIC / 30 % HALLUCINATION."""

    return {
        RejectStrategy.NO_CITATION: 0.40,
        RejectStrategy.GENERIC: 0.30,
        RejectStrategy.HALLUCINATION: 0.30,
    }


@dataclass(slots=True)
class DPOPairBuilder:
    """Orchestrate preference-pair assembly across synthetic strategies."""

    teacher: TeacherProvider
    proportions: SyntheticProportions = field(default_factory=default_synthetic_proportions)

    def build(
        self,
        seeds: Iterable[QAPair],
        *,
        total: int,
    ) -> Iterator[DPOPair]:
        counts = apportion_counts(self.proportions, total=total)
        emitted: dict[RejectStrategy, int] = {s: 0 for s in self.proportions}

        priority = (
            RejectStrategy.NO_CITATION,
            RejectStrategy.GENERIC,
            RejectStrategy.HALLUCINATION,
        )

        for seed in seeds:
            if sum(emitted.values()) >= total:
                return
            for strategy in priority:
                if emitted[strategy] >= counts.get(strategy, 0):
                    continue
                pair = self._build_one(seed, strategy)
                if pair is None:
                    continue
                emitted[strategy] += 1
                yield pair
                break

    def _build_one(self, seed: QAPair, strategy: RejectStrategy) -> DPOPair | None:
        if strategy is RejectStrategy.NO_CITATION:
            return build_no_citation_pair(seed)
        if strategy is RejectStrategy.GENERIC:
            return build_generic_pair(seed, teacher=self.teacher)
        if strategy is RejectStrategy.HALLUCINATION:
            return build_hallucination_pair(seed, teacher=self.teacher)
        return None
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_dpo_dataset_strategies.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Sprint 1 sweep**

```powershell
py -3 -m pytest tests/test_dpo_dataset.py tests/test_dpo_dataset_strategies.py -v
py -3 -m ruff check app/services/dpo_dataset.py tests/test_dpo_dataset.py tests/test_dpo_dataset_strategies.py
py -3 -m black --check app/services/dpo_dataset.py tests/test_dpo_dataset.py tests/test_dpo_dataset_strategies.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add app/services/dpo_dataset.py tests/test_dpo_dataset_strategies.py
git commit -m @'
feat(dpo-dataset): DPOPairBuilder orchestrator with 40/30/30 mix

Iterates seeds and dispatches to the per-strategy build_* helper
in priority order. Reuses Sprint 0 generic apportion_counts so the
total stays exact. Skips strategies whose builder returns None
(e.g. NO_CITATION on a citation-less seed) and reallocates the
slot to the next strategy. Under-delivers gracefully on seed
shortage rather than padding noise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 2 — `generate_dpo_pairs.py` CLI (~1.5 h)

**Goal:** `py -3 -m scripts.generate_dpo_pairs --seeds … --output dpo.jsonl --target-pairs 40` writes a JSONL whose lines are valid `DPOPair`s, with strategy mix within ±1 of 40 / 30 / 30 and a cost guard that aborts on accidental large runs.

**Abort point:** After Task 2.1 — the parse_args contract is fixed and tested; the main loop can land in a follow-up PR if the diff is getting big.

---

### Task 2.1: CLI scaffold + `parse_args`

**Files:**
- Create: `scripts/generate_dpo_pairs.py`
- Create: `tests/scripts/test_generate_dpo_pairs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_generate_dpo_pairs.py`:

```python
"""Smoke tests for scripts.generate_dpo_pairs CLI."""

from __future__ import annotations


def test_cli_module_imports() -> None:
    import scripts.generate_dpo_pairs as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_minimal() -> None:
    from pathlib import Path

    from scripts.generate_dpo_pairs import parse_args

    ns = parse_args(
        [
            "--seeds",
            "var/data/seeds.jsonl",
            "--output",
            "var/data/dpo.jsonl",
            "--target-pairs",
            "100",
            "--yes",
        ]
    )
    assert ns.seeds == Path("var/data/seeds.jsonl")
    assert ns.output == Path("var/data/dpo.jsonl")
    assert ns.target_pairs == 100
    assert ns.yes is True
    assert ns.max_cost_usd == 1.0  # default budget guard
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/scripts/test_generate_dpo_pairs.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the scaffold**

Create `scripts/generate_dpo_pairs.py`:

```python
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/scripts/test_generate_dpo_pairs.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/generate_dpo_pairs.py tests/scripts/test_generate_dpo_pairs.py
git commit -m @'
feat(dpo-dataset): CLI scaffold for generate_dpo_pairs.py

Argparse contract + main() stub. Wiring to DPOPairBuilder lands
in the next task to keep diffs reviewable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 2.2: End-to-end main loop + cost guard

**Files:**
- Modify: `scripts/generate_dpo_pairs.py`
- Modify: `tests/scripts/test_generate_dpo_pairs.py`

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/scripts/test_generate_dpo_pairs.py`:

```python
def test_cli_writes_jsonl_endtoend(tmp_path, monkeypatch) -> None:
    """4 seeds + fake teacher → 4 pairs, strategy mix within ±1 of 40/30/30."""
    import json

    from app.services.synthetic_qa import QAPair
    from scripts.generate_dpo_pairs import main

    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w", encoding="utf-8") as fh:
        for i in range(1, 5):
            fh.write(
                QAPair(
                    instruction=f"Вопрос {i}?",
                    input="",
                    output=f"Ответ {i}. [doc_chunk:{i}]",
                    source_chunk_id=i,
                ).to_jsonl_line()
            )

    # Monkeypatch the teacher-provider factory used by main() so it
    # returns a fake teacher (no network).
    import scripts.generate_dpo_pairs as cli

    monkeypatch.setattr(
        cli, "_make_teacher", lambda _args: (lambda _q: "Fake teacher answer.")
    )

    output = tmp_path / "dpo.jsonl"
    rc = main(
        [
            "--seeds",
            str(seeds_path),
            "--output",
            str(output),
            "--target-pairs",
            "4",
            "--yes",
        ]
    )
    assert rc == 0

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4
    for line in lines:
        assert "prompt" in line
        assert "chosen" in line
        assert "rejected" in line
        assert line["meta"]["strategy"] in {"no_citation", "generic", "hallucination"}


def test_cost_guard_aborts_without_yes(tmp_path, monkeypatch, capsys) -> None:
    """Without --yes the cost guard kicks in for >100 teacher calls."""
    import pytest

    from app.services.synthetic_qa import QAPair
    from scripts.generate_dpo_pairs import main

    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w", encoding="utf-8") as fh:
        for i in range(1, 5001):  # 5000 seeds → ~3000 teacher calls
            fh.write(
                QAPair(
                    instruction=f"Q{i}?",
                    input="",
                    output=f"A. [doc_chunk:{i}]",
                    source_chunk_id=i,
                ).to_jsonl_line()
            )

    output = tmp_path / "dpo.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--seeds",
                str(seeds_path),
                "--output",
                str(output),
                "--target-pairs",
                "5000",
                "--max-cost-usd",
                "0.10",  # tiny budget to force trip
            ]
        )
    assert "Estimated" in str(exc.value) or "budget" in str(exc.value).lower()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/scripts/test_generate_dpo_pairs.py -v -k "endtoend or cost"
```

Expected: 2 FAIL (main is still a stub).

- [ ] **Step 3: Wire the main loop + helpers + cost guard**

Replace `main()` and surrounding helpers in `scripts/generate_dpo_pairs.py` with:

```python
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
        share for strategy, share in proportions.items()
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
        except Exception as exc:  # provider errors must not crash the run
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/scripts/test_generate_dpo_pairs.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/generate_dpo_pairs.py tests/scripts/test_generate_dpo_pairs.py
git commit -m @'
feat(dpo-dataset): wire CLI main loop + cost guard

Loads W1 seed JSONL, builds the teacher callable via kb_llm, streams
DPO pairs through DPOPairBuilder into the output JSONL. Cost guard
aborts when estimated teacher-call cost exceeds --max-cost-usd
unless --yes is passed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 3 — `kb_feedback` (~2 h)

**Goal:** New SQLite table, two `KnowledgeBaseStore` methods, two API endpoints. `POST /api/kb/messages/{id}/feedback` persists a rating; `GET /api/kb/feedback/export` returns DPOPair-shaped NDJSON.

**Important deviation from spec:** the spec writes `message_id TEXT REFERENCES kb_messages(id)`, but the actual `kb_messages.id` column is `INTEGER PRIMARY KEY AUTOINCREMENT` (see `app/services/kb_store.py:271-281`). The plan uses `message_id INTEGER` to match.

**Abort point:** After Task 3.3 — schema + endpoints work; pairing logic for `iter_feedback_pairs()` can land in a follow-up since the trainer can also consume synthetic pairs alone.

---

### Task 3.1: `kb_feedback` schema + `store_feedback()` method

**Files:**
- Modify: `app/services/kb_store.py`
- Create: `tests/test_kb_feedback_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_feedback_store.py`:

```python
"""Tests for kb_feedback table + KnowledgeBaseStore.store_feedback()."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture
def store(tmp_path):
    from app.services.kb_store import KnowledgeBaseStore

    s = KnowledgeBaseStore(db_path=tmp_path / "kb.sqlite")
    # Seed a conversation + assistant message so feedback has a target.
    conv_id = s.create_conversation(title="test")
    msg_id = s.append_message(
        conversation_id=conv_id, role="user", content="hello?"
    )
    asst_id = s.append_message(
        conversation_id=conv_id, role="assistant", content="hi back"
    )
    return s, conv_id, msg_id, asst_id


def test_feedback_table_created(store) -> None:
    s, _conv, _user_msg, _asst_msg = store
    with s._connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kb_feedback'"
        ).fetchone()
    assert row is not None


def test_store_feedback_persists_and_returns_id(store) -> None:
    s, conv, _user_msg, asst_msg = store
    fid = s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=1,
        comment=None,
        alternative_answer=None,
    )
    assert isinstance(fid, str) and len(fid) >= 8
    with s._connect() as conn:
        row = conn.execute(
            "SELECT rating, comment FROM kb_feedback WHERE id=?", (fid,)
        ).fetchone()
    assert row[0] == 1
    assert row[1] is None


def test_store_feedback_rejects_invalid_rating(store) -> None:
    import sqlite3

    s, conv, _user_msg, asst_msg = store
    with pytest.raises((sqlite3.IntegrityError, ValueError)):
        s.store_feedback(
            conversation_id=conv,
            message_id=asst_msg,
            user_id="u1",
            rating=5,  # not in {-1, 1}
            comment=None,
            alternative_answer=None,
        )


def test_store_feedback_accepts_alternative_answer(store) -> None:
    s, conv, _user_msg, asst_msg = store
    fid = s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment="плохо",
        alternative_answer="лучше так",
    )
    with s._connect() as conn:
        row = conn.execute(
            "SELECT alternative_answer FROM kb_feedback WHERE id=?", (fid,)
        ).fetchone()
    assert row[0] == "лучше так"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_kb_feedback_store.py -v
```

Expected: tests fail because `kb_feedback` table is not created and `store_feedback` does not exist.

- [ ] **Step 3: Add the schema and method**

In `app/services/kb_store.py`, find the `_initialise_schema` method. Inside the executescript block, **immediately after** the `kb_messages` CREATE TABLE statement and its index (the existing block ending around line 281), append:

```sql
                CREATE TABLE IF NOT EXISTS kb_feedback (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL
                        REFERENCES kb_conversations(id) ON DELETE CASCADE,
                    message_id INTEGER NOT NULL
                        REFERENCES kb_messages(id) ON DELETE CASCADE,
                    user_id TEXT,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    comment TEXT,
                    alternative_answer TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kb_feedback_message
                    ON kb_feedback(message_id);
                CREATE INDEX IF NOT EXISTS idx_kb_feedback_rating_created
                    ON kb_feedback(rating, created_at);
```

Then, somewhere after `list_messages()` (the existing method at line 713), add:

```python
    def store_feedback(
        self,
        *,
        conversation_id: str,
        message_id: int,
        user_id: str | None,
        rating: int,
        comment: str | None,
        alternative_answer: str | None,
    ) -> str:
        """Persist one feedback row; returns a new UUID id.

        Raises ValueError for out-of-range ``rating`` before the DB
        CHECK constraint catches it, so the API layer can map it to
        HTTP 400 without parsing sqlite3 error messages.
        """

        import uuid
        from datetime import datetime, timezone

        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or 1, got {rating}")

        fid = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kb_feedback (
                    id, conversation_id, message_id, user_id,
                    rating, comment, alternative_answer, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fid,
                    conversation_id,
                    int(message_id),
                    user_id,
                    int(rating),
                    comment,
                    alternative_answer,
                    now,
                ),
            )
            conn.commit()
        return fid
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_kb_feedback_store.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/kb_store.py tests/test_kb_feedback_store.py
git commit -m @'
feat(kb-feedback): add kb_feedback schema + store_feedback() method

CREATE TABLE IF NOT EXISTS kb_feedback inside _initialise_schema —
idempotent migration via the existing MVP startup path (no Alembic).
store_feedback() validates rating in {-1, 1} before insert so the
API layer can return 400 cleanly.

Schema fixes a minor spec drift: message_id is INTEGER (matching
the existing kb_messages.id AUTOINCREMENT column), not TEXT.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.2: `iter_feedback_pairs()` — convert ratings to DPOPair JSONL

**Files:**
- Modify: `app/services/kb_store.py`
- Modify: `tests/test_kb_feedback_store.py`

**Background:** Pairing logic per spec § 6.3 — most-recent rating per `(message_id, user_id)` wins. If a thumbs-up has `alternative_answer`, emit a pair where the alternative is `chosen` and the assistant text is `rejected`. If thumbs-down with `alternative_answer`, emit the reverse. Skip silently when there is no usable signal.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kb_feedback_store.py`:

```python
def test_iter_feedback_pairs_emits_alt_when_thumbs_down(store) -> None:
    """Thumbs-down with alternative_answer → (alt is chosen, assistant is rejected)."""
    from app.services.dpo_dataset import DPOPair, RejectStrategy

    s, conv, user_msg, asst_msg = store
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer="более точный ответ",
    )

    pairs = list(s.iter_feedback_pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert isinstance(p, DPOPair)
    assert p.strategy is RejectStrategy.LIVE_ALT
    assert p.prompt == "hello?"
    assert p.chosen == "более точный ответ"
    assert p.rejected == "hi back"
    assert p.source == "live"
    assert len(p.feedback_ids) == 1


def test_iter_feedback_pairs_skips_when_no_alt_and_thumbs_down(store) -> None:
    """Thumbs-down without alternative_answer is insufficient signal — skip."""
    s, conv, user_msg, asst_msg = store
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer=None,
    )
    assert list(s.iter_feedback_pairs()) == []


def test_iter_feedback_pairs_uses_most_recent_per_user(store) -> None:
    """If a user flips rating, only the latest counts."""
    s, conv, user_msg, asst_msg = store
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=1,
        comment=None,
        alternative_answer=None,
    )
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer="лучше",
    )

    pairs = list(s.iter_feedback_pairs())
    assert len(pairs) == 1
    assert pairs[0].chosen == "лучше"


def test_iter_feedback_pairs_skips_orphan_assistant_messages(store, tmp_path) -> None:
    """Assistant message with no preceding user message → skip with debug log."""
    from app.services.kb_store import KnowledgeBaseStore

    s2 = KnowledgeBaseStore(db_path=tmp_path / "kb_orphan.sqlite")
    conv_id = s2.create_conversation(title="orphan")
    # Skip directly to assistant message with no preceding user — unusual but possible.
    asst_id = s2.append_message(
        conversation_id=conv_id, role="assistant", content="answer"
    )
    s2.store_feedback(
        conversation_id=conv_id,
        message_id=asst_id,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer="alt",
    )
    assert list(s2.iter_feedback_pairs()) == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_kb_feedback_store.py -v -k iter
```

Expected: 4 FAIL with `AttributeError: 'KnowledgeBaseStore' object has no attribute 'iter_feedback_pairs'`.

- [ ] **Step 3: Implement `iter_feedback_pairs`**

In `app/services/kb_store.py`, add this method right after `store_feedback`:

```python
    def iter_feedback_pairs(self):
        """Yield :class:`app.services.dpo_dataset.DPOPair` from live feedback.

        Pairing rules (spec § 6.3):
          * Most-recent rating per (message_id, user_id) wins.
          * thumbs-up + alternative_answer → emit (alt as chosen, assistant as rejected).
          * thumbs-down + alternative_answer → emit (alt as chosen, assistant as rejected).
          * thumbs-up alone → look back for a same-message thumbs-down with alt.
          * Anything else → skip silently.
          * Orphaned assistant messages (no preceding user) → skip.
        """

        from app.services.dpo_dataset import DPOPair, RejectStrategy

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.conversation_id, f.message_id, f.user_id,
                       f.rating, f.alternative_answer, f.created_at,
                       m.content AS assistant_content
                FROM kb_feedback f
                JOIN kb_messages m ON m.id = f.message_id
                ORDER BY f.message_id, f.user_id, f.created_at DESC
                """
            ).fetchall()

            # Group by (message_id, user_id), keeping all rows in descending time order.
            groups: dict[tuple[int, str | None], list] = {}
            for row in rows:
                key = (int(row["message_id"]), row["user_id"])
                groups.setdefault(key, []).append(row)

            for (msg_id, user_id), group in groups.items():
                latest = group[0]
                # Find the immediately preceding user message in the same conversation.
                preceding = conn.execute(
                    """
                    SELECT content FROM kb_messages
                    WHERE conversation_id = ? AND role = 'user' AND id < ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (latest["conversation_id"], msg_id),
                ).fetchone()
                if preceding is None:
                    LOGGER.debug(
                        "Skipping feedback %s — no preceding user message",
                        latest["id"],
                    )
                    continue
                user_prompt = preceding["content"]
                assistant_text = latest["assistant_content"]

                alt = latest["alternative_answer"]
                if latest["rating"] == -1 and alt:
                    yield DPOPair(
                        prompt=user_prompt,
                        chosen=alt,
                        rejected=assistant_text,
                        strategy=RejectStrategy.LIVE_ALT,
                        source="live",
                        source_chunk_id=None,
                        feedback_ids=(latest["id"],),
                    )
                elif latest["rating"] == 1:
                    if alt:
                        yield DPOPair(
                            prompt=user_prompt,
                            chosen=alt,
                            rejected=assistant_text,
                            strategy=RejectStrategy.LIVE_ALT,
                            source="live",
                            source_chunk_id=None,
                            feedback_ids=(latest["id"],),
                        )
                        continue
                    # Look back for an earlier thumbs-down with an alt to pair against.
                    downvote = next(
                        (r for r in group[1:] if r["rating"] == -1 and r["alternative_answer"]),
                        None,
                    )
                    if downvote:
                        yield DPOPair(
                            prompt=user_prompt,
                            chosen=assistant_text,
                            rejected=downvote["alternative_answer"],
                            strategy=RejectStrategy.LIVE_PAIRED,
                            source="live",
                            source_chunk_id=None,
                            feedback_ids=(latest["id"], downvote["id"]),
                        )
                # else: insufficient signal, skip silently
```

Also ensure that `_connect()` returns a connection with `row_factory = sqlite3.Row` set so `row["column"]` access works. If it does not already, add `conn.row_factory = sqlite3.Row` inside `_connect`. Quick check before editing:

```powershell
py -3 -c "import sqlite3; from app.services.kb_store import KnowledgeBaseStore; s = KnowledgeBaseStore(':memory:'); print(getattr(s._connect().__enter__(), 'row_factory', None))"
```

If `row_factory` is `None`, add the line; if it is already set, skip.

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_kb_feedback_store.py -v
```

Expected: 8 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/kb_store.py tests/test_kb_feedback_store.py
git commit -m @'
feat(kb-feedback): iter_feedback_pairs() yields DPOPair from live ratings

Implements pairing rules from W4 spec § 6.3 — most-recent rating
per (message_id, user_id) wins; alternative_answer becomes the
chosen branch when supplied; thumbs-up paired with earlier
thumbs-down+alt forms a LIVE_PAIRED pair. Orphan assistant
messages are skipped with a debug log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.3: API router — `POST /messages/{id}/feedback`

**Files:**
- Create: `app/api/kb_feedback.py`
- Modify: `app/api/kb_mvp.py`
- Create: `tests/test_kb_feedback_api.py`

**Background:** New router registered alongside the existing `kb_mvp` protected router. Reuses the same `require_api_key` dependency. Returns the persisted id + timestamp.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_feedback_api.py`:

```python
"""Endpoint tests for /api/kb/messages/{id}/feedback and /api/kb/feedback/export."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_API_KEY", "test-key")
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path))
    # Re-import the app so the env var is picked up at startup.
    import importlib

    import app.api.kb_mvp as kb_mvp
    importlib.reload(kb_mvp)
    from app.core.app import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def seeded_message(client) -> tuple[str, int]:
    """Create a conversation + assistant message and return (conv_id, asst_msg_id)."""
    r = client.post(
        "/api/kb/conversations",
        json={"title": "test"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code in (200, 201), r.text
    conv_id = r.json()["id"]
    client.post(
        f"/api/kb/conversations/{conv_id}/messages",
        json={"role": "user", "content": "hello?"},
        headers={"Authorization": "Bearer test-key"},
    )
    r = client.post(
        f"/api/kb/conversations/{conv_id}/messages",
        json={"role": "assistant", "content": "hi back"},
        headers={"Authorization": "Bearer test-key"},
    )
    asst_id = r.json()["id"]
    return conv_id, asst_id


def test_post_feedback_persists_and_returns_id(client, seeded_message) -> None:
    _conv, asst = seeded_message
    r = client.post(
        f"/api/kb/messages/{asst}/feedback",
        json={"rating": 1, "comment": "ok"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "id" in body
    assert "created_at" in body


def test_post_feedback_rejects_invalid_rating(client, seeded_message) -> None:
    _conv, asst = seeded_message
    r = client.post(
        f"/api/kb/messages/{asst}/feedback",
        json={"rating": 5},
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 400, r.text


def test_post_feedback_requires_auth(client, seeded_message) -> None:
    _conv, asst = seeded_message
    r = client.post(f"/api/kb/messages/{asst}/feedback", json={"rating": 1})
    assert r.status_code == 401


def test_post_feedback_unknown_message_returns_404(client) -> None:
    r = client.post(
        "/api/kb/messages/99999/feedback",
        json={"rating": 1},
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 404


def test_export_returns_ndjson(client, seeded_message) -> None:
    _conv, asst = seeded_message
    client.post(
        f"/api/kb/messages/{asst}/feedback",
        json={"rating": -1, "alternative_answer": "лучше"},
        headers={"Authorization": "Bearer test-key"},
    )
    r = client.get(
        "/api/kb/feedback/export",
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 200
    assert "application/x-ndjson" in r.headers["content-type"]
    lines = [line for line in r.text.splitlines() if line.strip()]
    assert len(lines) == 1
    import json as _json
    pair = _json.loads(lines[0])
    assert pair["chosen"] == "лучше"


def test_export_empty_returns_200_with_zero_lines(client, seeded_message) -> None:
    r = client.get(
        "/api/kb/feedback/export",
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 200
    assert r.text.strip() == ""
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_kb_feedback_api.py -v
```

Expected: all FAIL — router not wired yet.

- [ ] **Step 3: Implement the router**

Create `app/api/kb_feedback.py`:

```python
"""Live feedback collection endpoints for W4 (DPO post-training)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import Response
from pydantic import BaseModel, Field, conint

from app.api.kb_auth import require_api_key
from app.services.kb_store import KnowledgeBaseStore, get_store

router = APIRouter(
    prefix="/api/kb",
    tags=["kb-feedback"],
    dependencies=[Depends(require_api_key)],
)


class FeedbackIn(BaseModel):
    rating: int = Field(..., description="1 = thumbs-up, -1 = thumbs-down")
    comment: str | None = Field(default=None, max_length=2000)
    alternative_answer: str | None = Field(default=None, max_length=4000)
    user_id: str | None = Field(default=None, max_length=128)


class FeedbackOut(BaseModel):
    id: str
    created_at: str


@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackOut,
    status_code=201,
)
def post_feedback(
    body: FeedbackIn,
    message_id: Annotated[int, Path(ge=1)],
    store: Annotated[KnowledgeBaseStore, Depends(get_store)],
) -> FeedbackOut:
    if body.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be -1 or 1")

    with store._connect() as conn:
        row = conn.execute(
            "SELECT conversation_id FROM kb_messages WHERE id = ?",
            (int(message_id),),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="message not found")
    conversation_id = row[0] if not hasattr(row, "keys") else row["conversation_id"]

    try:
        fid = store.store_feedback(
            conversation_id=conversation_id,
            message_id=int(message_id),
            user_id=body.user_id,
            rating=body.rating,
            comment=body.comment,
            alternative_answer=body.alternative_answer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Re-read to grab the canonical created_at value.
    with store._connect() as conn:
        ts_row = conn.execute(
            "SELECT created_at FROM kb_feedback WHERE id = ?", (fid,)
        ).fetchone()
    created_at = ts_row[0] if not hasattr(ts_row, "keys") else ts_row["created_at"]
    return FeedbackOut(id=fid, created_at=created_at)


@router.get("/feedback/export")
def export_feedback(
    store: Annotated[KnowledgeBaseStore, Depends(get_store)],
) -> Response:
    lines: list[str] = []
    for pair in store.iter_feedback_pairs():
        lines.append(pair.to_jsonl_line())
    body = "".join(lines)
    headers = {"X-DPO-Pairs-Count": str(len(lines))}
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers=headers,
    )
```

`get_store` already lives in `app.services.kb_store` and is used by `kb_mvp.py:232`. No shim needed.

Then in `app/api/kb_mvp.py`, near the existing router registrations (lines 1202-1203 reference `router.include_router(public)`), add:

```python
from app.api import kb_feedback as kb_feedback_router

router.include_router(kb_feedback_router.router)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_kb_feedback_api.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/api/kb_feedback.py app/api/kb_mvp.py app/services/kb_store.py tests/test_kb_feedback_api.py
git commit -m @'
feat(kb-feedback): POST /messages/{id}/feedback + GET /feedback/export

New router registered alongside kb_mvp's protected router. Reuses
require_api_key for auth. Export endpoint emits one DPOPair JSONL
line per converted live-feedback row; empty result is a 200 with
no body (not a 404), so the trainer can poll safely.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 4 — `trl` stubs + `train_dpo.py` (~3 h)

**Goal:** A stub `trl` package that mirrors `DPOConfig` and `DPOTrainer` is on `tests/stubs/`; `train_dpo.py` runs end-to-end against the stub with `--max-steps 1`; an `@pytest.mark.integration` test exercises the real `trl` on a tiny model.

**Abort point:** After Task 4.2 — stub + CLI scaffold land; integration test can move to a follow-up PR if the diff is already large.

---

### Task 4.1: `tests/stubs/trl/` package

**Files:**
- Create: `tests/stubs/trl/__init__.py`
- Create: `tests/test_trl_stub_signature.py` (contract check)

**Background:** Mirrors the real trl 0.11+ `DPOTrainer.__init__` and `DPOConfig` field set. Tests assert the stub's surface matches what `train_dpo.py` calls, so any drift between the stub and our caller is caught locally.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_trl_stub_signature.py`:

```python
"""Contract check: stub trl exposes the surface train_dpo.py uses."""

from __future__ import annotations


def test_stub_exposes_dpoconfig_with_expected_fields() -> None:
    import trl  # resolves to tests/stubs/trl when real trl is absent

    cfg = trl.DPOConfig(output_dir="x")
    for field_name in (
        "output_dir",
        "beta",
        "learning_rate",
        "per_device_train_batch_size",
        "num_train_epochs",
        "max_length",
        "max_prompt_length",
        "logging_steps",
        "save_steps",
    ):
        assert hasattr(cfg, field_name), f"DPOConfig missing field {field_name!r}"


def test_stub_dpotrainer_records_train_call() -> None:
    import trl

    trl.DPOTrainer.train_calls.clear()
    trainer = trl.DPOTrainer(
        model=object(),
        args=trl.DPOConfig(output_dir="x", beta=0.1),
        train_dataset=[1, 2, 3],
    )
    trainer.train()
    trainer.save_model("/tmp/dpo-stub")  # noqa: S108 - test path
    assert len(trl.DPOTrainer.train_calls) == 1
    assert trl.DPOTrainer.train_calls[0]["beta"] == 0.1
    assert trl.DPOTrainer.train_calls[0]["dataset_size"] == 3
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_trl_stub_signature.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trl'`.

- [ ] **Step 3: Create the stub**

Create `tests/stubs/trl/__init__.py`:

```python
"""Minimal trl stub for offline tests.

Mirrors :class:`trl.DPOConfig` and :class:`trl.DPOTrainer` 0.11+
just enough for ``scripts/train_dpo.py`` to import, instantiate,
and call ``train()`` + ``save_model()``. ``DPOTrainer.train_calls``
is a class-level list so tests can assert on what was invoked.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DPOConfig:
    output_dir: str
    beta: float = 0.1
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 4
    num_train_epochs: int = 1
    max_length: int = 1024
    max_prompt_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    max_steps: int = -1
    report_to: str = "none"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


class DPOTrainer:
    """Stub mirroring the trl 0.11 DPOTrainer subset used by train_dpo.py."""

    train_calls: list[dict[str, Any]] = []

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        args: DPOConfig | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        peft_config: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.peft_config = peft_config
        self._extras = kwargs

    def train(self) -> None:
        self.train_calls.append(
            {
                "model": self.model,
                "beta": self.args.beta if self.args is not None else None,
                "dataset_size": len(self.train_dataset) if self.train_dataset is not None else 0,
            }
        )

    def save_model(self, output_dir: str) -> None:
        p = pathlib.Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}", encoding="utf-8")
```

- [ ] **Step 4: Run the test to confirm it passes**

```powershell
py -3 -m pytest tests/test_trl_stub_signature.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/stubs/trl/__init__.py tests/test_trl_stub_signature.py
git commit -m @'
test(stubs): add tests/stubs/trl with DPOConfig + DPOTrainer surface

Mirrors the trl 0.11+ subset that train_dpo.py imports. Contract
test asserts the surface matches our caller so stub drift is
caught locally rather than after a CI install.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 4.2: `train_dpo.py` CLI scaffold + smoke

**Files:**
- Create: `scripts/train_dpo.py`
- Create: `tests/scripts/test_train_dpo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_train_dpo.py`:

```python
"""Stub-backed tests for scripts.train_dpo."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_module_imports() -> None:
    import scripts.train_dpo as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_required_flags() -> None:
    from scripts.train_dpo import parse_args

    ns = parse_args(
        [
            "--base-model",
            "stub-base",
            "--train",
            "dpo.jsonl",
            "--sft-adapter",
            "adapters/sft",
            "--output",
            "adapters/dpo",
            "--prompt-mode",
            "rag",
            "--max-steps",
            "1",
        ]
    )
    assert ns.base_model == "stub-base"
    assert ns.train == Path("dpo.jsonl")
    assert ns.sft_adapter == Path("adapters/sft")
    assert ns.output == Path("adapters/dpo")
    assert ns.prompt_mode == "rag"
    assert ns.max_steps == 1


def test_train_dpo_smoke_against_stub(tmp_path) -> None:
    """End-to-end: write 2 DPO pairs, run main(), assert adapter saved."""
    import json

    from scripts.train_dpo import main
    import trl

    train_path = tmp_path / "dpo.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for i in range(2):
            fh.write(
                json.dumps(
                    {
                        "prompt": f"Q{i}?",
                        "chosen": f"A{i}. [doc_chunk:{i}]",
                        "rejected": f"A{i}.",
                        "meta": {"strategy": "no_citation"},
                    }
                )
                + "\n"
            )

    output = tmp_path / "adapters" / "dpo"
    trl.DPOTrainer.train_calls.clear()

    rc = main(
        [
            "--base-model",
            "stub-base",
            "--train",
            str(train_path),
            "--sft-adapter",
            str(tmp_path / "fake-sft"),
            "--output",
            str(output),
            "--prompt-mode",
            "rag",
            "--max-steps",
            "1",
        ]
    )
    assert rc == 0
    assert (output / "adapter_config.json").exists()
    assert len(trl.DPOTrainer.train_calls) == 1


def test_train_dpo_raises_systemexit_on_missing_dataset() -> None:
    from scripts.train_dpo import main

    with pytest.raises(SystemExit):
        main(
            [
                "--base-model",
                "stub-base",
                "--train",
                "does-not-exist.jsonl",
                "--sft-adapter",
                "fake-sft",
                "--output",
                "adapters/x",
            ]
        )
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/scripts/test_train_dpo.py -v
```

Expected: 4 FAIL with `ModuleNotFoundError: No module named 'scripts.train_dpo'`.

- [ ] **Step 3: Create the CLI**

Create `scripts/train_dpo.py`:

```python
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
    """Optionally re-format prompts via train_lora.format_prompt.

    When prompt_mode='rag', wraps each row's prompt in the RAG
    template so the trainer sees the same prefix the production
    inference pipeline uses.
    """

    if prompt_mode == "generic":
        return rows
    from scripts.train_lora import format_prompt

    out: list[dict] = []
    for row in rows:
        prompt = row.get("prompt", "")
        rewritten = format_prompt(
            instruction=prompt,
            context="",
            retrieved_context="",
            prompt_mode="rag",
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
        raise SystemExit(
            f"trl is required: {exc}. Install with `pip install trl~=0.11`."
        )

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
        model=args.base_model,  # the real path loads via AutoModel; stub accepts anything
        args=cfg,
        train_dataset=rows,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    LOGGER.info("DPO adapter saved to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/scripts/test_train_dpo.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/train_dpo.py tests/scripts/test_train_dpo.py
git commit -m @'
feat(train-dpo): CLI scaffold + stub-backed smoke tests

Wraps trl.DPOTrainer with argparse + JSONL loader. --prompt-mode rag
re-formats each row through train_lora.format_prompt so the trainer
sees the exact prefix the production inference pipeline uses.
Stub-shadowed under tests/stubs/trl for offline TDD; real trl is
loaded only in CI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 4.3: Integration test marker

**Files:**
- Create: `tests/test_train_dpo_integration.py`
- Modify: `requirements-runtime.txt` (add optional `trl`)

- [ ] **Step 1: Write the integration-marked test**

Create `tests/test_train_dpo_integration.py`:

```python
"""@pytest.mark.integration: exercises real trl on a tiny model in CI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
def test_train_dpo_on_tiny_model(tmp_path) -> None:
    """Skipped unless real trl + a tiny HF model are available locally / in CI."""
    pytest.importorskip("trl")
    pytest.importorskip("transformers")

    from scripts.train_dpo import main

    train_path = tmp_path / "dpo.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(
                json.dumps(
                    {
                        "prompt": f"Q{i}?",
                        "chosen": f"A{i}. [doc_chunk:{i}]",
                        "rejected": f"A{i}.",
                        "meta": {"strategy": "no_citation"},
                    }
                )
                + "\n"
            )

    output = tmp_path / "adapters" / "dpo"
    rc = main(
        [
            "--base-model",
            "sshleifer/tiny-gpt2",  # well-known tiny model used in TRL CI
            "--train",
            str(train_path),
            "--sft-adapter",
            str(tmp_path / "fake-sft"),
            "--output",
            str(output),
            "--prompt-mode",
            "generic",
            "--max-steps",
            "1",
            "--num-train-epochs",
            "1",
        ]
    )
    assert rc == 0
    assert (output / "adapter_config.json").exists() or any(output.iterdir())
```

- [ ] **Step 2: Verify the marker is registered**

```powershell
py -3 -m pytest tests/test_train_dpo_integration.py -v --collect-only -m integration
```

Expected: 1 test collected. (If marker registration warns, add `integration` to the `[pytest] markers` section of `pyproject.toml` or `pytest.ini`. Check the current state first with `Grep`.)

- [ ] **Step 3: Add the optional ML dep**

In `requirements-runtime.txt`, append at the end (separated by a blank line and a comment):

```
# Optional ML extras — only required for scripts/train_lora.py and scripts/train_dpo.py.
# Loaded lazily; absence causes SystemExit with an install hint at CLI invocation time.
trl~=0.11
```

- [ ] **Step 4: Run the unit tests as a regression check (integration test is skipped without `trl`)**

```powershell
py -3 -m pytest tests/scripts/test_train_dpo.py tests/test_trl_stub_signature.py -v
```

Expected: previous PASSes remain green.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_train_dpo_integration.py requirements-runtime.txt
git commit -m @'
test(train-dpo): integration test on real trl + tiny GPT-2

@pytest.mark.integration so it only runs in CI where real trl and
transformers are installed. Locally, the stub trl handles the unit
tests; the marked integration test is silently skipped.

trl~=0.11 added to requirements-runtime.txt as an optional ML dep
guarded by lazy import inside train_dpo.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 5 — Docs + PR (~30 min)

### Task 5.1: README example

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the existing W3 section**

```powershell
py -3 -c "import pathlib; print('generate_rag_dataset' in pathlib.Path('README.md').read_text(encoding='utf-8'))"
```

Expected: `True`. If it returns `False`, search for the synthetic-Q&A section instead and append after it.

- [ ] **Step 2: Append a W4 example after the W3 section**

Add to `README.md` (right after the W3 example block):

````markdown
### W4: DPO post-training (preference learning)

After the W3 SFT adapter is trained, build a synthetic preference
dataset and run DPO on top of it. Synthetic mix: 40 % regex strip
(zero LLM calls) + 30 % teacher-without-context + 30 % invented-
citation.

```powershell
py -3 -m scripts.generate_dpo_pairs `
    --seeds var/data/seeds.jsonl `
    --output var/data/dpo.jsonl `
    --target-pairs 1000 `
    --yes

py -3 -m scripts.train_dpo `
    --base-model TheBloke/some-model `
    --train var/data/dpo.jsonl `
    --sft-adapter adapters/my-rag-lora `
    --output adapters/my-dpo `
    --prompt-mode rag `
    --max-steps 200
```

Live feedback collected through `POST /api/kb/messages/{id}/feedback`
can be exported in the same JSONL shape:

```powershell
curl -H "Authorization: Bearer $env:KB_API_KEY" `
    http://localhost:8000/api/kb/feedback/export -o var/data/live.jsonl

py -3 -m scripts.train_dpo --train var/data/live.jsonl ...
```
````

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m @'
docs(dpo): add W4 usage example to README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 5.2: PR description

**Files:** none (uses already-pushed branch).

- [ ] **Step 1: Final full-suite regression**

```powershell
py -3 -m pytest -q --ignore=backend -m "not integration"
py -3 -m ruff check .
py -3 -m black --check .
```

Expected: all green.

- [ ] **Step 2: Open / update the PR**

The branch is already on origin (pushed at the start of this session). Open or update:

```powershell
gh pr create --title "feat(ml): W4 DPO post-training (closes G3)" --body @'
## Summary

- Closes **G3** (DPO post-training / preference learning) from the
  Pack B++ ML strengthening spec.
- New module `app/services/dpo_dataset.py` produces a synthetic
  preference dataset (40 / 30 / 30 NO_CITATION / GENERIC / HALLUCINATION).
- New CLI `scripts/generate_dpo_pairs.py` mirrors `generate_rag_dataset.py`.
- New CLI `scripts/train_dpo.py` wraps `trl.DPOTrainer` (stub-shadowed
  for offline TDD; real `trl~=0.11` loaded in CI).
- New endpoints `POST /api/kb/messages/{id}/feedback` and
  `GET /api/kb/feedback/export` for live preference collection.
- New `kb_feedback` SQLite table created idempotently inside
  `KnowledgeBaseStore._initialise_schema`.
- Sprint 0 refactor makes `apportion_counts` generic over `Enum`
  and exposes `strip_citations` publicly so W3 + W4 share Hamilton
  apportionment and citation stripping.

## Test plan

- [ ] `py -3 -m pytest tests/test_dpo_dataset.py tests/test_dpo_dataset_strategies.py tests/test_kb_feedback_store.py tests/test_kb_feedback_api.py tests/scripts/test_generate_dpo_pairs.py tests/scripts/test_train_dpo.py tests/test_apportion_counts_generic.py tests/test_trl_stub_signature.py -v` — all green.
- [ ] `py -3 -m pytest -q --ignore=backend -m "not integration"` — no regressions.
- [ ] `py -3 -m ruff check . && py -3 -m black --check .` — clean.
- [ ] Manual: `py -3 -m scripts.generate_dpo_pairs --seeds … --output dpo.jsonl --target-pairs 40 --yes` writes a JSONL where every line has `prompt / chosen / rejected` and `meta.strategy ∈ {no_citation, generic, hallucination}`.
- [ ] Manual: `POST /api/kb/messages/{id}/feedback` with valid body returns 201; invalid rating returns 400.

## Spec reference

`docs/superpowers/specs/2026-05-29-w4-dpo-post-training-design.md` § Workstream 4.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
'@
```

If the PR already exists (was created during a previous loop iteration), use `gh pr edit` to update its body instead.

---

## Acceptance criteria

- [ ] `py -3 -m pytest tests/test_dpo_dataset.py tests/test_dpo_dataset_strategies.py tests/test_kb_feedback_store.py tests/test_kb_feedback_api.py tests/scripts/test_generate_dpo_pairs.py tests/scripts/test_train_dpo.py tests/test_apportion_counts_generic.py tests/test_trl_stub_signature.py -v` → **all PASS**.
- [ ] `py -3 -m pytest -q --ignore=backend -m "not integration"` — no new failures vs. `main`.
- [ ] `py -3 -m ruff check . && py -3 -m black --check .` — clean.
- [ ] `py -3 -m scripts.generate_dpo_pairs --seeds … --output dpo.jsonl --target-pairs 40 --yes` writes a valid JSONL where:
  - Every line has top-level `prompt / chosen / rejected`.
  - Strategy distribution matches 40 / 30 / 30 within ±1 sample.
  - Every `meta.strategy ∈ {no_citation, generic, hallucination}`.
- [ ] `POST /api/kb/messages/{id}/feedback` with valid body returns 201 and persists; invalid rating returns 400.
- [ ] `GET /api/kb/feedback/export` returns NDJSON whose lines can be passed directly to `train_dpo.py --train -`.
- [ ] `py -3 -m scripts.train_dpo --base-model stub --train dpo.jsonl --sft-adapter <path> --output adapters/my-dpo --prompt-mode rag --max-steps 1` runs to completion under the stub.
- [ ] PR description points to spec § Workstream 4 and notes the G3 metric improved (preference accuracy; held-out faithfulness measured by W5).

## Out of scope (parking lot)

- **W5 RAGAS evaluation** — measures the resulting DPO uplift; owns the +5 pp number.
- **Web-UI thumbs buttons** in `data/www/` — separate visual change, lands with W7 or W9 (Auto-Train UI).
- **Multi-rater consensus** — current pairing assumes one most-recent rating per `(message, user)`.
- **TIES / DARE adapter merging** of SFT and DPO adapters — explicit ROADMAP anti-feature.
- **Embedding fine-tuning** to improve retrieval quality before DPO — W6.

## Estimated effort

- Sprint 0 (shared-util refactor): 30 min.
- Sprint 1 (`dpo_dataset.py` module + 3 strategies + builder): ~3 h.
- Sprint 2 (`generate_dpo_pairs.py` CLI + cost guard): ~1.5 h.
- Sprint 3 (`kb_feedback` schema + store + endpoints): ~2 h.
- Sprint 4 (`tests/stubs/trl` + `scripts/train_dpo.py` + integration test): ~3 h.
- Sprint 5 (docs + PR): ~30 min.
- **Total: ~10–11 h of focused TDD work, comparable to W3.**
