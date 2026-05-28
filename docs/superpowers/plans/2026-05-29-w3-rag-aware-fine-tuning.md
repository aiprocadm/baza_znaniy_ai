# W3 RAG-aware Fine-tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `app/services/rag_dataset.py` + `scripts/generate_rag_dataset.py` that compose with W1's synthetic Q&A generator and `KnowledgeBaseStore.search()` to produce a 4-variant RAG-aware SFT dataset (RELEVANT / IRRELEVANT / PARTIAL / EMPTY). Wire `scripts/train_lora.py` with a new `--prompt-mode rag` flag that consumes the extended schema. This closes gap **G2** (RAG-aware fine-tuning prompts, faithfulness uplift of −15 to −25pp without it).

**Architecture:** Mirror W1's split: pure-logic module `app/services/rag_dataset.py` (no I/O) + thin CLI wrapper `scripts/generate_rag_dataset.py` (argparse + JSONL streaming + budget guard + resume). The module composes injected dependencies: an `LLMProvider` (already typed in `synthetic_qa`), a callable `retriever: (query, top_k) -> list[SearchHit]` (defaulting to `KnowledgeBaseStore.search`), and the existing `SyntheticQAGenerator` for seed Q&A creation. The 4-variant mix is encoded as integer counts via a deterministic proportions calculator (Hamilton apportionment) so the output is reproducible across CLI re-runs. `train_lora.py` grows a `PROMPT_TEMPLATE_RAG` constant and a `--prompt-mode {generic,rag}` switch; the existing generic path stays untouched and is the default.

**Tech Stack:** Python 3.12, pytest, dataclasses, argparse, json, existing `OpenAICompatibleProvider` and `KnowledgeBaseStore` from `app/services/`, existing `SyntheticQAGenerator` from `app/services/synthetic_qa.py`. No new PyPI deps.

**Spec reference:** Workstream W3 in [`docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md`](../specs/2026-05-25-ml-strengthening-pack-b-design.md). Composes with W1 plan [`docs/superpowers/plans/2026-05-25-w1-synthetic-qa-generation-plan.md`](2026-05-25-w1-synthetic-qa-generation-plan.md).

**Plans dir conventions:** Per-task atomic commits using `git commit -m @'…'@` here-strings on PowerShell (see [repo-pythonenv-py-launcher](../../../memory/repo_pythonenv_py_launcher.md) for the `py -3` launcher and no-venv setup). Every task = failing test first, then minimal implementation, then commit.

---

## File Structure

**Create:**
- `app/services/rag_dataset.py` — pure logic: `RAGVariant`, `RAGSample`, `ProportionSpec`, `RAGSampleBuilder`, JSONL I/O helpers
- `scripts/generate_rag_dataset.py` — CLI wrapper (argparse + budget guard + resume)
- `tests/test_rag_dataset.py` — unit tests for the module (fake provider + fake retriever)
- `tests/test_rag_dataset_proportions.py` — proportions calculator tests
- `tests/scripts/test_generate_rag_dataset.py` — CLI smoke test
- `tests/test_train_lora_prompt_mode.py` — `--prompt-mode rag` flag tests

**Modify:**
- `scripts/train_lora.py` — add `PROMPT_TEMPLATE_RAG`, add `--prompt-mode` arg, route through `_format_prompt`, extend `_normalise_example` for `retrieved_context`
- `README.md` — short usage example for `generate_rag_dataset.py` (similar to the existing W1 section)

**NOT modified:**
- `app/services/synthetic_qa.py` — composed, not extended (treat W1 module as a stable dependency)
- `app/services/kb_store.py` — used as-is via `search()` and `iter_chunks()`

**Output schema (JSONL line):**
```json
{
  "instruction": "<question>",
  "input": "",
  "output": "<answer with [doc_chunk:X] citations or refusal text>",
  "retrieved_context": "<top-k chunks joined by \\n\\n or empty>",
  "meta": {
    "source_chunk_id": 42,
    "variant": "relevant",
    "retrieved_chunk_ids": [42, 17, 9]
  }
}
```

Compatible with `scripts/validate_dataset.py` because it ignores unknown top-level keys; the new `retrieved_context` and the variant metadata travel as plain strings/ints.

---

## Sprint 1 — `rag_dataset.py` module + 4-variant builder (~4h)

**Goal:** All 8 module-level tests pass. The module produces deterministic `RAGSample` objects for each variant given a fake provider and fake retriever.

**Abort point:** After Task 1.5 (RELEVANT + IRRELEVANT variants done) — that already covers ~85% of the dataset distribution and the rest can be filled by W1 samples if time is tight.

---

### Task 1.1: Module skeleton + import test

**Files:**
- Create: `app/services/rag_dataset.py`
- Create: `tests/test_rag_dataset.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_rag_dataset.py`:
```python
"""Tests for app.services.rag_dataset — pure-logic RAG dataset builder."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module imports without side effects."""
    from app.services import rag_dataset

    assert rag_dataset.__name__ == "app.services.rag_dataset"
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_rag_dataset.py::test_module_imports -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.rag_dataset'`.

- [ ] **Step 3: Create the minimal module**

Create `app/services/rag_dataset.py`:
```python
"""Compose a RAG-aware SFT dataset from a KB corpus + teacher LLM.

This module is the pure-logic core of Workstream 3 (RAG-aware
fine-tuning) in the Pack B++ ML strengthening plan. It builds on
W1's :mod:`app.services.synthetic_qa` for seed Q&A generation and
on :class:`app.services.kb_store.KnowledgeBaseStore` for retrieval.

The module is intentionally I/O free: provider, retriever, and chunk
source are injected so the logic is deterministic in tests.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)
```

- [ ] **Step 4: Run the test to confirm it passes**

```powershell
py -3 -m pytest tests/test_rag_dataset.py::test_module_imports -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): module skeleton for W3 RAG-aware dataset builder

Empty module + import test as the foundation for Workstream 3
(closes G2: RAG-aware fine-tuning prompts).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.2: `RAGVariant` enum + `RAGSample` dataclass

**Files:**
- Modify: `app/services/rag_dataset.py` (append)
- Modify: `tests/test_rag_dataset.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:
```python
def test_rag_variant_values() -> None:
    """The four canonical variants are exposed as string enum members."""
    from app.services.rag_dataset import RAGVariant

    assert {v.value for v in RAGVariant} == {
        "relevant",
        "irrelevant",
        "partial",
        "empty",
    }


def test_rag_sample_to_jsonl_line() -> None:
    """RAGSample.to_jsonl_line() emits one JSON object per line."""
    import json

    from app.services.rag_dataset import RAGSample, RAGVariant

    sample = RAGSample(
        instruction="Что такое отпуск?",
        input="",
        output="Отпуск — это [doc_chunk:7]",
        retrieved_context="Фрагмент [doc_chunk:7]: ...",
        variant=RAGVariant.RELEVANT,
        source_chunk_id=7,
        retrieved_chunk_ids=(7, 12),
    )
    line = sample.to_jsonl_line()
    assert line.endswith("\n")

    data = json.loads(line)
    assert data["instruction"] == "Что такое отпуск?"
    assert data["retrieved_context"].startswith("Фрагмент")
    assert data["meta"]["variant"] == "relevant"
    assert data["meta"]["source_chunk_id"] == 7
    assert data["meta"]["retrieved_chunk_ids"] == [7, 12]
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v
```
Expected: 2 FAIL, 1 PASS (`ImportError: cannot import name 'RAGVariant'`).

- [ ] **Step 3: Implement `RAGVariant` and `RAGSample`**

Append to `app/services/rag_dataset.py`:
```python
import json
from dataclasses import dataclass
from enum import Enum


class RAGVariant(str, Enum):
    """The four training-distribution variants from the W3 spec.

    See ``docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md``
    section "Workstream 3" for the rationale and target proportions.
    """

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class RAGSample:
    """One RAG-aware training example ready for SFT.

    The top-level layout (``instruction`` / ``input`` / ``output``) keeps
    ``scripts/validate_dataset.py`` happy. ``retrieved_context`` is the
    new field consumed by ``train_lora.py --prompt-mode rag``. The
    ``meta`` sidecar carries variant + retrieval lineage so resume and
    audit queries work.
    """

    instruction: str
    input: str
    output: str
    retrieved_context: str
    variant: RAGVariant
    source_chunk_id: int
    retrieved_chunk_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "retrieved_context": self.retrieved_context,
            "meta": {
                "source_chunk_id": int(self.source_chunk_id),
                "variant": self.variant.value,
                "retrieved_chunk_ids": [int(c) for c in self.retrieved_chunk_ids],
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): add RAGVariant enum and RAGSample dataclass

Defines the four training-distribution variants (relevant,
irrelevant, partial, empty) and the JSONL-serialisable RAGSample.
Mirrors the QAPair layout from W1 with an extra retrieved_context
field consumed by train_lora.py --prompt-mode rag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.3: Proportions calculator (Hamilton apportionment)

**Files:**
- Create: `tests/test_rag_dataset_proportions.py`
- Modify: `app/services/rag_dataset.py`

**Background:** The spec lists target shares 70 / 15 / 10 / 5. Naïve `int(target * total)` would round-down all four and lose samples. Hamilton's largest-remainder method keeps the sum exact and is deterministic.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_dataset_proportions.py`:
```python
"""Hamilton apportionment for the RAG variant distribution."""

from __future__ import annotations

import pytest


def test_proportion_spec_default_70_15_10_5() -> None:
    """Defaults match the spec — 70 / 15 / 10 / 5."""
    from app.services.rag_dataset import RAGVariant, default_proportions

    p = default_proportions()
    assert p[RAGVariant.RELEVANT] == 0.70
    assert p[RAGVariant.IRRELEVANT] == 0.15
    assert p[RAGVariant.PARTIAL] == 0.10
    assert p[RAGVariant.EMPTY] == 0.05


def test_apportion_sum_matches_total() -> None:
    """Apportionment never loses or invents samples."""
    from app.services.rag_dataset import apportion_counts, default_proportions

    counts = apportion_counts(default_proportions(), total=100)
    assert sum(counts.values()) == 100


@pytest.mark.parametrize("total", [1, 7, 23, 100, 257])
def test_apportion_total_invariant(total: int) -> None:
    from app.services.rag_dataset import apportion_counts, default_proportions

    counts = apportion_counts(default_proportions(), total=total)
    assert sum(counts.values()) == total


def test_apportion_zero_total_yields_zeros() -> None:
    from app.services.rag_dataset import RAGVariant, apportion_counts, default_proportions

    counts = apportion_counts(default_proportions(), total=0)
    assert all(counts[v] == 0 for v in RAGVariant)


def test_custom_proportions_validated() -> None:
    """Proportions must sum to 1.0 within float tolerance."""
    from app.services.rag_dataset import RAGVariant, apportion_counts

    with pytest.raises(ValueError, match="proportions must sum"):
        apportion_counts(
            {RAGVariant.RELEVANT: 0.5, RAGVariant.IRRELEVANT: 0.4},  # 0.9
            total=10,
        )
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_rag_dataset_proportions.py -v
```
Expected: 5 FAIL with `ImportError`.

- [ ] **Step 3: Implement `default_proportions` and `apportion_counts`**

Append to `app/services/rag_dataset.py`:
```python
from typing import Mapping


ProportionSpec = Mapping["RAGVariant", float]


def default_proportions() -> dict[RAGVariant, float]:
    """Return the W3 spec defaults: 70 / 15 / 10 / 5."""

    return {
        RAGVariant.RELEVANT: 0.70,
        RAGVariant.IRRELEVANT: 0.15,
        RAGVariant.PARTIAL: 0.10,
        RAGVariant.EMPTY: 0.05,
    }


_PROPORTION_TOLERANCE = 1e-6


def apportion_counts(
    proportions: ProportionSpec,
    *,
    total: int,
) -> dict[RAGVariant, int]:
    """Hamilton's largest-remainder method.

    Given target shares summing to 1.0, return integer counts per
    variant whose sum equals ``total`` exactly. Deterministic ordering
    (RELEVANT, IRRELEVANT, PARTIAL, EMPTY) breaks remainder ties.
    """

    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    share_sum = sum(proportions.values())
    if abs(share_sum - 1.0) > _PROPORTION_TOLERANCE:
        raise ValueError(
            f"proportions must sum to 1.0 (within {_PROPORTION_TOLERANCE}); got {share_sum}"
        )

    counts: dict[RAGVariant, int] = {v: 0 for v in RAGVariant}
    if total == 0:
        return counts

    raw = [(v, proportions.get(v, 0.0) * total) for v in RAGVariant]
    floors = [(v, int(value)) for v, value in raw]
    assigned = sum(c for _, c in floors)
    leftover = total - assigned

    remainders = sorted(
        ((v, value - int(value)) for v, value in raw),
        key=lambda item: (-item[1], list(RAGVariant).index(item[0])),
    )
    for v, count in floors:
        counts[v] = count
    for i in range(leftover):
        counts[remainders[i][0]] += 1
    return counts
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_rag_dataset_proportions.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset_proportions.py
git commit -m @'
feat(rag-dataset): Hamilton apportionment for variant proportions

default_proportions() returns the W3 spec defaults (70/15/10/5).
apportion_counts() distributes a target total across the four
RAGVariant members so the sum is exact and the ordering is
deterministic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.4: RELEVANT-variant builder

**Files:**
- Modify: `app/services/rag_dataset.py`
- Modify: `tests/test_rag_dataset.py`

**Background:** A RELEVANT sample takes a seed Q&A pair (from W1's `SyntheticQAGenerator`), retrieves top-k chunks for the question, and emits a sample whose `retrieved_context` is the joined chunk text and whose `output` carries the seed answer (already citation-formatted by W1). The source chunk should appear in the retrieved set; if it does not, the W1 seed was probably generated against a stale embedding and the sample is dropped.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:
```python
from dataclasses import dataclass
from typing import Sequence

from app.services.synthetic_qa import QAPair


@dataclass(frozen=True)
class _FakeHit:
    """Minimal stand-in for app.services.kb_store.SearchHit."""

    chunk_index: int
    text: str
    document_id: int = 1
    document_title: str = "doc"
    score: float = 0.9
    source: str = "text"


def _retriever_with(hits_by_query: dict[str, list[_FakeHit]]):
    def _retrieve(query: str, top_k: int) -> Sequence[_FakeHit]:
        return list(hits_by_query.get(query, []))[:top_k]

    return _retrieve


def test_build_relevant_sample_joins_top_k_chunks() -> None:
    from app.services.rag_dataset import RAGVariant, build_relevant_sample

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    retriever = _retriever_with(
        {
            "Что такое отпуск?": [
                _FakeHit(chunk_index=7, text="Отпуск — это перерыв."),
                _FakeHit(chunk_index=12, text="Сотрудник имеет право."),
            ],
        }
    )

    sample = build_relevant_sample(seed, retriever=retriever, top_k=3)
    assert sample is not None
    assert sample.variant is RAGVariant.RELEVANT
    assert sample.source_chunk_id == 7
    assert 7 in sample.retrieved_chunk_ids
    assert "Отпуск — это перерыв." in sample.retrieved_context
    assert sample.output.endswith("[doc_chunk:7]")


def test_build_relevant_drops_when_source_chunk_missing() -> None:
    """If retrieval can't find the seed chunk, the sample is unsafe — drop it."""
    from app.services.rag_dataset import build_relevant_sample

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    retriever = _retriever_with(
        {"Что такое отпуск?": [_FakeHit(chunk_index=99, text="Совсем не про отпуск.")]}
    )

    assert build_relevant_sample(seed, retriever=retriever, top_k=3) is None
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k relevant
```
Expected: 2 FAIL with `ImportError` on `build_relevant_sample`.

- [ ] **Step 3: Implement `build_relevant_sample`**

Append to `app/services/rag_dataset.py`:
```python
from typing import Callable, Sequence

from app.services.synthetic_qa import QAPair

# Retriever callbacks receive (query, top_k) and return any sequence
# of objects exposing ``.chunk_index`` (int) and ``.text`` (str). That
# is intentionally a subset of ``kb_store.SearchHit`` so unit tests can
# pass lightweight fakes without importing the heavy real type.
Retriever = Callable[[str, int], Sequence[object]]


def _join_chunks(hits: Sequence[object]) -> str:
    blocks: list[str] = []
    for hit in hits:
        cid = int(getattr(hit, "chunk_index"))
        text = str(getattr(hit, "text", "")).strip()
        if text:
            blocks.append(f"Фрагмент [doc_chunk:{cid}]:\n{text}")
    return "\n\n".join(blocks)


def _chunk_ids(hits: Sequence[object]) -> tuple[int, ...]:
    return tuple(int(getattr(hit, "chunk_index")) for hit in hits)


def build_relevant_sample(
    seed: QAPair,
    *,
    retriever: Retriever,
    top_k: int = 3,
) -> RAGSample | None:
    """Promote a seed Q&A to a RELEVANT variant by attaching retrieved context.

    Returns ``None`` when the seed chunk is not in the top-k retrieval
    set — that means the seed answer cannot be grounded in the context
    we plan to feed at inference time, so training on it would teach
    the model to hallucinate.
    """

    hits = list(retriever(seed.instruction, top_k))
    ids = _chunk_ids(hits)
    if seed.source_chunk_id not in ids:
        return None

    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=seed.output,
        retrieved_context=_join_chunks(hits),
        variant=RAGVariant.RELEVANT,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=ids,
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k relevant
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): RELEVANT variant builder

Promotes a W1 seed QAPair to a RELEVANT RAGSample by attaching the
joined top-k retrieved chunks as retrieved_context. Drops the seed
when the source chunk is not in the top-k — training on
ungrounded answers would teach hallucination.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.5: IRRELEVANT-variant builder

**Files:**
- Modify: `app/services/rag_dataset.py`
- Modify: `tests/test_rag_dataset.py`

**Background:** An IRRELEVANT sample teaches the model to refuse. We take a seed question and pair it with chunks from a *different* document — picked via the retriever for a contrasting query that the caller supplies, or via random sampling from a `negative_pool` of chunks. The output is a fixed refusal string.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:
```python
def test_build_irrelevant_sample_uses_negative_chunks_and_refusal() -> None:
    from app.services.rag_dataset import (
        IRRELEVANT_REFUSAL,
        RAGVariant,
        build_irrelevant_sample,
    )

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    negative_chunks = [
        _FakeHit(chunk_index=200, text="Калибровка манометра — раз в год."),
        _FakeHit(chunk_index=201, text="Поверка средств измерения."),
    ]

    sample = build_irrelevant_sample(
        seed,
        negative_chunks=negative_chunks,
    )

    assert sample.variant is RAGVariant.IRRELEVANT
    assert sample.output == IRRELEVANT_REFUSAL
    assert sample.retrieved_chunk_ids == (200, 201)
    assert "Калибровка" in sample.retrieved_context
    assert sample.source_chunk_id == seed.source_chunk_id


def test_irrelevant_refusal_is_localised() -> None:
    """The refusal string mentions documents (not generic AI talk)."""
    from app.services.rag_dataset import IRRELEVANT_REFUSAL

    assert "документ" in IRRELEVANT_REFUSAL.lower()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k irrelevant
```
Expected: 2 FAIL.

- [ ] **Step 3: Implement `build_irrelevant_sample`**

Append to `app/services/rag_dataset.py`:
```python
IRRELEVANT_REFUSAL = "Не удалось найти в документах информацию для ответа."


def build_irrelevant_sample(
    seed: QAPair,
    *,
    negative_chunks: Sequence[object],
) -> RAGSample:
    """Pair the seed question with unrelated context and a refusal answer.

    Caller is responsible for picking truly unrelated ``negative_chunks``
    (e.g. from a different document). The W3 spec target share is 15 %.
    """

    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=IRRELEVANT_REFUSAL,
        retrieved_context=_join_chunks(negative_chunks),
        variant=RAGVariant.IRRELEVANT,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=_chunk_ids(negative_chunks),
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k irrelevant
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): IRRELEVANT variant builder + refusal constant

Pairs a seed question with unrelated chunks (caller-provided) and a
fixed Russian refusal output. Trains the model to say "не нашёл" on
out-of-corpus questions instead of hallucinating.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.6: PARTIAL-variant builder

**Files:**
- Modify: `app/services/rag_dataset.py`
- Modify: `tests/test_rag_dataset.py`

**Background:** PARTIAL means the top-k retrieval contains the source chunk plus distractors, and the answer is rewritten to start with a hedging clause (`"По доступным фрагментам..."`) before citing only the supporting chunk. The caller supplies `distractor_chunks`; we mix them with the seed chunk in random-but-deterministic order (seed first to keep tests reproducible).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:
```python
def test_build_partial_sample_mixes_seed_with_distractors() -> None:
    from app.services.rag_dataset import (
        PARTIAL_PREFIX,
        RAGVariant,
        build_partial_sample,
    )

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    seed_hit = _FakeHit(chunk_index=7, text="Отпуск — это перерыв в работе.")
    distractors = [
        _FakeHit(chunk_index=200, text="Калибровка манометра — раз в год."),
        _FakeHit(chunk_index=201, text="Поверка средств измерения."),
    ]

    sample = build_partial_sample(
        seed,
        seed_hit=seed_hit,
        distractor_chunks=distractors,
    )

    assert sample.variant is RAGVariant.PARTIAL
    assert sample.output.startswith(PARTIAL_PREFIX)
    assert "[doc_chunk:7]" in sample.output
    assert sample.retrieved_chunk_ids[0] == 7
    assert 200 in sample.retrieved_chunk_ids
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k partial
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_partial_sample`**

Append to `app/services/rag_dataset.py`:
```python
PARTIAL_PREFIX = "По доступным фрагментам: "


def build_partial_sample(
    seed: QAPair,
    *,
    seed_hit: object,
    distractor_chunks: Sequence[object],
) -> RAGSample:
    """Mix the seed chunk with distractors and hedge the answer.

    The hedged output keeps the seed citation intact so the model
    still learns the citation format; the prefix teaches caution when
    only part of the context is on-topic.
    """

    mixed = [seed_hit, *distractor_chunks]
    hedged_output = PARTIAL_PREFIX + seed.output
    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=hedged_output,
        retrieved_context=_join_chunks(mixed),
        variant=RAGVariant.PARTIAL,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=_chunk_ids(mixed),
    )
```

- [ ] **Step 4: Run the test to confirm it passes**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k partial
```
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): PARTIAL variant builder

Mixes the seed chunk with caller-supplied distractors and hedges
the answer with a Russian prefix. Teaches the model to be cautious
when retrieval is only partly on-topic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.7: EMPTY-variant builder

**Files:**
- Modify: `app/services/rag_dataset.py`
- Modify: `tests/test_rag_dataset.py`

**Background:** EMPTY samples train the model on closed-book knowledge with no retrieved context. We keep the seed Q&A but strip the citation suffix from `output` (it would be misleading without context) and set `retrieved_context=""`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:
```python
def test_build_empty_sample_strips_citation_and_context() -> None:
    from app.services.rag_dataset import RAGVariant, build_empty_sample

    seed = QAPair(
        instruction="Какой сегодня день недели по тексту?",
        input="",
        output="Понедельник. [doc_chunk:42]",
        source_chunk_id=42,
    )
    sample = build_empty_sample(seed)

    assert sample.variant is RAGVariant.EMPTY
    assert sample.retrieved_context == ""
    assert sample.retrieved_chunk_ids == ()
    assert sample.output == "Понедельник."
    assert "[doc_chunk:" not in sample.output
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k empty
```
Expected: FAIL.

- [ ] **Step 3: Implement `build_empty_sample`**

Append to `app/services/rag_dataset.py`:
```python
import re

_CITATION_RE = re.compile(r"\s*\[doc_chunk:\d+\]\s*")


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text).strip()


def build_empty_sample(seed: QAPair) -> RAGSample:
    """Drop retrieved context and citation suffix from a seed Q&A."""

    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=_strip_citations(seed.output),
        retrieved_context="",
        variant=RAGVariant.EMPTY,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=(),
    )
```

- [ ] **Step 4: Run the test to confirm it passes**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k empty
```
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): EMPTY variant builder

Strips retrieved_context and the citation suffix from the seed Q&A
so the model practices closed-book answers without dangling
citations. Target share is 5% per the W3 spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.8: `RAGSampleBuilder` orchestrator + sprint sweep

**Files:**
- Modify: `app/services/rag_dataset.py`
- Modify: `tests/test_rag_dataset.py`

**Background:** The orchestrator pulls it together: given an iterator of seed Q&A pairs, a retriever, a negative-chunk pool, target proportions, and a total, it yields `RAGSample` instances respecting the apportioned mix. It iterates seeds round-robin across variants and skips variants whose quota is already filled.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_dataset.py`:
```python
def test_rag_sample_builder_respects_proportions() -> None:
    from collections import Counter

    from app.services.rag_dataset import RAGSampleBuilder, default_proportions

    seeds = [
        QAPair(
            instruction=f"Вопрос {i}?",
            input="",
            output=f"Ответ. [doc_chunk:{i}]",
            source_chunk_id=i,
        )
        for i in range(1, 21)
    ]
    seed_hits = {i: _FakeHit(chunk_index=i, text=f"Текст {i}") for i in range(1, 21)}

    def retriever(query: str, top_k: int):
        i = int(query.split()[1].rstrip("?"))
        return [seed_hits[i]]

    negatives = [_FakeHit(chunk_index=900 + j, text=f"Шум {j}") for j in range(5)]
    distractors = [_FakeHit(chunk_index=800 + j, text=f"Помеха {j}") for j in range(5)]

    builder = RAGSampleBuilder(
        retriever=retriever,
        negative_pool=negatives,
        distractor_pool=distractors,
        proportions=default_proportions(),
    )

    samples = list(builder.build(seeds, total=20))
    assert len(samples) == 20
    counts = Counter(s.variant.value for s in samples)
    assert counts["relevant"] == 14  # 70% of 20
    assert counts["irrelevant"] == 3  # 15% of 20
    assert counts["partial"] == 2  # 10% of 20
    assert counts["empty"] == 1  # 5% of 20


def test_rag_sample_builder_skips_relevant_when_source_missing() -> None:
    """If the retriever can't find the seed chunk, that slot is re-allocated."""
    from app.services.rag_dataset import RAGSampleBuilder, default_proportions

    seeds = [
        QAPair(
            instruction="Q1?",
            input="",
            output="A. [doc_chunk:1]",
            source_chunk_id=1,
        ),
        QAPair(
            instruction="Q2?",
            input="",
            output="A. [doc_chunk:2]",
            source_chunk_id=2,
        ),
    ]

    def retriever(query: str, top_k: int):
        return []  # always empty — every RELEVANT slot should be dropped

    builder = RAGSampleBuilder(
        retriever=retriever,
        negative_pool=[_FakeHit(chunk_index=900, text="нет")],
        distractor_pool=[_FakeHit(chunk_index=800, text="нет")],
        proportions=default_proportions(),
    )

    # We asked for 2 — both would have been RELEVANT but retrieval failed;
    # the builder is allowed to return fewer than requested.
    samples = list(builder.build(seeds, total=2))
    for s in samples:
        assert s.variant.value != "relevant"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k builder
```
Expected: 2 FAIL.

- [ ] **Step 3: Implement `RAGSampleBuilder`**

Append to `app/services/rag_dataset.py`:
```python
from dataclasses import field
from typing import Iterable, Iterator


@dataclass(slots=True)
class RAGSampleBuilder:
    """Orchestrate variant assembly across an iterable of seed Q&A pairs.

    The builder is the only place that knows about proportions; the
    per-variant ``build_*`` helpers stay independent and reusable.
    """

    retriever: Retriever
    negative_pool: Sequence[object]
    distractor_pool: Sequence[object]
    proportions: ProportionSpec = field(default_factory=default_proportions)
    top_k: int = 3

    def build(
        self,
        seeds: Iterable[QAPair],
        *,
        total: int,
    ) -> Iterator[RAGSample]:
        counts = apportion_counts(self.proportions, total=total)
        emitted: dict[RAGVariant, int] = {v: 0 for v in RAGVariant}

        order = (
            RAGVariant.RELEVANT,
            RAGVariant.IRRELEVANT,
            RAGVariant.PARTIAL,
            RAGVariant.EMPTY,
        )

        for seed in seeds:
            if sum(emitted.values()) >= total:
                return
            for variant in order:
                if emitted[variant] >= counts[variant]:
                    continue
                sample = self._build_one(seed, variant)
                if sample is None:
                    continue
                emitted[variant] += 1
                yield sample
                break

    def _build_one(self, seed: QAPair, variant: RAGVariant) -> RAGSample | None:
        if variant is RAGVariant.RELEVANT:
            return build_relevant_sample(seed, retriever=self.retriever, top_k=self.top_k)
        if variant is RAGVariant.IRRELEVANT:
            return build_irrelevant_sample(seed, negative_chunks=self.negative_pool)
        if variant is RAGVariant.PARTIAL:
            hits = list(self.retriever(seed.instruction, self.top_k))
            seed_hit = next((h for h in hits if int(getattr(h, "chunk_index")) == seed.source_chunk_id), None)
            if seed_hit is None:
                return None
            return build_partial_sample(
                seed,
                seed_hit=seed_hit,
                distractor_chunks=self.distractor_pool,
            )
        if variant is RAGVariant.EMPTY:
            return build_empty_sample(seed)
        return None
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_rag_dataset.py -v -k builder
```
Expected: 2 PASS.

- [ ] **Step 5: Sprint regression sweep**

```powershell
py -3 -m pytest tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py -v
py -3 -m ruff check app/services/rag_dataset.py tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py
py -3 -m black --check app/services/rag_dataset.py tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py
```
Expected: all green.

- [ ] **Step 6: Commit**

```powershell
git add app/services/rag_dataset.py tests/test_rag_dataset.py
git commit -m @'
feat(rag-dataset): RAGSampleBuilder orchestrator across all 4 variants

Iterates over seed QAPairs, dispatches to the per-variant builder,
respects apportioned counts, and gracefully under-delivers when
RELEVANT/PARTIAL retrievals fail (rather than padding with bogus
samples).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 2 — CLI wrapper + `train_lora.py` integration (~2-3h)

**Goal:** `python -m scripts.generate_rag_dataset --corpus … --output …` produces a valid JSONL on a tiny fixture. `python -m scripts.train_lora --prompt-mode rag --train …` accepts the new flag and routes through the new prompt template.

**Abort point:** After Task 2.3 — CLI works end-to-end; the train_lora integration can land as a separate PR if reviewers want smaller diffs.

---

### Task 2.1: CLI scaffold (`scripts/generate_rag_dataset.py`)

**Files:**
- Create: `scripts/generate_rag_dataset.py`
- Create: `tests/scripts/test_generate_rag_dataset.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/scripts/test_generate_rag_dataset.py`:
```python
"""Smoke tests for scripts.generate_rag_dataset CLI."""

from __future__ import annotations


def test_cli_module_imports() -> None:
    import scripts.generate_rag_dataset as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_minimal() -> None:
    from pathlib import Path

    from scripts.generate_rag_dataset import parse_args

    ns = parse_args(
        [
            "--corpus",
            "var/data/kb.sqlite",
            "--seeds",
            "var/data/seeds.jsonl",
            "--output",
            "var/data/rag.jsonl",
            "--target-pairs",
            "100",
        ]
    )
    assert ns.corpus == Path("var/data/kb.sqlite")
    assert ns.seeds == Path("var/data/seeds.jsonl")
    assert ns.output == Path("var/data/rag.jsonl")
    assert ns.target_pairs == 100
    assert ns.top_k == 3  # default
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/scripts/test_generate_rag_dataset.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.generate_rag_dataset'`.

- [ ] **Step 3: Create the scaffold**

Create `scripts/generate_rag_dataset.py`:
```python
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/scripts/test_generate_rag_dataset.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/generate_rag_dataset.py tests/scripts/test_generate_rag_dataset.py
git commit -m @'
feat(rag-dataset): CLI scaffold for generate_rag_dataset.py

Argparse contract + main() stub. Wiring to RAGSampleBuilder lands
in the next task to keep diffs reviewable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 2.2: CLI main loop — seed loading, retrieval, JSONL write

**Files:**
- Modify: `scripts/generate_rag_dataset.py`
- Modify: `tests/scripts/test_generate_rag_dataset.py`

- [ ] **Step 1: Write the failing end-to-end smoke test**

Append to `tests/scripts/test_generate_rag_dataset.py`:
```python
def test_cli_writes_jsonl_endtoend(tmp_path) -> None:
    """Tiny fixture: 2 seed Q&A pairs + 3 chunks in SQLite → 2 RAG samples."""
    import json
    import sqlite3
    from app.services.synthetic_qa import QAPair
    from scripts.generate_rag_dataset import main

    db = tmp_path / "kb.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE kb_documents (id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE kb_chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            chunk_index INTEGER,
            text TEXT
        );
        INSERT INTO kb_documents VALUES (1, 'd1'), (2, 'd2');
        INSERT INTO kb_chunks VALUES
            (1, 1, 0, 'Отпуск — это перерыв в работе.'),
            (2, 1, 1, 'Сотрудник имеет право на 28 дней отпуска.'),
            (3, 2, 0, 'Калибровка манометра проводится раз в год.');
        """
    )
    conn.commit()
    conn.close()

    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w", encoding="utf-8") as fh:
        fh.write(
            QAPair(
                instruction="Что такое отпуск?",
                input="",
                output="Перерыв в работе. [doc_chunk:1]",
                source_chunk_id=1,
            ).to_jsonl_line()
        )
        fh.write(
            QAPair(
                instruction="Сколько дней отпуска?",
                input="",
                output="28 дней. [doc_chunk:2]",
                source_chunk_id=2,
            ).to_jsonl_line()
        )

    output = tmp_path / "rag.jsonl"
    rc = main(
        [
            "--corpus",
            str(db),
            "--seeds",
            str(seeds_path),
            "--output",
            str(output),
            "--target-pairs",
            "2",
            "--top-k",
            "2",
        ]
    )
    assert rc == 0
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 1
    assert all("retrieved_context" in line for line in lines)
    assert all("variant" in line["meta"] for line in lines)
```

- [ ] **Step 2: Run the test to confirm it fails**

```powershell
py -3 -m pytest tests/scripts/test_generate_rag_dataset.py::test_cli_writes_jsonl_endtoend -v
```
Expected: FAIL (CLI is still a stub).

- [ ] **Step 3: Wire the main loop**

Replace the `main()` body in `scripts/generate_rag_dataset.py` with:
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


def _negative_pool(store, *, exclude_document_id: int | None, negative_document_id: int | None):
    from app.services.synthetic_qa import iter_chunks
    from app.services.kb_store import SearchHit

    pool: list[SearchHit] = []
    with store._connect() as conn:  # noqa: SLF001
        sql = "SELECT id, document_id, chunk_index, text FROM kb_chunks"
        params: tuple = ()
        clauses: list[str] = []
        if negative_document_id is not None:
            clauses.append("document_id = ?")
            params = (int(negative_document_id),)
        elif exclude_document_id is not None:
            clauses.append("document_id != ?")
            params = (int(exclude_document_id),)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id ASC LIMIT 50"
        for row in conn.execute(sql, params):
            pool.append(
                SearchHit(
                    document_id=int(row[1]),
                    document_title="",
                    chunk_index=int(row[2]),
                    text=str(row[3] or ""),
                    score=0.0,
                )
            )
    return pool


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.services.kb_store import KnowledgeBaseStore
    from app.services.rag_dataset import RAGSampleBuilder, default_proportions

    if not args.corpus.is_file():
        raise SystemExit(f"Corpus file not found: {args.corpus}")
    if not args.seeds.is_file():
        raise SystemExit(f"Seeds file not found: {args.seeds}")

    store = KnowledgeBaseStore(db_path=args.corpus)
    seeds = _load_seeds(args.seeds)
    if not seeds:
        LOGGER.warning("No seeds loaded from %s; nothing to do.", args.seeds)
        return 0

    if args.resume:
        already = _resume_seed_ids(args.output)
        before = len(seeds)
        seeds = [s for s in seeds if s.source_chunk_id not in already]
        LOGGER.info("Resume: skipping %d seeds already in output.", before - len(seeds))

    def retriever(query: str, top_k: int):
        return store.search(query, top_k=top_k)

    pool = _negative_pool(
        store,
        exclude_document_id=None,
        negative_document_id=args.negative_document_id,
    )

    builder = RAGSampleBuilder(
        retriever=retriever,
        negative_pool=pool,
        distractor_pool=pool,
        proportions=default_proportions(),
        top_k=args.top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    with args.output.open(open_mode, encoding="utf-8") as fh:
        for sample in builder.build(seeds, total=args.target_pairs):
            fh.write(sample.to_jsonl_line())
            fh.flush()
            written += 1

    LOGGER.info("Done: %d RAG samples written to %s", written, args.output)
    return 0


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
                try:
                    seen.add(int(meta.get("source_chunk_id")))
                except (TypeError, ValueError):
                    continue
    return seen
```

- [ ] **Step 4: Run the smoke test to confirm it passes**

```powershell
py -3 -m pytest tests/scripts/test_generate_rag_dataset.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/generate_rag_dataset.py tests/scripts/test_generate_rag_dataset.py
git commit -m @'
feat(rag-dataset): wire CLI main loop end-to-end

Loads W1 seed JSONL, retrieves top-k chunks via KnowledgeBaseStore,
samples a negative pool from a different document, and streams
RAG samples through RAGSampleBuilder into the output JSONL.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 2.3: `train_lora.py --prompt-mode rag` integration

**Files:**
- Modify: `scripts/train_lora.py`
- Create: `tests/test_train_lora_prompt_mode.py`

**Background:** Adds `PROMPT_TEMPLATE_RAG`, a new `--prompt-mode {generic,rag}` argument defaulting to `generic`, and routes `_format_prompt` through whichever template is active. `_normalise_example` learns to pick `retrieved_context` when present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_lora_prompt_mode.py`:
```python
"""Tests for the --prompt-mode flag added by W3."""

from __future__ import annotations


def test_prompt_template_rag_exists() -> None:
    from scripts.train_lora import PROMPT_TEMPLATE_RAG

    assert "{retrieved_context}" in PROMPT_TEMPLATE_RAG
    assert "{instruction}" in PROMPT_TEMPLATE_RAG


def test_parse_args_accepts_prompt_mode_rag() -> None:
    from pathlib import Path
    from scripts.train_lora import parse_args

    ns = parse_args(
        [
            "--base-model",
            "stub",
            "--train",
            "train.jsonl",
            "--output",
            "out",
            "--prompt-mode",
            "rag",
        ]
    )
    assert ns.prompt_mode == "rag"


def test_parse_args_prompt_mode_defaults_to_generic() -> None:
    from scripts.train_lora import parse_args

    ns = parse_args(
        [
            "--base-model",
            "stub",
            "--train",
            "train.jsonl",
            "--output",
            "out",
        ]
    )
    assert ns.prompt_mode == "generic"


def test_format_prompt_rag_uses_retrieved_context() -> None:
    from scripts.train_lora import format_prompt

    out = format_prompt(
        instruction="Что такое отпуск?",
        context="",
        retrieved_context="Фрагмент [doc_chunk:7]: Отпуск — перерыв.",
        prompt_mode="rag",
    )
    assert "Фрагмент [doc_chunk:7]" in out
    assert "Что такое отпуск?" in out


def test_format_prompt_generic_ignores_retrieved_context() -> None:
    """Backwards compat: generic mode behaves like before W3."""
    from scripts.train_lora import format_prompt

    out = format_prompt(
        instruction="Hi",
        context="extra",
        retrieved_context="should be ignored",
        prompt_mode="generic",
    )
    assert "should be ignored" not in out
    assert "Hi" in out
```

- [ ] **Step 2: Run the tests to confirm they fail**

```powershell
py -3 -m pytest tests/test_train_lora_prompt_mode.py -v
```
Expected: 5 FAIL with `ImportError` on `PROMPT_TEMPLATE_RAG` and `format_prompt`.

- [ ] **Step 3: Modify `scripts/train_lora.py`**

After the existing `PROMPT_TEMPLATE = ...` line (around line 29), add:
```python
PROMPT_TEMPLATE_RAG = (
    "<s>[INST] <<SYS>>\n"
    "Ответь на вопрос, используя контекст и свои знания. Если контекст "
    "релевантен — приоритизируй его. Указывай источник цитаты в "
    "формате [doc_chunk:X].\n"
    "<</SYS>>\n\n"
    "Контекст:\n{retrieved_context}\n\n"
    "Вопрос: {instruction} [/INST]\n"
)
```

Replace `_format_prompt` (around line 160) with a public `format_prompt` and a back-compat `_format_prompt` alias:
```python
def format_prompt(
    instruction: str,
    context: str,
    *,
    retrieved_context: str = "",
    prompt_mode: str = "generic",
) -> str:
    """Return the prefix that precedes the model's response.

    ``prompt_mode='generic'`` keeps the pre-W3 behaviour. ``'rag'`` uses
    the system+context template defined in PROMPT_TEMPLATE_RAG.
    """

    if prompt_mode == "rag":
        return PROMPT_TEMPLATE_RAG.format(
            instruction=instruction,
            retrieved_context=retrieved_context or "",
        )
    return PROMPT_TEMPLATE.format(instruction=instruction, input=context or "")


def _format_prompt(instruction: str, context: str) -> str:  # pragma: no cover - back-compat
    return format_prompt(instruction, context)
```

Add the argparse flag inside `parse_args` (before `return parser.parse_args(...)`):
```python
    parser.add_argument(
        "--prompt-mode",
        choices=["generic", "rag"],
        default="generic",
        help="Prompt template selection. 'rag' expects 'retrieved_context' in dataset rows.",
    )
```

Update `_normalise_example` to return a 4-tuple and `_build_feature` to consume it:
```python
def _normalise_example(example: dict[str, Any]) -> tuple[str, str, str, str]:
    def pick(fields: tuple[str, ...]) -> str:
        for field_name in fields:
            value = example.get(field_name)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    instruction = pick(("instruction", "prompt", "question"))
    output = pick(("output", "response", "answer"))
    context = pick(("input", "context", "background"))
    retrieved_context = pick(("retrieved_context",))
    if not instruction or not output:
        raise ValueError("Dataset rows must contain instruction/prompt and output/response fields")
    return instruction, context, output, retrieved_context
```

Then in `_build_feature`, replace only the signature and the first two lines of the body (everything after `prompt_tokens = tokenizer(...)` stays as it was):
```python
def _build_feature(
    example: dict[str, Any],
    tokenizer: AutoTokenizer,
    *,
    max_seq_len: int,
    prompt_mode: str = "generic",
) -> dict[str, list[int]]:
    instruction, context, output, retrieved_context = _normalise_example(example)
    prompt_prefix = format_prompt(
        instruction,
        context,
        retrieved_context=retrieved_context,
        prompt_mode=prompt_mode,
    )
    prompt_tokens = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    # ... (the rest of the original body is unchanged)
```

Thread `prompt_mode` through `TrainingConfig` so `_load_datasets` can see it:

1. In the `@dataclass(slots=True) class TrainingConfig:` block, append a field after `logging_steps`:
   ```python
       prompt_mode: str = "generic"
   ```

2. In `_load_config(args)`, add to the kwargs passed to `TrainingConfig(...)`:
   ```python
       prompt_mode=args.prompt_mode,
   ```

3. In `_load_datasets(config, tokenizer)`, change the nested `_process` closure to forward the mode:
   ```python
       def _process(example: dict[str, Any]) -> dict[str, Any]:
           return _build_feature(
               example,
               tokenizer,
               max_seq_len=config.max_seq_len,
               prompt_mode=config.prompt_mode,
           )
   ```

- [ ] **Step 4: Run the tests to confirm they pass**

```powershell
py -3 -m pytest tests/test_train_lora_prompt_mode.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Run a broader regression to catch breakage**

```powershell
py -3 -m pytest tests/ -k "train_lora or lora" -v
```
Expected: no new failures vs. main.

- [ ] **Step 6: Commit**

```powershell
git add scripts/train_lora.py tests/test_train_lora_prompt_mode.py
git commit -m @'
feat(train-lora): add --prompt-mode {generic,rag} flag (closes G2)

New PROMPT_TEMPLATE_RAG carries a Russian system+context+question
layout that teaches the model to ground answers in retrieval and
cite sources as [doc_chunk:X]. The generic path is unchanged and
remains the default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Sprint 3 — Docs + release readiness (~30 min)

### Task 3.1: README example

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the existing W1 (synthetic Q&A) usage section**

```powershell
py -3 -c "import pathlib; print('generate_synthetic_qa' in pathlib.Path('README.md').read_text(encoding='utf-8'))"
```
Expected: `True`.

- [ ] **Step 2: Append a short W3 example immediately after the W1 example**

Add to `README.md` (right after the synthetic-qa section):
````markdown
### W3: RAG-aware fine-tuning dataset

Compose W1 seeds with retrieval to produce a 4-variant dataset that
trains the model to use retrieved context, refuse out-of-corpus
questions, hedge on partial matches, and answer closed-book when no
context is provided:

```powershell
py -3 -m scripts.generate_rag_dataset `
    --corpus var/data/kb_mvp.sqlite `
    --seeds var/data/seeds.jsonl `
    --output var/data/train_rag.jsonl `
    --target-pairs 1000

py -3 -m scripts.train_lora `
    --base-model TheBloke/some-model `
    --train var/data/train_rag.jsonl `
    --output adapters/my-rag-lora `
    --prompt-mode rag
```
````

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m @'
docs(rag-dataset): add W3 usage example to README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

### Task 3.2: Open the PR

- [ ] **Step 1: Push the branch**

```powershell
git push -u origin HEAD
```

- [ ] **Step 2: Open PR with summary**

```powershell
gh pr create --title "feat(ml): W3 RAG-aware fine-tuning dataset + train_lora flag" --body @'
## Summary

- Closes G2 (RAG-aware fine-tuning prompts) from the Pack B++ ML
  strengthening spec.
- New module `app/services/rag_dataset.py` composes W1 seeds + KB
  retrieval into 4 training variants (relevant / irrelevant /
  partial / empty) at deterministic proportions (Hamilton).
- New CLI `scripts/generate_rag_dataset.py` (mirrors
  `generate_synthetic_qa.py`).
- `scripts/train_lora.py` gains `--prompt-mode rag`; default
  `generic` keeps existing behaviour.

## Test plan

- [ ] `py -3 -m pytest tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py tests/scripts/test_generate_rag_dataset.py tests/test_train_lora_prompt_mode.py -v` — all green
- [ ] `py -3 -m pytest -q --ignore=backend` — no regressions
- [ ] `py -3 -m ruff check . && py -3 -m black --check .` — clean

## Manual smoke

```powershell
py -3 -m scripts.generate_synthetic_qa --corpus … --output seeds.jsonl --mode single
py -3 -m scripts.generate_rag_dataset --corpus … --seeds seeds.jsonl --output rag.jsonl --target-pairs 40
py -3 -m scripts.validate_dataset rag.jsonl
```

## Spec reference

`docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md` § Workstream 3.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
'@
```

---

## Acceptance criteria

- [ ] `py -3 -m pytest tests/test_rag_dataset.py tests/test_rag_dataset_proportions.py tests/scripts/test_generate_rag_dataset.py tests/test_train_lora_prompt_mode.py -v` → **all PASS**.
- [ ] `py -3 -m pytest -q --ignore=backend` — no new failures vs. main.
- [ ] `py -3 -m ruff check . && py -3 -m black --check .` — clean.
- [ ] CLI smoke against a tiny fixture produces a JSONL where every line has `retrieved_context` and `meta.variant`.
- [ ] `scripts/validate_dataset.py rag.jsonl` accepts the output.
- [ ] Variant distribution on a 100-pair run matches 70 / 15 / 10 / 5 within ±1 sample.
- [ ] `train_lora --prompt-mode rag` does NOT regress the generic path (existing LoRA tests green).
- [ ] PR description points to spec § W3 and lists which G2 metric improves (faithfulness, refusal rate — measured separately in W5 / W10).

## Out of scope (open separate issues if encountered)

- Measuring the +10pp faithfulness uplift on a held-out set → W5 RAGAS evaluation harness owns that.
- Embedding fine-tuning to make the retriever feed higher-quality context → W6.
- A UI for kicking off RAG-dataset generation → W9 Auto-Train UI.
- Replacing W1's hashing-embedder default for evaluation → that's a config concern documented in CLAUDE.md "Embedder gotcha" section.

## Estimated effort

- Sprint 1 (module + 4 variants + orchestrator): 4-5 hours.
- Sprint 2 (CLI + `train_lora.py` integration): 2-3 hours.
- Sprint 3 (docs + PR): 30 minutes.
- **Total: 6-8 hours of focused TDD work.**
