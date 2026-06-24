# Pravo Reranker Quality Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock in the measured `bge-reranker-v2-m3` domain win on `golden_pravo_natural` (+0.111 hit@1 / +0.085 mrr@5 over base) behind a deterministic offline CI gate (absolute floors + teacher-over-base deltas), plus a marked live integration check.

**Architecture:** Mirror the repo's established frozen-gate pattern (`tests/test_eval_frozen.py` + `data/eval/ci_thresholds.json`). A one-time offline script runs the real base and teacher pipelines and freezes per-query ranked chunk-key lists into a committed fixture. The CI test recomputes metrics from those frozen lists with pure functions (no model in CI). A `@pytest.mark.integration` test re-validates against the live reranker before any refreeze.

**Tech Stack:** Python, pytest, existing `app/eval/*` harness (`adapter.py`, `dataset.py`, `metrics.py`, `retrieval_eval.py`), `app/services/kb_store`, `app/services/kb_rerank`, sentence-transformers (offline only).

**Spec:** `docs/superpowers/specs/2026-06-25-pravo-reranker-trackA-quality-gate-design.md`

**Conventions reminder (this repo, Windows):** run Python via `py -3.13` (not bare `py -3`). No venv. Lint/style: `py -3.13 -m ruff check .` and `py -3.13 -m black --check .`. Comments in code in Russian; chat explanations in Russian.

---

## File Structure

- Create: `app/eval/pravo_gate.py` — pure gate logic (aggregate one frozen side; compute floor+delta failures). No I/O, no model.
- Create: `tests/test_pravo_gate.py` — deterministic unit tests for the pure logic (happy + 2 edges).
- Create: `scripts/freeze_pravo_eval.py` — one-time offline freezer (real models) → writes the fixture.
- Create (generated): `data/eval/frozen_pravo_natural.json` — committed frozen rankings + measured numbers.
- Create: `data/eval/ci_thresholds_pravo.json` — pravo floors + min deltas (separate from the public file).
- Create: `tests/test_eval_frozen_pravo.py` — THE GATE: sig check + floors/deltas from frozen fixture (no model).
- Create: `tests/test_pravo_rerank_integration.py` — `@pytest.mark.integration` live re-validation.
- Modify: `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md` — append a "refreeze the gate" note + latency-deferred note.

---

## Task 0: Branch setup

**Files:** none (git only).

- [ ] **Step 1: Create the feature branch off main and bring the spec + plan docs**

The spec was committed on `chore/reranker-stage1-deepen` as commit `5028244`. This plan file is currently untracked on disk and will carry across the checkout.

Run:
```bash
git checkout main
git checkout -b feat/pravo-reranker-quality-gate
git cherry-pick 5028244
git add docs/superpowers/plans/2026-06-25-pravo-reranker-quality-gate.md
git commit -m "docs(eval): Track A implementation plan for pravo reranker gate"
```
Expected: branch `feat/pravo-reranker-quality-gate` with two doc commits; `git status` clean.

> If `git cherry-pick 5028244` reports the commit is unreachable (e.g. branch was rebased), instead copy the spec file from the other branch: `git checkout chore/reranker-stage1-deepen -- docs/superpowers/specs/2026-06-25-pravo-reranker-trackA-quality-gate-design.md` then commit it.

---

## Task 1: Pure gate logic module

**Files:**
- Create: `app/eval/pravo_gate.py`
- Test: `tests/test_pravo_gate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pravo_gate.py`:
```python
"""Юнит-тесты чистой логики frozen-гейта pravo (без модели, детерминированно)."""

from __future__ import annotations

from app.eval.pravo_gate import aggregate_side, gate_failures


# Два вопроса, по три релевантных ключа. base ставит релевант на 2-ю позицию,
# teacher — на 1-ю, поэтому hit@1 teacher = 1.0, base = 0.0.
_ITEMS = [
    {
        "relevant": ["a:0"],
        "base_ranked": ["x:0", "a:0", "y:0"],
        "teacher_ranked": ["a:0", "x:0", "y:0"],
    },
    {
        "relevant": ["b:0"],
        "base_ranked": ["z:0", "b:0", "w:0"],
        "teacher_ranked": ["b:0", "z:0", "w:0"],
    },
]


def test_aggregate_side_computes_hit_at_1() -> None:
    base = aggregate_side(_ITEMS, "base_ranked")
    teacher = aggregate_side(_ITEMS, "teacher_ranked")
    assert base["hit@1"] == 0.0
    assert teacher["hit@1"] == 1.0
    # base нашёл релевант на позиции 2 → mrr@5 = 0.5; teacher на 1 → 1.0
    assert base["mrr@5"] == 0.5
    assert teacher["mrr@5"] == 1.0


def test_gate_passes_when_floors_and_deltas_met() -> None:
    base = {"hit@1": 0.0, "mrr@5": 0.5}
    teacher = {"hit@1": 1.0, "mrr@5": 1.0}
    thresholds = {
        "teacher_floors": {"hit@1": 0.84, "mrr@5": 0.86},
        "min_delta_over_base": {"hit@1": 0.05, "mrr@5": 0.04},
    }
    assert gate_failures(base, teacher, thresholds) == []


def test_gate_flags_floor_violation() -> None:
    base = {"hit@1": 0.0}
    teacher = {"hit@1": 0.80}  # ниже floor 0.84
    thresholds = {"teacher_floors": {"hit@1": 0.84}, "min_delta_over_base": {}}
    failures = gate_failures(base, teacher, thresholds)
    assert len(failures) == 1 and "floor" in failures[0]


def test_gate_flags_delta_violation() -> None:
    base = {"hit@1": 0.85}
    teacher = {"hit@1": 0.87}  # выше floor, но дельта 0.02 < 0.05
    thresholds = {
        "teacher_floors": {"hit@1": 0.84},
        "min_delta_over_base": {"hit@1": 0.05},
    }
    failures = gate_failures(base, teacher, thresholds)
    assert len(failures) == 1 and "delta" in failures[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/test_pravo_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval.pravo_gate'`.

- [ ] **Step 3: Write the module**

Create `app/eval/pravo_gate.py`:
```python
"""Чистая логика frozen-гейта качества reranker'а на корпусе права.

Без I/O и без модели: работает по уже замороженным спискам ранжированных
chunk-ключей, поэтому гейт исполняется детерминированно в CI без загрузки
``bge-reranker-v2-m3``. Две проверки: метрики teacher не ниже абсолютных floors
и превосходство teacher над base не меньше зафиксированных дельт.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.eval.metrics import RETRIEVAL_KS, aggregate, score_item

# Допуск на float-сравнения (зеркалит публичный гейт в test_eval_frozen.py).
EPS = 1e-9


def aggregate_side(
    items: Sequence[Mapping[str, Any]],
    side: str,
    ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[str, float]:
    """Агрегировать метрики ретривала для одной замороженной стороны.

    ``side`` — ключ списка ранжирования в каждом item ("base_ranked" /
    "teacher_ranked"); ``item["relevant"]`` — релевантные chunk-ключи.
    """
    rows = [score_item(it["relevant"], it[side], ks) for it in items]
    return aggregate(rows)


def gate_failures(
    base: Mapping[str, float],
    teacher: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Вернуть список человекочитаемых нарушений; пустой список = гейт пройден."""
    failures: list[str] = []
    for metric, floor in thresholds.get("teacher_floors", {}).items():
        got = float(teacher.get(metric, 0.0))
        if got + EPS < float(floor):
            failures.append(f"teacher {metric}={got:.4f} below floor {floor}")
    for metric, dmin in thresholds.get("min_delta_over_base", {}).items():
        delta = float(teacher.get(metric, 0.0)) - float(base.get(metric, 0.0))
        if delta + EPS < float(dmin):
            failures.append(f"teacher-over-base {metric} delta={delta:.4f} below min {dmin}")
    return failures
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_pravo_gate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint + commit**

Run: `py -3.13 -m ruff check app/eval/pravo_gate.py tests/test_pravo_gate.py && py -3.13 -m black app/eval/pravo_gate.py tests/test_pravo_gate.py`
Then:
```bash
git add app/eval/pravo_gate.py tests/test_pravo_gate.py
git commit -m "feat(eval): pure frozen-gate logic for pravo reranker (floors + deltas)"
```

---

## Task 2: Offline freeze script (produces the fixture)

**Files:**
- Create: `scripts/freeze_pravo_eval.py`
- Generated: `data/eval/frozen_pravo_natural.json`

This task runs the REAL models once (offline, local). The teacher pass over 36 queries is ~1–2 min plus model load; it fits inside this machine's ~10-min background-task window. The script is idempotent (overwrites the fixture each run).

- [ ] **Step 1: Write the freeze script**

Create `scripts/freeze_pravo_eval.py`:
```python
"""Заморозить ранжирования base и teacher на golden_pravo_natural для офлайн-гейта.

Гоняет ОБА пайплайна один раз реальными моделями и пишет коммитимый фикстур с
ранжированными chunk-ключами на каждый вопрос, чтобы tests/test_eval_frozen_pravo.py
пересчитывал метрики детерминированно, не загружая bge-reranker-v2-m3 в CI.

Запуск (Windows, эмбеддер st обязателен — иначе хэш-эмбеддер даёт мусор):
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.freeze_pravo_eval
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_STORE = "var/data/pravo_public.sqlite"
DEFAULT_GOLDEN = "data/eval/golden_pravo_natural.jsonl"
DEFAULT_OUT = "data/eval/frozen_pravo_natural.json"
_HEADLINE = ("hit@1", "hit@3", "mrr@5", "recall@10")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    # Стор выбирается через env до первого get_store(); сбрасываем кэш стора.
    os.environ["KB_MVP_DB_PATH"] = str(Path(args.store))
    from app.services.kb_store import get_store, reset_default_store

    reset_default_store()
    store = get_store()

    from app.eval.adapter import (
        compute_signature,
        make_mvp_reranking_retriever,
        make_mvp_retriever,
    )
    from app.eval.dataset import load_golden
    from app.eval.pravo_gate import aggregate_side

    golden = load_golden(Path(args.golden))
    if not golden:
        raise SystemExit(f"empty golden: {args.golden}")

    base = make_mvp_retriever(store)
    teacher = make_mvp_reranking_retriever(store)  # enabled форсится, модель = bge по умолчанию

    items: list[dict[str, object]] = []
    for it in golden:
        items.append(
            {
                "relevant": list(it.relevant_chunks),
                "base_ranked": [h.chunk_key for h in base(it.question, args.top_k)],
                "teacher_ranked": [h.chunk_key for h in teacher(it.question, args.top_k)],
            }
        )

    measured = {
        "base": aggregate_side(items, "base_ranked"),
        "teacher": aggregate_side(items, "teacher_ranked"),
    }
    fixture = {"_sig": compute_signature(store).to_dict(), "_measured": measured, "items": items}
    Path(args.out).write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"wrote {args.out}  n={len(items)}  store={args.store}")
    for metric in _HEADLINE:
        b = measured["base"][metric]
        t = measured["teacher"][metric]
        print(f"  {metric:10s} base={b:.3f} teacher={t:.3f} delta={t - b:+.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the freeze offline**

Run: `KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.freeze_pravo_eval`
Expected: prints `wrote data/eval/frozen_pravo_natural.json n=36 ...` and a table where every headline metric has `teacher` ≥ `base` and `delta` positive for hit@1/hit@3/mrr@5. Roughly: hit@1 base≈0.78 teacher≈0.89; mrr@5 base≈0.83 teacher≈0.92.

> If `delta` is negative or teacher ≈ base, STOP — the reranker is not actually running (check `KB_EMBEDDINGS_BACKEND=st`, that `bge-reranker-v2-m3` loaded, and that `make_mvp_reranking_retriever` forced `enabled=True`). Do not proceed to thresholds until the win reproduces.

- [ ] **Step 3: Sanity-check the fixture sig against the golden sig**

Run: `py -3.13 -c "import json; f=json.load(open('data/eval/frozen_pravo_natural.json',encoding='utf-8')); g=json.load(open('data/eval/golden_pravo_natural.sig.json',encoding='utf-8')); print('SIG MATCH' if f['_sig']==g else ('MISMATCH', f['_sig'], g))"`
Expected: `SIG MATCH`.

> If MISMATCH, the freeze store differs from the store the golden was built against. Investigate which store has `doc_count=6141, max_chunk_id=14231, embedder_name=st, dim=384` before committing — a mismatched fixture makes the gate meaningless.

- [ ] **Step 4: Commit the script + fixture**

Run: `py -3.13 -m ruff check scripts/freeze_pravo_eval.py && py -3.13 -m black scripts/freeze_pravo_eval.py`
Then:
```bash
git add scripts/freeze_pravo_eval.py data/eval/frozen_pravo_natural.json
git commit -m "feat(eval): freeze script + frozen base/teacher rankings for pravo gate"
```

---

## Task 3: Thresholds file

**Files:**
- Create: `data/eval/ci_thresholds_pravo.json`

- [ ] **Step 1: Derive the numbers from the freeze output**

From Task 2's printed table, set:
- `teacher_floors` = teacher metric rounded DOWN to ~0.05 below measured (absorbs cross-platform tie-flips).
- `min_delta_over_base` = measured delta minus a safety margin (~half the delta).

With the expected artifact numbers (teacher hit@1≈0.889, mrr@5≈0.917, recall@10≈0.707; deltas ≈ +0.111 / +0.085) this yields the values below. If your fresh freeze differs materially, adjust to keep floors ~0.05 below measured and deltas with a clear margin — but never set a floor ABOVE the measured value.

Create `data/eval/ci_thresholds_pravo.json`:
```json
{
  "_comment": "Pravo retrieval floors + min teacher-over-base deltas for the frozen pravo gate (tests/test_eval_frozen_pravo.py). Measured on golden_pravo_natural by scripts/freeze_pravo_eval.py. Floors sit ~0.05 below measured teacher; deltas sit below the measured teacher-minus-base gap. Raise (never lower) only after a real re-measure. Separate from ci_thresholds.json, which is scoped to golden_public.",
  "_measured_2026_06_25": {
    "base": { "hit@1": 0.778, "hit@3": 0.861, "mrr@5": 0.832, "recall@10": 0.694 },
    "teacher": { "hit@1": 0.889, "hit@3": 0.944, "mrr@5": 0.917, "recall@10": 0.707 }
  },
  "teacher_floors": { "hit@1": 0.84, "hit@3": 0.89, "mrr@5": 0.86, "recall@10": 0.65 },
  "min_delta_over_base": { "hit@1": 0.05, "mrr@5": 0.04 }
}
```

- [ ] **Step 2: Commit**

```bash
git add data/eval/ci_thresholds_pravo.json
git commit -m "feat(eval): pravo gate thresholds (floors + min deltas)"
```

---

## Task 4: The frozen gate test (CI, no model)

**Files:**
- Test: `tests/test_eval_frozen_pravo.py`

- [ ] **Step 1: Write the test**

Create `tests/test_eval_frozen_pravo.py`:
```python
"""Офлайн frozen-гейт качества reranker'а на корпусе права.

Детерминированный, БЕЗ модели: грузит замороженные ранжирования base/teacher,
пересчитывает метрики и ассертит абсолютные floors + дельты teacher-over-base из
data/eval/ci_thresholds_pravo.json. Зеркалит публичный гейт в test_eval_frozen.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.dataset import read_signature
from app.eval.pravo_gate import aggregate_side, gate_failures

FROZEN = Path("data/eval/frozen_pravo_natural.json")
GOLDEN = Path("data/eval/golden_pravo_natural.jsonl")
THRESHOLDS = Path("data/eval/ci_thresholds_pravo.json")


def _load_frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_frozen_sig_matches_golden() -> None:
    """Заморозка должна быть построена против того же корпуса, что и golden."""
    frozen = _load_frozen()
    gold_sig = read_signature(GOLDEN)
    assert gold_sig is not None, "golden_pravo_natural has no .sig.json"
    assert frozen["_sig"] == gold_sig.to_dict(), (
        "frozen fixture sig drift — re-run scripts/freeze_pravo_eval.py"
    )


def test_pravo_teacher_meets_floors_and_beats_base() -> None:
    """ГЕЙТ: teacher ≥ floors И teacher − base ≥ min deltas на замороженном наборе."""
    frozen = _load_frozen()
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    items = frozen["items"]
    assert items, "frozen fixture has no items"

    base = aggregate_side(items, "base_ranked")
    teacher = aggregate_side(items, "teacher_ranked")
    failures = gate_failures(base, teacher, thresholds)
    assert not failures, f"pravo reranker gate failed: {failures}"
```

- [ ] **Step 2: Run the gate**

Run: `py -3.13 -m pytest tests/test_eval_frozen_pravo.py -v`
Expected: 2 passed (`test_frozen_sig_matches_golden`, `test_pravo_teacher_meets_floors_and_beats_base`).

> If `test_pravo_teacher_meets_floors_and_beats_base` fails, the thresholds in Task 3 are stricter than the frozen measurement — reconcile them with the actual `_measured` numbers (floors must be ≤ measured teacher; deltas ≤ measured gap).

- [ ] **Step 3: Commit**

```bash
git add tests/test_eval_frozen_pravo.py
git commit -m "test(eval): frozen pravo reranker gate (floors + teacher-over-base deltas)"
```

---

## Task 5: Live integration re-validation test

**Files:**
- Test: `tests/test_pravo_rerank_integration.py`

- [ ] **Step 1: Write the marked integration test**

Create `tests/test_pravo_rerank_integration.py`:
```python
"""Живой integration-чек reranker'а права (вне дефолтного CI).

Гоняет настоящий bge-reranker-v2-m3 по реальному стору и ассертит те же floors +
дельты, что и frozen-гейт. Это то, что запускают перед рефризом фикстуры:
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m pytest -m integration -k pravo_rerank
Громко skip'ается, если стор/модель/эмбеддер недоступны.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.eval.dataset import load_golden
from app.eval.pravo_gate import aggregate_side, gate_failures

STORE = Path("var/data/pravo_public.sqlite")
GOLDEN = Path("data/eval/golden_pravo_natural.jsonl")
THRESHOLDS = Path("data/eval/ci_thresholds_pravo.json")


@pytest.mark.integration
def test_pravo_teacher_beats_base_live() -> None:
    if not STORE.exists():
        pytest.skip(f"pravo store missing: {STORE}")
    if os.environ.get("KB_EMBEDDINGS_BACKEND", "st") != "st":
        pytest.skip("KB_EMBEDDINGS_BACKEND is not 'st' — would score with hashing embedder")

    os.environ["KB_MVP_DB_PATH"] = str(STORE)
    try:
        from app.services.kb_store import get_store, reset_default_store

        reset_default_store()
        store = get_store()
        from app.eval.adapter import make_mvp_reranking_retriever, make_mvp_retriever
    except Exception as exc:  # noqa: BLE001 — любая проблема загрузки = громкий skip
        pytest.skip(f"eval stack unavailable: {exc}")

    golden = load_golden(GOLDEN)
    assert golden, "golden_pravo_natural is empty"

    base_r = make_mvp_retriever(store)
    teacher_r = make_mvp_reranking_retriever(store)
    items = [
        {
            "relevant": list(it.relevant_chunks),
            "base_ranked": [h.chunk_key for h in base_r(it.question, 10)],
            "teacher_ranked": [h.chunk_key for h in teacher_r(it.question, 10)],
        }
        for it in golden
    ]

    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    base = aggregate_side(items, "base_ranked")
    teacher = aggregate_side(items, "teacher_ranked")
    failures = gate_failures(base, teacher, thresholds)
    assert not failures, f"LIVE pravo reranker gate failed: {failures}"
```

- [ ] **Step 2: Verify it is excluded from the default run, then run it explicitly**

Run (default — should NOT collect it): `py -3.13 -m pytest tests/test_pravo_rerank_integration.py -v`
Expected: `1 deselected` or `no tests ran` (the default `-m "not integration"` posture; if the repo's default addopts do not deselect, it may run — that's acceptable as long as the live run passes).

Run (explicit live): `KB_EMBEDDINGS_BACKEND=st py -3.13 -m pytest -m integration -k pravo_rerank -v`
Expected: 1 passed (or a loud skip if the store/model is unavailable on this machine).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pravo_rerank_integration.py
git commit -m "test(eval): live integration re-validation of pravo reranker gate"
```

---

## Task 6: Docs — refreeze runbook note + latency-deferred

**Files:**
- Modify: `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md`

- [ ] **Step 1: Append a runbook section**

Append to the end of `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md`:
```markdown

## Quality gate (Track A, 2026-06-25)

The teacher reranker win is now guarded by an offline frozen gate.

- **Gate (runs in default CI, no model):** `tests/test_eval_frozen_pravo.py` —
  recomputes metrics from `data/eval/frozen_pravo_natural.json` and asserts the
  floors + teacher-over-base deltas in `data/eval/ci_thresholds_pravo.json`.
- **Refreeze (after corpus/golden/model change):**
  `KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.freeze_pravo_eval`, then re-run
  the live check `py -3.13 -m pytest -m integration -k pravo_rerank` before committing.
- **Latency is deliberately OUT of scope here.** Teacher CPU p95 ≈ 1.2 s for 20
  candidates (budget 200 ms) is a known, deferred item — see
  `scripts/quantize_reranker.py` notes (ONNX Runtime / fewer candidates / revised
  budget). Track A guards quality only.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md
git commit -m "docs(eval): runbook note for pravo gate refreeze + latency deferred"
```

---

## Task 7: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full lint + style**

Run: `py -3.13 -m ruff check . && py -3.13 -m black --check .`
Expected: no errors; black reports "would reformat 0 files".

- [ ] **Step 2: Run the new tests + a relevant slice**

Run: `py -3.13 -m pytest tests/test_pravo_gate.py tests/test_eval_frozen_pravo.py -v`
Expected: 6 passed.

- [ ] **Step 3: Confirm no regression in the broader eval suite**

Run: `py -3.13 -m pytest tests/ -k "eval or frozen or rerank" -m "not integration"`
Confirm pass/fail via exit code (piping drops the summary line): append `; echo "EXIT=$?"` and expect `EXIT=0`.

- [ ] **Step 4: mypy on the one new typed module**

Run: `py -3.13 -m mypy app/eval/pravo_gate.py`
Expected: no NEW errors attributable to this file (the repo carries a pre-existing baseline; judge by errors on the touched file's own lines).

- [ ] **Step 5: Final summary (no commit needed)**

Confirm the branch `feat/pravo-reranker-quality-gate` contains: gate logic + tests, freeze script + fixture, thresholds, gate test, integration test, runbook + spec + plan docs. Acceptance criteria from the spec are met: fresh measure committed, frozen gate green in default pytest (floors + deltas), live integration test present.

---

## Self-Review (filled by plan author)

- **Spec coverage:** §2 acceptance #1 (fresh measure committed) → Task 2/3. #2 (frozen gate green, floors+deltas) → Task 1/4. #3 (marked live integration) → Task 5. #4 (ruff/black/pytest green, floors not lowered) → Task 7 (new file `ci_thresholds_pravo.json`, existing floors untouched). Latency anti-scope → documented in Task 6. ✓
- **Placeholder scan:** thresholds numbers are concrete (artifact-derived) with an explicit "adjust only if fresh measure differs" rule — not a TODO. All code blocks complete. ✓
- **Type consistency:** `aggregate_side(items, side, ks)` and `gate_failures(base, teacher, thresholds)` signatures identical across module, unit tests, gate test, and integration test. Fixture keys (`relevant`, `base_ranked`, `teacher_ranked`, `_sig`, `_measured`, `items`) consistent between freeze script (writer) and both tests (readers). ✓
