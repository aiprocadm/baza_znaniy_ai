# Pravo Reranker Phase 1 — Training (mMARCO → pravo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `kbai-reranker-ru` (rubert-tiny2 cross-encoder) in two stages — general ranking on Russian mMARCO, then domain adaptation on structurally-mined pravo pairs — and gate it on `golden_pravo_natural`.

**Architecture:** Two new dataset builders emit the existing `{query, text, teacher_score}` jsonl format; the existing `train_reranker.py` runs both stages (with one new `--init-from` flag chaining stage-1 weights into stage-2). mMARCO uses synthetic pairwise labels (pos=1.0/neg=0.0); pravo uses bge-reranker-v2-m3 teacher scores on structurally-mined hard negatives. Everything reuses the production bi-encoder store and the device-agnostic trainer (CPU smoke → GPU full run).

**Tech Stack:** Python (`py -3.13`), HuggingFace `datasets` (streaming) + `transformers`, `sentence_transformers` CrossEncoder (bge teacher), the MVP SQLite store (`app.services.kb_store`), existing eval harness (`scripts/eval_rag.py`).

**Spec:** [2026-06-17-pravo-reranker-phase1-training-design.md](../specs/2026-06-17-pravo-reranker-phase1-training-design.md)

---

## File Structure

- **Create** `scripts/build_mmarco_pairs.py` — stage-1 dataset builder: stream Russian mMARCO triples → `{query, text, teacher_score}` jsonl (synthetic binary labels). Pure helpers (`to_pairs`, `subsample_indices`) unit-tested; `datasets` import lazy.
- **Create** `scripts/build_pravo_pairs.py` — stage-2 structural miner: heading→query, article→positive, bi-encoder top-k→hard-negs, bge teacher scores, anti-leak vs golden_pravo. Reuses `heading_to_query` (build_pravo_golden) + `build_pairs`/`normalize_question` (build_rerank_dataset). Store/teacher imports lazy.
- **Modify** `scripts/train_reranker.py` — add `--init-from` (default `BASE_MODEL`); thread into `from_pretrained`. ~4 lines.
- **Create** `tests/scripts/test_build_mmarco_pairs.py` — pure-function tests, no ML deps.
- **Create** `tests/scripts/test_build_pravo_pairs.py` — pure-function tests, no ML deps.
- **Modify** `tests/scripts/test_train_reranker.py` — add `--init-from` arg-parse test.

All four scripts follow the established convention: heavy imports (`datasets`, `transformers`, `sentence_transformers`, store) are **lazy inside functions** so stub-backed unit tests never touch ML deps.

---

## Task 0: Environment — confirm `datasets` availability

**Files:** none (environment check).

- [ ] **Step 1: Check whether `datasets` is importable**

Run: `py -3.13 -c "import datasets; print(datasets.__version__)"`
Expected: either a version string (skip Step 2) or `ModuleNotFoundError` (do Step 2). As of plan authoring it is NOT installed.

- [ ] **Step 2: Install `datasets` if missing**

Run: `py -3.13 -m pip install "datasets>=2.0"`
Expected: install succeeds; re-run Step 1 and see a version string.

- [ ] **Step 3: Inspect the real Russian mMARCO triples schema**

Run:
```bash
py -3.13 -c "from datasets import load_dataset; ds=load_dataset('unicamp-dl/mmarco','russian',split='train',streaming=True); print(next(iter(ds)))"
```
Expected: one record printed. **Record the exact column names** (expected `query`, `positive`, `negative`; if the config name or columns differ, note them — Task 2's `iter_triples` must match the real keys). This is the single dataset-specific unknown; everything downstream is pure and tested.

---

## Task 1: Stage-1 dataset builder — pure core (`build_mmarco_pairs.py`)

**Files:**
- Create: `scripts/build_mmarco_pairs.py`
- Test: `tests/scripts/test_build_mmarco_pairs.py`

- [ ] **Step 1: Write the failing test for `to_pairs`**

```python
# tests/scripts/test_build_mmarco_pairs.py
"""Pure-function tests for the mMARCO stage-1 builder (no ML deps)."""

from scripts.build_mmarco_pairs import subsample_indices, to_pairs


def test_to_pairs_emits_positive_then_negative_with_binary_scores():
    rows = to_pairs("какой срок исковой давности", "три года по общему правилу", "ставка налога")
    assert rows == [
        {"query": "какой срок исковой давности", "text": "три года по общему правилу", "teacher_score": 1.0},
        {"query": "какой срок исковой давности", "text": "ставка налога", "teacher_score": 0.0},
    ]


def test_to_pairs_skips_blank_fields():
    assert to_pairs("", "pos", "neg") == []
    assert to_pairs("q", "", "neg") == []
    assert to_pairs("q", "pos", "") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_mmarco_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_mmarco_pairs'`.

- [ ] **Step 3: Write the minimal module with `to_pairs`**

```python
# scripts/build_mmarco_pairs.py
"""Build the stage-1 reranker pre-train set from Russian mMARCO (spec Phase 1 §3.1).

Stream ``unicamp-dl/mmarco`` (ru split) triples -> ``{query, text, teacher_score}``
jsonl with synthetic binary labels (positive=1.0, negative=0.0). The pairwise loss
only needs within-query ordering, so no teacher pass over mMARCO is required.
``datasets`` is imported lazily so stub-backed unit tests never load it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS_OUT = Path("var/data/rerank/mmarco_pairs.jsonl")
MMARCO_DATASET = "unicamp-dl/mmarco"
MMARCO_CONFIG = "russian"


def to_pairs(query: str, positive: str, negative: str) -> list[dict]:
    """One (query, pos, neg) triple -> two scored rows; drop triples with any
    blank field (unusable — no ordering signal)."""
    if not (query.strip() and positive.strip() and negative.strip()):
        return []
    return [
        {"query": query, "text": positive, "teacher_score": 1.0},
        {"query": query, "text": negative, "teacher_score": 0.0},
    ]
```

- [ ] **Step 4: Run to verify `to_pairs` tests pass**

Run: `py -3.13 -m pytest tests/scripts/test_build_mmarco_pairs.py -v`
Expected: PASS (both `to_pairs` tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mmarco_pairs.py tests/scripts/test_build_mmarco_pairs.py
git commit -m "feat(reranker): mMARCO stage-1 builder — to_pairs core"
```

---

## Task 2: Stage-1 builder — deterministic subsample + streaming loader + CLI

**Files:**
- Modify: `scripts/build_mmarco_pairs.py`
- Test: `tests/scripts/test_build_mmarco_pairs.py`

- [ ] **Step 1: Write the failing test for `subsample_indices`**

```python
# append to tests/scripts/test_build_mmarco_pairs.py
def test_subsample_indices_is_deterministic_and_bounded():
    a = subsample_indices(total=1000, limit=10, seed=42)
    b = subsample_indices(total=1000, limit=10, seed=42)
    assert a == b                      # deterministic
    assert len(a) == 10
    assert all(0 <= i < 1000 for i in a)
    assert len(set(a)) == 10           # no dupes


def test_subsample_indices_caps_at_total():
    assert sorted(subsample_indices(total=5, limit=999, seed=1)) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_mmarco_pairs.py::test_subsample_indices_is_deterministic_and_bounded -v`
Expected: FAIL — `ImportError: cannot import name 'subsample_indices'`.

- [ ] **Step 3: Add `subsample_indices`**

```python
# add to scripts/build_mmarco_pairs.py (after to_pairs)
import random


def subsample_indices(*, total: int, limit: int, seed: int) -> set[int]:
    """Deterministic set of up to ``limit`` distinct indices in ``[0, total)``.

    Used to pick which streamed triples to keep without materializing the whole
    (huge) dataset: we know the stream length up front only loosely, so callers
    pass a target ``total`` window and we sample positions within it.
    """
    rng = random.Random(seed)
    if limit >= total:
        return set(range(total))
    return set(rng.sample(range(total), limit))
```

- [ ] **Step 4: Run to verify subsample tests pass**

Run: `py -3.13 -m pytest tests/scripts/test_build_mmarco_pairs.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Add the lazy streaming loader + `main` (no new test — IO/ML path)**

Match the column names confirmed in Task 0 Step 3. If they are not `query`/`positive`/`negative`, edit the three `row[...]` keys accordingly.

```python
# add to scripts/build_mmarco_pairs.py
def iter_triples(limit: int, seed: int, *, window: int):
    """Yield up to ``limit`` (query, positive, negative) tuples from streamed
    Russian mMARCO, sampling positions within the first ``window`` records.
    Lazy ``datasets`` import keeps unit tests ML-free."""
    from datasets import load_dataset

    keep = subsample_indices(total=window, limit=limit, seed=seed)
    ds = load_dataset(MMARCO_DATASET, MMARCO_CONFIG, split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= window:
            break
        if i in keep:
            yield row["query"], row["positive"], row["negative"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build_mmarco_pairs")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument("--limit", type=int, default=50000,
                        help="number of triples to keep (=> 2x rows)")
    parser.add_argument("--window", type=int, default=500000,
                        help="sample within the first N streamed records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out.open("w", encoding="utf-8") as fh:
        for q, pos, neg in iter_triples(args.limit, args.seed, window=args.window):
            for row in to_pairs(q, pos, neg):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"Wrote {n_rows} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/build_mmarco_pairs.py tests/scripts/test_build_mmarco_pairs.py
git commit -m "feat(reranker): mMARCO subsample + streaming loader + CLI"
```

---

## Task 3: Stage-2 structural pravo miner — pure core (`build_pravo_pairs.py`)

**Files:**
- Create: `scripts/build_pravo_pairs.py`
- Test: `tests/scripts/test_build_pravo_pairs.py`

The miner reuses two existing, already-tested helpers — `heading_to_query` (from `scripts.build_pravo_golden`) and `build_pairs` + `normalize_question` (from `scripts.build_rerank_dataset`). This task adds only the glue that turns store documents into `(query, source_key)` pairs and applies the heading→query transform.

- [ ] **Step 1: Write the failing test for `articles_to_queries`**

```python
# tests/scripts/test_build_pravo_pairs.py
"""Pure-function tests for the structural pravo miner (no ML deps)."""

from scripts.build_pravo_pairs import articles_to_queries


def test_articles_to_queries_uses_heading_topic_and_source_key():
    docs = [
        ("gk_rf_0001.md", "Статья 196. Общий срок исковой давности", [0]),
        ("gk_rf_0002.md", "Статья 197. Специальные сроки", [0, 1]),
    ]
    assert articles_to_queries(docs) == [
        ("Общий срок исковой давности", "gk_rf_0001.md"),
        ("Специальные сроки", "gk_rf_0002.md"),
    ]


def test_articles_to_queries_skips_empty_topic():
    # A heading with no topic after the «Статья N.» prefix yields no query.
    docs = [("x.md", "Статья 5.", [0]), ("y.md", "Статья 6. Тема", [0])]
    assert articles_to_queries(docs) == [("Тема", "y.md")]
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_pravo_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_pravo_pairs'`.

- [ ] **Step 3: Write the minimal module with `articles_to_queries`**

```python
# scripts/build_pravo_pairs.py
"""Build the stage-2 reranker fine-tune set: structural pravo pairs (spec Phase 1 §3.2).

Heading topic -> query; the article is the positive; hard negatives are the
bi-encoder's top-k confusable neighbours from the pravo store; teacher scores
come from bge-reranker-v2-m3. No LLM — this removes v1/v2's CPU query-generation
bottleneck. Reuses ``heading_to_query`` (build_pravo_golden) and ``build_pairs`` /
``normalize_question`` (build_rerank_dataset). Heavy imports (store, teacher) are
lazy so stub-backed unit tests stay ML-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_pravo_golden import heading_to_query

PAIRS_OUT = Path("var/data/rerank/pravo_pairs.jsonl")
GOLDEN_PRAVO = Path("data/eval/golden_pravo.jsonl")
DEFAULT_TEACHER = "BAAI/bge-reranker-v2-m3"


def articles_to_queries(docs) -> list[tuple[str, str]]:
    """``(filename, title, [chunk_index, ...])`` rows -> ``(query, source_key)``.

    Query = heading topic (the «Статья N.» prefix stripped); source_key = the
    article's filename (threads through build_pairs for resume bookkeeping).
    Rows whose heading has no topic are dropped — they cannot be a query.
    """
    out: list[tuple[str, str]] = []
    for filename, title, _indices in docs:
        query = heading_to_query(title)
        if query:
            out.append((query, filename))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -3.13 -m pytest tests/scripts/test_build_pravo_pairs.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_pravo_pairs.py tests/scripts/test_build_pravo_pairs.py
git commit -m "feat(reranker): structural pravo miner — articles_to_queries core"
```

---

## Task 4: Stage-2 miner — anti-leak wiring + store/teacher pipeline + CLI

**Files:**
- Modify: `scripts/build_pravo_pairs.py`
- Test: `tests/scripts/test_build_pravo_pairs.py`

- [ ] **Step 1: Write the failing test for golden-query loading + leak exclusion**

```python
# append to tests/scripts/test_build_pravo_pairs.py
import json as _json

from scripts.build_pravo_pairs import load_golden_questions


def test_load_golden_questions_reads_question_field(tmp_path):
    p = tmp_path / "golden_pravo.jsonl"
    p.write_text(
        _json.dumps({"question": "Общий срок исковой давности", "relevant_chunks": ["a.md:0"]}) + "\n"
        + _json.dumps({"question": "Специальные сроки", "relevant_chunks": ["b.md:0"]}) + "\n",
        encoding="utf-8",
    )
    assert load_golden_questions(p) == frozenset(
        {"Общий срок исковой давности", "Специальные сроки"}
    )


def test_load_golden_questions_missing_file_is_empty(tmp_path):
    assert load_golden_questions(tmp_path / "nope.jsonl") == frozenset()
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_pravo_pairs.py::test_load_golden_questions_reads_question_field -v`
Expected: FAIL — `ImportError: cannot import name 'load_golden_questions'`.

- [ ] **Step 3: Add `load_golden_questions`**

```python
# add to scripts/build_pravo_pairs.py
def load_golden_questions(path: Path) -> frozenset[str]:
    """Held-out golden questions to exclude from mined training pairs (anti-leak,
    spec §3.2). Missing file => empty set (golden not built yet is not an error
    here; the leak assert in build_pairs is the real backstop)."""
    if not path.exists():
        return frozenset()
    questions = {
        json.loads(line)["question"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return frozenset(questions)
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -3.13 -m pytest tests/scripts/test_build_pravo_pairs.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Add the lazy store+teacher pipeline + `main` (no new test — ML path)**

Reuses the existing orchestration verbatim: `score_and_flush_by_chunk(queries, retrieve, golden, score_fn, *, out, k)` mines top-k via `build_pairs` (anti-leak assert included), teacher-scores each chunk, and flushes — exactly as `build_rerank_dataset.main` does (lines ~444–463). The retriever is the shared `as_retrieve(make_mvp_retriever(store))`; the teacher is a `CrossEncoder`. The ONLY difference from `build_rerank_dataset` is the query source: our structural `articles_to_queries`, not LLM generation.

```python
# add to scripts/build_pravo_pairs.py
def main(argv: list[str] | None = None) -> None:
    import logging

    parser = argparse.ArgumentParser(prog="build_pravo_pairs")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument("--golden", default=str(GOLDEN_PRAVO))
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--k", type=int, default=20, help="hard negatives mined per query")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.eval.adapter import make_mvp_retriever
    from app.services.kb_store import get_store
    from scripts.build_pravo_golden import documents_with_chunks
    from scripts.build_rerank_dataset import Pair, as_retrieve, count_rows, score_and_flush_by_chunk
    from sentence_transformers import CrossEncoder

    store = get_store()
    docs = documents_with_chunks(store)
    if not docs:
        raise SystemExit("Store is empty — run scripts.ingest_pravo first (check KB_MVP_DB_PATH).")

    queries = articles_to_queries(docs)          # [(query, source_key=filename), ...]
    golden = load_golden_questions(Path(args.golden))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # fresh run — keep source_key resume markers clean

    retrieve = as_retrieve(make_mvp_retriever(store))
    encoder = CrossEncoder(args.teacher, max_length=512)

    def _score(chunk_pairs) -> list[float]:
        scores = encoder.predict([(p.query, p.text) for p in chunk_pairs], batch_size=args.batch)
        return [float(s) for s in scores]

    new_pairs = score_and_flush_by_chunk(
        queries, retrieve, golden, _score, out=out, k=args.k
    )
    if count_rows(out) == 0:
        raise SystemExit("No pairs mined — check the corpus and golden exclusion.")
    print(f"Wrote {new_pairs} teacher-scored pairs to {out}")


if __name__ == "__main__":
    main()
```

> **Execution note:** `score_and_flush_by_chunk`, `as_retrieve`, `count_rows`, `Pair` are all real exports of `scripts/build_rerank_dataset.py` (verified at plan authoring); `make_mvp_retriever` is in `app/eval/adapter.py`. If `make_mvp_retriever`'s wrapper shape differs from what `as_retrieve` expects, mirror exactly how `build_rerank_dataset.main` (line ~429) wires them — that is the canonical working example.

- [ ] **Step 6: Run the anti-leak guard on the mined pairs (after a real run)**

Run: `py -3.13 -m scripts.check_rerank_leak --pairs var/data/rerank/pravo_pairs.jsonl --golden data/eval/golden_pravo.jsonl`
Expected: exits 0, no leaked queries reported.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_pravo_pairs.py tests/scripts/test_build_pravo_pairs.py
git commit -m "feat(reranker): pravo miner — anti-leak + store/teacher pipeline + CLI"
```

---

## Task 5: Trainer — add `--init-from` for two-stage chaining

**Files:**
- Modify: `scripts/train_reranker.py` (`BASE_MODEL` usage at line ~181; `main` argparse at line ~302; `train` signature at line ~156)
- Test: `tests/scripts/test_train_reranker.py`

- [ ] **Step 1: Write the failing test for the new arg**

```python
# append to tests/scripts/test_train_reranker.py
def test_init_from_defaults_to_base_model(monkeypatch):
    import scripts.train_reranker as tr

    captured = {}

    def fake_train(rows_train, rows_val, **kw):
        captured.update(kw)
        return {"val_pairs": 0, "val_pearson_vs_teacher": 0.0, "device": "cpu"}

    monkeypatch.setattr(tr, "train", fake_train)
    monkeypatch.setattr(tr, "load_pairs", lambda p: [
        {"query": "a", "text": "t", "teacher_score": 1.0},
        {"query": "b", "text": "u", "teacher_score": 0.0},
    ])
    tr.main(["--pairs", "x.jsonl", "--out", "o", "--epochs", "1"])
    assert captured["init_from"] == "cointegrated/rubert-tiny2"


def test_init_from_override_is_passed_through(monkeypatch):
    import scripts.train_reranker as tr

    captured = {}
    monkeypatch.setattr(tr, "train", lambda rt, rv, **kw: captured.update(kw) or
                        {"val_pairs": 0, "val_pearson_vs_teacher": 0.0, "device": "cpu"})
    monkeypatch.setattr(tr, "load_pairs", lambda p: [
        {"query": "a", "text": "t", "teacher_score": 1.0},
        {"query": "b", "text": "u", "teacher_score": 0.0},
    ])
    tr.main(["--pairs", "x.jsonl", "--out", "o", "--init-from", "var/models/stage1"])
    assert captured["init_from"] == "var/models/stage1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_train_reranker.py -k init_from -v`
Expected: FAIL — `KeyError: 'init_from'` (arg not parsed / not threaded).

- [ ] **Step 3: Thread `--init-from` through argparse, `main`, and `train`**

In `main`, add the argument (next to `--pairs`):
```python
    parser.add_argument(
        "--init-from",
        default=BASE_MODEL,
        help="start checkpoint: rubert-tiny2 (stage 1) or a stage-1 dir (stage 2)",
    )
```
Pass it into the `train(...)` call and the `meta` dict:
```python
    metrics = train(
        rows_train,
        rows_val,
        out_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        loss=args.loss,
        device=args.device,
        init_from=args.init_from,
    )
    meta = {
        "base_model": BASE_MODEL,
        "init_from": args.init_from,
        ...  # existing keys unchanged
    }
```
In `train`, add the parameter and use it for both tokenizer and model load:
```python
def train(
    rows_train: list[dict],
    rows_val: list[dict],
    *,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    loss: str = "bce",
    device: str | None = None,
    init_from: str = BASE_MODEL,
) -> dict:
    ...
    tokenizer = AutoTokenizer.from_pretrained(init_from)
    model = AutoModelForSequenceClassification.from_pretrained(init_from, num_labels=1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -3.13 -m pytest tests/scripts/test_train_reranker.py -k init_from -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the full trainer test file (no regressions)**

Run: `py -3.13 -m pytest tests/scripts/test_train_reranker.py -v`
Expected: PASS (all, including pre-existing pairing tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/train_reranker.py tests/scripts/test_train_reranker.py
git commit -m "feat(reranker): train_reranker --init-from for two-stage chaining"
```

---

## Task 6: CPU smoke run — validate the end-to-end pipeline (not quality)

**Files:** none (operational; produces untracked artifacts under `var/`).

Per spec §3.4 Step A: tiny run proving the pipeline is green before renting GPU. Run detached per the `detached-long-runs` rule; do not foreground (background tasks die ~10 min into CPU ML work).

- [ ] **Step 1: Confirm the pravo store exists (or build it)**

Run: `py -3.13 -c "from app.services.kb_store import get_store; print(len(__import__('scripts.build_pravo_golden', fromlist=['documents_with_chunks']).documents_with_chunks(get_store())))"`
Expected: a count ~ thousands. If `0` / error → first run `scripts.ingest_pravo` (see Phase 0 runbook) with the e5-small env pins.

- [ ] **Step 2: Build a 10k-triple mMARCO smoke set**

Run: `py -3.13 -m scripts.build_mmarco_pairs --limit 10000 --out var/data/rerank/mmarco_smoke.jsonl`
Expected: `Wrote 20000 rows to var/data/rerank/mmarco_smoke.jsonl`.

- [ ] **Step 3: Stage-1 smoke train (1 epoch, CPU, detached)**

Run (detached, then babysit with Monitor):
```bash
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/mmarco_smoke.jsonl \
  --out var/models/kbai-reranker-ru-stage1 --loss pairwise --epochs 1 --device cpu
```
Expected: completes; `train_meta.json` written with non-NaN `val_pearson_vs_teacher`.

- [ ] **Step 4: Build pravo pairs + Stage-2 smoke train (1 epoch, CPU, detached)**

Run:
```bash
py -3.13 -m scripts.build_pravo_pairs --out var/data/rerank/pravo_pairs.jsonl
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/pravo_pairs.jsonl \
  --out var/models/kbai-reranker-ru --init-from var/models/kbai-reranker-ru-stage1 \
  --loss pairwise --epochs 1 --lr 1e-5 --device cpu
```
Expected: both complete; stage-2 `train_meta.json` shows `init_from` = the stage-1 dir, non-NaN pearson.

- [ ] **Step 5: Gate check — three-way eval on the natural golden**

Run:
```bash
py -3.13 -m scripts.eval_rag --golden data/eval/golden_pravo_natural.jsonl   # base
KB_RERANK_MODEL=var/models/kbai-reranker-ru py -3.13 -m scripts.eval_rag --golden data/eval/golden_pravo_natural.jsonl  # student
KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3 py -3.13 -m scripts.eval_rag --golden data/eval/golden_pravo_natural.jsonl      # teacher
```
Expected: three metric blocks (hit@1 / hit@5 / recall@5 / mrr@5). **Smoke success = pipeline green + student ≥ base** (10k-pair smoke is not expected to hit the full +0.05 gate). Record numbers in the runbook.

- [ ] **Step 6: Decision — go/no-go on GPU**

If pipeline green and student ≥ base on mrr@5/hit@1 → proceed to the GPU full run (Task 7). If student < base, diagnose per spec §4 (compare stage1-only vs two-stage; check pair counts) before spending GPU hours. Record the decision in `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md`.

---

## Task 7: GPU full run + gate (operational, after smoke go)

**Files:** none (operational; runbook update).

- [ ] **Step 1: Build full datasets**

Run: `py -3.13 -m scripts.build_mmarco_pairs --limit 50000 --out var/data/rerank/mmarco_pairs.jsonl`
(pravo pairs from Task 6 Step 4 are reused — domain set is small and fixed.)

- [ ] **Step 2: Stage-1 full train on GPU**

Run (on the rented GPU box, CUDA auto-detected):
```bash
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/mmarco_pairs.jsonl \
  --out var/models/kbai-reranker-ru-stage1 --loss pairwise --epochs 2
```
Expected: completes in hours not days; non-NaN pearson.

- [ ] **Step 3: Stage-2 full train on GPU**

Run:
```bash
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/pravo_pairs.jsonl \
  --out var/models/kbai-reranker-ru --init-from var/models/kbai-reranker-ru-stage1 \
  --loss pairwise --epochs 2 --lr 1e-5
```

- [ ] **Step 4: Final gate eval (three-way)**

Run the three `eval_rag` commands from Task 6 Step 5.
Expected gate (spec §4): student beats base by **mrr@5 ≥ +0.05 OR hit@1 ≥ +0.05**, and student recall@5 ≥ base. GO → Phase 2 (quantize/latency/rollout, separate spec). NO-GO → diagnose per §4.

- [ ] **Step 5: Record verdict in the runbook and commit**

```bash
git add docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md
git commit -m "docs(reranker): Phase 1 GPU run — <GO/NO-GO> verdict + metrics"
```

---

## Self-Review notes

- **Spec coverage:** §3.1 → Tasks 1–2; §3.2 → Tasks 3–4; §3.3 (`--init-from`, stage table) → Task 5; §3.4 Step A (CPU smoke) → Task 6; §3.4 Step B (GPU) → Task 7; §4 gate → Tasks 6–7; §5 anti-leak → Task 4 (load_golden_questions + check_rerank_leak). All covered.
- **Lazy-import convention:** every ML/IO entry point (`iter_triples`, `main` in both builders, `train`) imports heavy deps inside the function — unit tests in Tasks 1/3/5 import only pure helpers.
- **Verified internal APIs:** `score_and_flush_by_chunk(queries, retrieve, golden, score_fn, *, out, k)`, `as_retrieve`, `count_rows`, `Pair` (build_rerank_dataset), `heading_to_query` / `documents_with_chunks` (build_pravo_golden), `make_mvp_retriever` (app.eval.adapter), and the `CrossEncoder(...).predict` teacher pattern were all read from source at plan authoring — Task 4 mirrors `build_rerank_dataset.main` exactly.
- **Single genuine execution-time unknown (flagged inline, not a placeholder):** the real Russian mMARCO column names — Task 0 Step 3 records them; Task 2 Step 5 adapts the three `row[...]` keys if they differ from `query`/`positive`/`negative`. This is an external-dataset fact to confirm, not undecided design.
