# Pravo Reranker Phase 1 — Training (mr-TyDi → pravo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `kbai-reranker-ru` (rubert-tiny2 cross-encoder) in two stages — general ranking on Russian mr-TyDi, then domain adaptation on structurally-mined pravo pairs — and gate it on `golden_pravo_natural`.

**Architecture:** Two new dataset builders emit the existing `{query, text, teacher_score}` jsonl format; the existing `train_reranker.py` runs both stages (with one new `--init-from` flag chaining stage-1 weights into stage-2). mr-TyDi gives natural Russian queries each with 1 positive + ~30 pre-mined hard negatives → synthetic pairwise labels (pos=1.0/neg=0.0); pravo uses bge-reranker-v2-m3 teacher scores on structurally-mined hard negatives. Everything reuses the production bi-encoder store and the device-agnostic trainer (CPU smoke → GPU full run).

**Tech Stack:** Python (`py -3.13`), HuggingFace `datasets==3.6.0` (streaming, `trust_remote_code=True` — mr-TyDi is a script dataset, unsupported in datasets 4.0+) + `transformers`, `sentence_transformers` CrossEncoder (bge teacher), the MVP SQLite store (`app.services.kb_store`), existing eval harness (`scripts/eval_rag.py`).

**Revision note (execution 2026-06-17):** stage-1 source changed from mMARCO to mr-TyDi — mMARCO's Russian split is not streamable text triples but a 5.77 GB passage-corpus ID-join. mr-TyDi russian is natural Russian (not MT), hundreds of MB, with pre-mined hard negatives. See spec §6.

**Spec:** [2026-06-17-pravo-reranker-phase1-training-design.md](../specs/2026-06-17-pravo-reranker-phase1-training-design.md)

---

## File Structure

- **Create** `scripts/build_mrtydi_pairs.py` — stage-1 dataset builder: stream Russian mr-TyDi records → `{query, text, teacher_score}` jsonl (synthetic binary labels; 1 positive + capped hard negatives per query). Pure helpers (`to_pairs`, `record_to_texts`, `take_first`) unit-tested; `datasets` import lazy.
- **Create** `scripts/build_pravo_pairs.py` — stage-2 structural miner: heading→query, article→positive, bi-encoder top-k→hard-negs, bge teacher scores, anti-leak vs golden_pravo **and** golden_pravo_natural (via canonical `load_golden`). Reuses `heading_to_query` (build_pravo_golden) + `build_pairs`/`normalize_question` (build_rerank_dataset). Store/teacher imports lazy.
- **Modify** `scripts/train_reranker.py` — add `--init-from` (default `BASE_MODEL`); thread into `from_pretrained`. ~4 lines.
- **Create** `tests/scripts/test_build_mrtydi_pairs.py` — pure-function tests, no ML deps.
- **Create** `tests/scripts/test_build_pravo_pairs.py` — pure-function tests, no ML deps.
- **Modify** `tests/scripts/test_train_reranker.py` — add `--init-from` arg-parse test.

All four scripts follow the established convention: heavy imports (`datasets`, `transformers`, `sentence_transformers`, store) are **lazy inside functions** so stub-backed unit tests never touch ML deps.

---

## Task 0: Environment — `datasets==3.6.0` + confirm mr-TyDi schema  ✅ DONE at plan authoring

**Files:** none (environment check). Completed during execution 2026-06-17 — recorded here so a fresh worker can reproduce/verify.

- [ ] **Step 1: Pin `datasets==3.6.0`** (4.0+ removed script-dataset support; mr-TyDi needs it)

Run: `py -3.13 -m pip install "datasets==3.6.0"`
Expected: `Successfully installed datasets-3.6.0`. (pandas 3.0.3 / py3.13 coexist fine; nothing else depends on `datasets`.)

- [ ] **Step 2: Confirm the real mr-TyDi russian schema**

Run:
```bash
py -3.13 -c "from datasets import load_dataset; ds=load_dataset('castorini/mr-tydi','russian',split='train',streaming=True,trust_remote_code=True); r=next(iter(ds)); print(list(r.keys())); print(len(r['positive_passages']), len(r['negative_passages']))"
```
Expected (confirmed): keys `['query_id', 'query', 'positive_passages', 'negative_passages']`; `positive_passages` = list of `{docid, text, title}` (≥1), `negative_passages` = list of `{docid, text, title}` (~30). Task 1/2 below are written to this exact schema — no remaining dataset-specific unknown.

---

## Task 1: Stage-1 dataset builder — pure core (`build_mrtydi_pairs.py`)

**Files:**
- Create: `scripts/build_mrtydi_pairs.py`
- Test: `tests/scripts/test_build_mrtydi_pairs.py`

- [ ] **Step 1: Write the failing tests for `to_pairs` and `record_to_texts`**

```python
# tests/scripts/test_build_mrtydi_pairs.py
"""Pure-function tests for the mr-TyDi stage-1 builder (no ML deps)."""

from scripts.build_mrtydi_pairs import record_to_texts, to_pairs


def test_to_pairs_emits_positive_then_negatives_with_binary_scores():
    rows = to_pairs("какой срок", "три года", ["ставка налога", "состав суда"])
    assert rows == [
        {"query": "какой срок", "text": "три года", "teacher_score": 1.0},
        {"query": "какой срок", "text": "ставка налога", "teacher_score": 0.0},
        {"query": "какой срок", "text": "состав суда", "teacher_score": 0.0},
    ]


def test_to_pairs_skips_blank_query_or_positive():
    assert to_pairs("", "pos", ["neg"]) == []
    assert to_pairs("q", "", ["neg"]) == []


def test_to_pairs_drops_blank_negatives_individually():
    rows = to_pairs("q", "pos", ["", "real neg", "   "])
    assert rows == [
        {"query": "q", "text": "pos", "teacher_score": 1.0},
        {"query": "q", "text": "real neg", "teacher_score": 0.0},
    ]


def test_record_to_texts_extracts_first_positive_and_caps_negatives():
    record = {
        "query": "вопрос",
        "positive_passages": [{"docid": "a", "text": "позитив", "title": "t"}],
        "negative_passages": [
            {"docid": "b", "text": "neg1", "title": "t"},
            {"docid": "c", "text": "neg2", "title": "t"},
            {"docid": "d", "text": "neg3", "title": "t"},
        ],
    }
    assert record_to_texts(record, max_negs=2) == ("вопрос", "позитив", ["neg1", "neg2"])


def test_record_to_texts_handles_missing_positive():
    record = {"query": "q", "positive_passages": [], "negative_passages": []}
    assert record_to_texts(record, max_negs=5) == ("q", "", [])
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_mrtydi_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_mrtydi_pairs'`.

- [ ] **Step 3: Write the minimal module**

```python
# scripts/build_mrtydi_pairs.py
"""Build the stage-1 reranker pre-train set from Russian mr-TyDi (spec Phase 1 §3.1).

Stream ``castorini/mr-tydi`` (russian) -> ``{query, text, teacher_score}`` jsonl with
synthetic binary labels (positive=1.0, negative=0.0). Each record carries 1 positive
and ~30 pre-mined hard negatives, so no teacher pass and no own negative-mining is
needed — the pairwise loss only needs within-query ordering. ``datasets`` is imported
lazily so stub-backed unit tests never load it. Requires ``datasets==3.6.0`` +
``trust_remote_code=True`` (mr-TyDi is a script dataset; datasets 4.0+ dropped it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS_OUT = Path("var/data/rerank/mrtydi_pairs.jsonl")
MRTYDI_DATASET = "castorini/mr-tydi"
MRTYDI_CONFIG = "russian"


def to_pairs(query: str, positive: str, negatives: list[str]) -> list[dict]:
    """One record -> scored rows: positive=1.0, each non-blank negative=0.0.
    A blank query or positive yields nothing (no usable ordering signal)."""
    if not (query.strip() and positive.strip()):
        return []
    rows = [{"query": query, "text": positive, "teacher_score": 1.0}]
    for neg in negatives:
        if neg.strip():
            rows.append({"query": query, "text": neg, "teacher_score": 0.0})
    return rows


def record_to_texts(record: dict, *, max_negs: int) -> tuple[str, str, list[str]]:
    """Pull (query, positive_text, [negative_texts]) from a mr-TyDi record.
    Uses the first positive passage; caps negatives at ``max_negs``."""
    query = record["query"]
    positives = record.get("positive_passages") or []
    negatives = record.get("negative_passages") or []
    positive = positives[0]["text"] if positives else ""
    neg_texts = [n["text"] for n in negatives[:max_negs]]
    return query, positive, neg_texts
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -3.13 -m pytest tests/scripts/test_build_mrtydi_pairs.py -v`
Expected: PASS (all five tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mrtydi_pairs.py tests/scripts/test_build_mrtydi_pairs.py
git commit -m "feat(reranker): mr-TyDi stage-1 builder — to_pairs/record_to_texts core"
```

---

## Task 2: Stage-1 builder — `take_first` + streaming loader + CLI

mr-TyDi russian train is small (~5k queries), so we keep the first `--limit` records (deterministic, correct for any dataset size) rather than windowed sampling. `take_first` is a pure generator (testable); `iter_records` wraps it with the lazy `datasets` load.

**Files:**
- Modify: `scripts/build_mrtydi_pairs.py`
- Test: `tests/scripts/test_build_mrtydi_pairs.py`

- [ ] **Step 1: Write the failing test for `take_first`**

```python
# append to tests/scripts/test_build_mrtydi_pairs.py
from scripts.build_mrtydi_pairs import take_first


def test_take_first_yields_at_most_limit_in_order():
    assert list(take_first(["a", "b", "c", "d"], 2)) == ["a", "b"]


def test_take_first_handles_fewer_than_limit():
    assert list(take_first(["a", "b"], 10)) == ["a", "b"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_mrtydi_pairs.py -k take_first -v`
Expected: FAIL — `ImportError: cannot import name 'take_first'`.

- [ ] **Step 3: Add `take_first`**

```python
# add to scripts/build_mrtydi_pairs.py (after record_to_texts)
from typing import Iterable, Iterator


def take_first(records: Iterable, limit: int) -> Iterator:
    """Yield the first ``limit`` items of an iterable (deterministic subsample of
    a streaming dataset). ``limit <= 0`` yields nothing."""
    for i, record in enumerate(records):
        if i >= limit:
            break
        yield record
```

- [ ] **Step 4: Run to verify take_first tests pass**

Run: `py -3.13 -m pytest tests/scripts/test_build_mrtydi_pairs.py -v`
Expected: PASS (all seven tests).

- [ ] **Step 5: Add the lazy streaming loader + `main` (no new test — IO/ML path)**

Matches the mr-TyDi schema confirmed in Task 0 Step 2 (`query`, `positive_passages`, `negative_passages`).

```python
# add to scripts/build_mrtydi_pairs.py
def iter_records(limit: int, *, max_negs: int):
    """Yield up to ``limit`` (query, positive, [negatives]) tuples from streamed
    Russian mr-TyDi. Lazy ``datasets`` import keeps unit tests ML-free."""
    from datasets import load_dataset

    ds = load_dataset(
        MRTYDI_DATASET, MRTYDI_CONFIG, split="train",
        streaming=True, trust_remote_code=True,
    )
    for record in take_first(ds, limit):
        yield record_to_texts(record, max_negs=max_negs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build_mrtydi_pairs")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument("--limit", type=int, default=10000,
                        help="number of queries to keep (dataset has ~5k; >size = all)")
    parser.add_argument("--negs", type=int, default=10,
                        help="hard negatives kept per query (mr-TyDi has ~30)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out.open("w", encoding="utf-8") as fh:
        for query, positive, negatives in iter_records(args.limit, max_negs=args.negs):
            for row in to_pairs(query, positive, negatives):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"Wrote {n_rows} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add scripts/build_mrtydi_pairs.py tests/scripts/test_build_mrtydi_pairs.py
git commit -m "feat(reranker): mr-TyDi take_first + streaming loader + CLI"
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

> **CRITICAL — real golden format.** The committed golden JSONL (`data/eval/golden_pravo*.jsonl`) is the `app.eval.dataset.GoldenItem` format: the question lives under the **`instruction`** key (NOT `question`), with `relevant_chunks` nested under `meta`. Read it with the canonical `load_golden` reader (which maps `instruction` → `.question`), exactly as `build_rerank_dataset.main` does — do NOT hand-roll a `["question"]` lookup (it crashes with `KeyError` on the real file). `app.eval.dataset` is a pure, fast import (no ML), and `build_pravo_golden` already imports from it at module top, so a top-level import here is consistent.

```python
# append to tests/scripts/test_build_pravo_pairs.py
import json as _json

from scripts.build_pravo_pairs import load_golden_questions


def test_load_golden_questions_reads_instruction_field(tmp_path):
    p = tmp_path / "golden_pravo.jsonl"
    p.write_text(
        _json.dumps({"instruction": "Общий срок исковой давности", "meta": {"relevant_chunks": ["a.md:0"]}}) + "\n"
        + _json.dumps({"instruction": "Специальные сроки", "meta": {"relevant_chunks": ["b.md:0"]}}) + "\n",
        encoding="utf-8",
    )
    assert load_golden_questions(p) == frozenset(
        {"Общий срок исковой давности", "Специальные сроки"}
    )


def test_load_golden_questions_missing_file_is_empty(tmp_path):
    assert load_golden_questions(tmp_path / "nope.jsonl") == frozenset()
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3.13 -m pytest tests/scripts/test_build_pravo_pairs.py::test_load_golden_questions_reads_instruction_field -v`
Expected: FAIL — `ImportError: cannot import name 'load_golden_questions'`.

- [ ] **Step 3: Add `load_golden_questions` + the natural-golden constant + the `load_golden` import**

Add to the module's top imports (next to `from scripts.build_pravo_golden import heading_to_query`): `from app.eval.dataset import load_golden`. Add the constant `GOLDEN_PRAVO_NATURAL = Path("data/eval/golden_pravo_natural.jsonl")` next to `GOLDEN_PRAVO`. Then:

```python
# add to scripts/build_pravo_pairs.py
def load_golden_questions(path: Path) -> frozenset[str]:
    """Held-out golden questions to exclude from mined training pairs (anti-leak,
    spec §3.2). Missing file => empty set (golden not built yet is not an error
    here; the leak assert in build_pairs is the real backstop). Reads the canonical
    GoldenItem JSONL (``instruction`` field) via ``load_golden`` — NOT a hand-rolled
    key, which previously crashed on the real format."""
    if not path.exists():
        return frozenset()
    return frozenset(item.question for item in load_golden(path))
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
    parser.add_argument("--golden-natural", default=str(GOLDEN_PRAVO_NATURAL))
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
    # Exclude BOTH golden sets (spec §3.2): the structural auto-golden AND the
    # natural-language eval golden — leaking the latter into training contaminates
    # the Phase 1 decision gate.
    golden = load_golden_questions(Path(args.golden)) | load_golden_questions(
        Path(args.golden_natural)
    )

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

- [ ] **Step 2: Build a mr-TyDi smoke set**

Run: `py -3.13 -m scripts.build_mrtydi_pairs --limit 2000 --negs 8 --out var/data/rerank/mrtydi_smoke.jsonl`
Expected: `Wrote <N> rows to var/data/rerank/mrtydi_smoke.jsonl` (≈ 2000 × (1+8) ≈ 18000 rows, fewer if the dataset is smaller).

- [ ] **Step 3: Stage-1 smoke train (1 epoch, CPU, detached)**

Run (detached, then babysit with Monitor):
```bash
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/mrtydi_smoke.jsonl \
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

Run: `py -3.13 -m scripts.build_mrtydi_pairs --limit 100000 --negs 20 --out var/data/rerank/mrtydi_pairs.jsonl`
(`--limit` above dataset size keeps all ~5k queries; raise `--negs` toward the ~30 available for more pairs. pravo pairs from Task 6 Step 4 are reused — domain set is small and fixed.)

- [ ] **Step 2: Stage-1 full train on GPU**

Run (on the rented GPU box, CUDA auto-detected):
```bash
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/mrtydi_pairs.jsonl \
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
- **Lazy-import convention:** every ML/IO entry point (`iter_records`, `main` in both builders, `train`) imports heavy deps inside the function — unit tests in Tasks 1/3/5 import only pure helpers.
- **Verified internal APIs:** `score_and_flush_by_chunk(queries, retrieve, golden, score_fn, *, out, k)`, `as_retrieve`, `count_rows`, `Pair` (build_rerank_dataset), `heading_to_query` / `documents_with_chunks` (build_pravo_golden), `make_mvp_retriever` (app.eval.adapter), and the `CrossEncoder(...).predict` teacher pattern were all read from source at plan authoring — Task 4 mirrors `build_rerank_dataset.main` exactly.
- **External dataset confirmed at execution:** mr-TyDi russian schema (`query`, `positive_passages[{docid,text,title}]`, `negative_passages[…]`) verified live (Task 0 Step 2); Tasks 1–2 are written to it. `datasets==3.6.0` + `trust_remote_code=True` required (script dataset). No remaining dataset unknowns.
