# Own Reranker Distillation (`kbai-reranker-ru`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the project's own ~29M cross-encoder reranker by distilling `BAAI/bge-reranker-v2-m3` into `cointegrated/rubert-tiny2`, gated by `golden_public` metrics and a CPU latency budget.

**Architecture:** Three new scripts: a dataset builder (synthetic queries → production bi-encoder candidates → teacher scores → `var/data/rerank/pairs.jsonl`), a trainer (HF `AutoModelForSequenceClassification`, BCE on soft teacher labels → `var/models/kbai-reranker-ru/`), and a latency bench. Evaluation reuses the existing harness verbatim: `scripts/eval_rag.py run --rerank` with `KB_RERANK_MODEL` pointed at the trained directory (`make_mvp_reranking_retriever` already exists in `app/eval/adapter.py`). Zero production-code changes.

**Tech Stack:** Python 3.13 (`py -3.13`, no venv — see CLAUDE.md), torch CPU + transformers + sentence-transformers (already installed), existing W1 synthetic-QA generator, existing eval harness.

**Spec:** `docs/superpowers/specs/2026-06-10-own-reranker-distillation-design.md`

---

## Preconditions (read before Task 1)

- **PR2 dependency:** `data/eval/corpus_public/` and `data/eval/golden_public.jsonl` live on `feat/eval-public-corpus` (PR2). This plan's branch MUST be cut from `main` **after PR2 merges** (or stacked on `feat/eval-public-corpus` if PR2 is still open — rebase after merge).
- Branch: `git checkout -b feat/own-reranker-distill <base>`.
- Heavy ML runs only at Task 6 (manual). Tasks 1–5 are stub-safe: **no module-level imports of torch/transformers/sentence_transformers in the new scripts** — lazy imports inside functions, same pattern as `app/services/kb_rerank.py:_get_reranker`.
- Tests live in `tests/scripts/` (directory exists since PR2). Run a single file with `py -3.13 -m pytest tests/scripts/test_build_rerank_dataset.py -v`.
- Long local runs (query generation on the ~4–6 tok/s GGUF, teacher scoring on CPU) follow the detached pattern: `Start-Process` + persistent monitor (memory: detached-long-runs). Harness background tasks die ~10 min in — do not use them for these runs.

### Data-flow summary

```
corpus_public/*.md ──(scripts.build_public_corpus ingest)──> SQLite store (KB_MVP_DB_PATH)
store chunks ──(W1 SyntheticQAGenerator, local GGUF)──> queries (golden questions excluded)
query ──(store.search top-20, e5-small)──> candidate texts (in-distribution hard negatives)
(query, text) ──(teacher bge-reranker-v2-m3 .predict)──> teacher_score in [0,1]
pairs.jsonl ──(train_reranker: rubert-tiny2 + BCEWithLogits on soft labels)──> var/models/kbai-reranker-ru/
model dir ──(eval_rag run --rerank, KB_RERANK_MODEL=<dir>)──> golden_public 3-way gate
```

---

### Task 1: Dataset builder — pure core (pairs, anti-leak, persistence)

**Files:**
- Create: `scripts/build_rerank_dataset.py`
- Test: `tests/scripts/test_build_rerank_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_build_rerank_dataset.py
"""Pure-function tests for the distillation dataset builder (no ML deps)."""

import json

import pytest

from scripts.build_rerank_dataset import (
    Pair,
    build_pairs,
    normalize_question,
    write_pairs,
)


def _retrieve(query: str, k: int):
    return [(f"doc.md:{i}", f"text {i} for {query}") for i in range(k)]


def test_normalize_question_strips_case_space_punctuation():
    assert normalize_question("  Какой Срок?  ") == normalize_question("какой срок")
    assert normalize_question("Что это?!") == "что это"


def test_build_pairs_excludes_golden_queries():
    queries = [("Сколько дней отпуск?", "a.md:1"), ("Уникальный вопрос?", "a.md:2")]
    golden = frozenset({"Сколько дней отпуск?"})
    pairs = build_pairs(queries, _retrieve, golden, k=3)
    assert {p.query for p in pairs} == {"Уникальный вопрос?"}


def test_build_pairs_yields_k_candidates_per_query():
    pairs = build_pairs([("вопрос?", "a.md:0")], _retrieve, frozenset(), k=5)
    assert len(pairs) == 5
    assert pairs[0] == Pair(query="вопрос?", chunk_key="doc.md:0", text="text 0 for вопрос?")


def test_build_pairs_filters_normalized_golden_variants():
    # Spec §3.4: the leak filter must catch case/punctuation variants, not
    # just exact matches.
    queries = [("сколько ДНЕЙ отпуск", "a.md:1")]
    golden = frozenset({"Сколько дней отпуск?"})
    assert build_pairs(queries, _retrieve, golden, k=2) == []


def test_write_pairs_roundtrip(tmp_path):
    pairs = [Pair(query="q", chunk_key="d.md:0", text="t")]
    out = tmp_path / "pairs.jsonl"
    write_pairs(out, pairs, scores=[0.75], meta={"teacher": "x"})
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row == {"query": "q", "chunk_key": "d.md:0", "text": "t", "teacher_score": 0.75}
    meta = json.loads((tmp_path / "pairs.meta.json").read_text(encoding="utf-8"))
    assert meta["teacher"] == "x"


def test_write_pairs_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_pairs(tmp_path / "p.jsonl", [Pair("q", "k", "t")], scores=[], meta={})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_build_rerank_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_rerank_dataset'`

- [ ] **Step 3: Write the minimal implementation**

```python
# scripts/build_rerank_dataset.py
"""Build the reranker-distillation training set (spec 2026-06-10).

Pipeline: synthetic queries per chunk (W1 generator) -> candidate mining via
the production bi-encoder (``store.search``) -> teacher scores
(bge-reranker-v2-m3). Output: ``var/data/rerank/pairs.jsonl`` + ``.meta.json``
sidecar. Queries colliding with the public golden are excluded (anti-leak,
spec §3.4) — enforced in code, with an assert as backstop.

Heavy imports (sentence_transformers, the LLM provider) are lazy: importing
this module must stay cheap so stub-backed unit tests never touch ML deps.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

LOGGER = logging.getLogger(__name__)

PAIRS_OUT = Path("var/data/rerank/pairs.jsonl")
GOLDEN_PUBLIC = Path("data/eval/golden_public.jsonl")
DEFAULT_TEACHER = "BAAI/bge-reranker-v2-m3"

# (query, k) -> [(chunk_key, text), ...]
Retrieve = Callable[[str, int], Sequence[tuple[str, str]]]


@dataclass(frozen=True)
class Pair:
    query: str
    chunk_key: str
    text: str


def normalize_question(q: str) -> str:
    """Collapse whitespace/case/trailing punctuation for leak comparison."""
    return " ".join(q.lower().split()).rstrip("?!. ")


def build_pairs(
    queries: Sequence[tuple[str, str]],
    retrieve: Retrieve,
    golden_questions: frozenset[str],
    *,
    k: int = 20,
) -> list[Pair]:
    """Mine top-*k* candidates per query, dropping golden-colliding queries."""
    banned = {normalize_question(q) for q in golden_questions}
    out: list[Pair] = []
    for query, _source_key in queries:
        if normalize_question(query) in banned:
            continue
        for chunk_key, text in retrieve(query, k):
            out.append(Pair(query=query, chunk_key=chunk_key, text=text))
    leaked = {normalize_question(p.query) for p in out} & banned
    assert not leaked, f"golden leak into training pairs: {sorted(leaked)[:3]}"
    return out


def write_pairs(path: Path, pairs: Sequence[Pair], scores: Sequence[float], meta: dict) -> None:
    if len(pairs) != len(scores):
        raise ValueError(f"pairs/scores length mismatch: {len(pairs)} != {len(scores)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair, score in zip(pairs, scores, strict=True):
            fh.write(
                json.dumps(
                    {
                        "query": pair.query,
                        "chunk_key": pair.chunk_key,
                        "text": pair.text,
                        "teacher_score": float(score),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/scripts/test_build_rerank_dataset.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```powershell
py -3.13 -m ruff check scripts/build_rerank_dataset.py tests/scripts/test_build_rerank_dataset.py && py -3.13 -m black scripts/build_rerank_dataset.py tests/scripts/test_build_rerank_dataset.py
git add scripts/build_rerank_dataset.py tests/scripts/test_build_rerank_dataset.py
git commit -m "feat(rerank-distill): dataset builder pure core (pairs, anti-leak, persistence)"
```

---

### Task 2: Dataset builder — query generation, retrieval wrapper, teacher, CLI

**Files:**
- Modify: `scripts/build_rerank_dataset.py` (append)
- Test: `tests/scripts/test_build_rerank_dataset.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_build_rerank_dataset.py`. Important: do NOT add a new import line mid-file (ruff E402) — extend the existing top-of-file import block to:

```python
from scripts.build_rerank_dataset import (
    Pair,
    as_retrieve,
    build_pairs,
    dedupe_queries,
    normalize_question,
    write_pairs,
)
```

then append the tests:

```python
class _Hit:
    def __init__(self, chunk_key: str, text: str):
        self.chunk_key = chunk_key
        self.text = text


def test_as_retrieve_adapts_eval_retriever_to_tuples():
    def eval_retriever(query: str, k: int):
        return [_Hit(f"d.md:{i}", f"t{i}") for i in range(k)]

    retrieve = as_retrieve(eval_retriever)
    assert retrieve("q", 2) == [("d.md:0", "t0"), ("d.md:1", "t1")]


def test_dedupe_queries_by_normalized_text_keeps_first():
    queries = [("Какой срок?", "a.md:0"), ("какой СРОК", "b.md:1"), ("Другой?", "c.md:2")]
    assert dedupe_queries(queries) == [("Какой срок?", "a.md:0"), ("Другой?", "c.md:2")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_build_rerank_dataset.py -v`
Expected: 2 new FAIL — `ImportError: cannot import name 'as_retrieve'`

- [ ] **Step 3: Implement the helpers + the heavy (manually-verified) plumbing**

Append to `scripts/build_rerank_dataset.py`:

```python
def as_retrieve(eval_retriever) -> Retrieve:
    """Adapt an ``app.eval.adapter`` Retriever (EvalHit) to (chunk_key, text)."""

    def _retrieve(query: str, k: int) -> list[tuple[str, str]]:
        return [(h.chunk_key, h.text) for h in eval_retriever(query, k)]

    return _retrieve


def dedupe_queries(queries: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for query, source_key in queries:
        norm = normalize_question(query)
        if norm in seen:
            continue
        seen.add(norm)
        out.append((query, source_key))
    return out


def generate_queries(store, provider, *, rounds: int, limit_chunks: int = 0) -> list[tuple[str, str]]:
    """Synthetic (query, source_chunk_key) via the W1 generator. LLM-slow."""
    from app.eval.adapter import build_global_id_key_map
    from app.services import synthetic_qa as sq

    generator = sq.SyntheticQAGenerator(provider=provider)
    key_map = build_global_id_key_map(store)
    chunks = list(sq.iter_chunks(store))
    if limit_chunks:
        chunks = chunks[:limit_chunks]
    queries: list[tuple[str, str]] = []
    for round_no in range(rounds):
        for chunk_id, text in chunks:
            for qa in generator.generate_for_chunk(
                chunks=[text], chunk_ids=[chunk_id], mode=sq.GenerationMode.SINGLE
            ):
                key = key_map.get(qa.source_chunk_id)
                if key is not None:
                    queries.append((qa.instruction, key))
        LOGGER.info("round %d/%d: %d queries so far", round_no + 1, rounds, len(queries))
    return dedupe_queries(queries)


def teacher_scores(pairs: Sequence[Pair], *, model_name: str, batch_size: int) -> list[float]:
    """Score (query, text) with the teacher cross-encoder. CPU-slow."""
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(model_name, max_length=512)
    scores = encoder.predict(
        [(p.query, p.text) for p in pairs], batch_size=batch_size, show_progress_bar=True
    )
    return [float(s) for s in scores]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="build_rerank_dataset")
    parser.add_argument("--out", default=str(PAIRS_OUT))
    parser.add_argument("--rounds", type=int, default=3, help="QA-generation passes per chunk")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit-chunks", type=int, default=0, help="smoke runs (0 = all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.eval.adapter import make_mvp_retriever
    from app.eval.dataset import load_golden
    from app.services.kb_store import get_store
    from scripts.eval_rag import _gen_provider

    store = get_store()
    queries = generate_queries(
        store, _gen_provider(), rounds=args.rounds, limit_chunks=args.limit_chunks
    )
    golden_questions = frozenset(item.question for item in load_golden(GOLDEN_PUBLIC))
    pairs = build_pairs(
        queries, as_retrieve(make_mvp_retriever(store)), golden_questions, k=args.candidates
    )
    LOGGER.info("scoring %d pairs with teacher %s", len(pairs), args.teacher)
    scores = teacher_scores(pairs, model_name=args.teacher, batch_size=args.batch)
    write_pairs(
        Path(args.out),
        pairs,
        scores,
        meta={
            "teacher": args.teacher,
            "rounds": args.rounds,
            "candidates": args.candidates,
            "n_queries": len(queries),
            "n_pairs": len(pairs),
            "golden_excluded": str(GOLDEN_PUBLIC),
        },
    )
    print(f"Wrote {len(pairs)} pairs ({len(queries)} queries) to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + full file lint**

Run: `py -3.13 -m pytest tests/scripts/test_build_rerank_dataset.py -v`
Expected: 8 passed
Run: `py -3.13 -m ruff check scripts/build_rerank_dataset.py && py -3.13 -m black --check scripts/build_rerank_dataset.py`
Expected: clean

- [ ] **Step 5: Verify module import stays light (stub safety)**

Run: `py -3.13 -c "import scripts.build_rerank_dataset; import sys; assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules; print('light import OK')"`
Expected: `light import OK`

- [ ] **Step 6: Commit**

```powershell
git add scripts/build_rerank_dataset.py tests/scripts/test_build_rerank_dataset.py
git commit -m "feat(rerank-distill): query generation, candidate mining, teacher scoring CLI"
```

---

### Task 3: Trainer — `scripts/train_reranker.py`

**Files:**
- Create: `scripts/train_reranker.py`
- Test: `tests/scripts/test_train_reranker.py`

- [ ] **Step 1: Write the failing tests (pure parts only — no torch in CI)**

```python
# tests/scripts/test_train_reranker.py
"""Data-plumbing tests for the distillation trainer. The training loop itself
is exercised by the manual gate run (plan Task 6), not by unit tests — it
needs torch+transformers which CI stubs out."""

import json

import pytest

from scripts.train_reranker import load_pairs, soft_label, split_by_query


def _write_pairs(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )


def test_load_pairs_reads_jsonl(tmp_path):
    p = tmp_path / "pairs.jsonl"
    _write_pairs(p, [{"query": "q", "text": "t", "teacher_score": 0.5}])
    assert load_pairs(p) == [{"query": "q", "text": "t", "teacher_score": 0.5}]


def test_load_pairs_rejects_empty(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_pairs(p)


def test_soft_label_passthrough_for_probabilities():
    assert soft_label(0.0) == 0.0
    assert soft_label(0.73) == 0.73
    assert soft_label(1.0) == 1.0


def test_soft_label_sigmoid_for_raw_logits():
    assert 0.95 < soft_label(4.0) < 1.0
    assert 0.0 < soft_label(-4.0) < 0.05


def test_split_by_query_is_query_disjoint_and_deterministic():
    rows = [{"query": f"q{i % 10}", "text": f"t{i}", "teacher_score": 0.1} for i in range(100)]
    train_a, val_a = split_by_query(rows, val_fraction=0.2, seed=42)
    train_b, val_b = split_by_query(rows, val_fraction=0.2, seed=42)
    assert (train_a, val_a) == (train_b, val_b)
    assert {r["query"] for r in train_a} & {r["query"] for r in val_a} == set()
    assert len(train_a) + len(val_a) == len(rows)
    assert val_a  # non-empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_train_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.train_reranker'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/train_reranker.py
"""Train ``kbai-reranker-ru``: distill teacher scores into rubert-tiny2.

Student = ``cointegrated/rubert-tiny2`` + 1-logit sequence-classification head;
loss = BCEWithLogits against the teacher's soft scores (spec 2026-06-10 §3.3).
The saved directory is a plain HF checkpoint, loadable by
``sentence_transformers.CrossEncoder(<dir>)`` — i.e. directly usable as
``KB_RERANK_MODEL``. torch/transformers imports are lazy (stub safety).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

BASE_MODEL = "cointegrated/rubert-tiny2"
DEFAULT_PAIRS = Path("var/data/rerank/pairs.jsonl")
DEFAULT_OUT = Path("var/models/kbai-reranker-ru")


def load_pairs(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"Empty pairs file: {path}")
    return rows


def soft_label(score: float) -> float:
    """Teacher scores are sigmoid-activated probabilities; tolerate raw logits."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


def split_by_query(
    rows: list[dict], *, val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Query-disjoint split: all pairs of one query land on the same side."""
    queries = sorted({r["query"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(queries)
    n_val = max(1, int(len(queries) * val_fraction))
    val_queries = set(queries[:n_val])
    train_rows = [r for r in rows if r["query"] not in val_queries]
    val_rows = [r for r in rows if r["query"] in val_queries]
    return train_rows, val_rows


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
) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)

    def collate(batch: list[dict]):
        enc = tokenizer(
            [b["query"] for b in batch],
            [b["text"] for b in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(
            [soft_label(float(b["teacher_score"])) for b in batch], dtype=torch.float32
        )
        return enc

    loader = DataLoader(rows_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for step, batch in enumerate(loader):
            labels = batch.pop("labels")
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {float(loss):.4f}")
        print(f"epoch {epoch} mean loss {total / max(1, len(loader)):.4f}")

    model.eval()
    val_loader = DataLoader(rows_val, batch_size=batch_size, shuffle=False, collate_fn=collate)
    preds: list[float] = []
    gold: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels")
            logits = model(**batch).logits.squeeze(-1)
            preds.extend(torch.sigmoid(logits).tolist())
            gold.extend(labels.tolist())
    pearson = float(np.corrcoef(preds, gold)[0, 1]) if len(preds) > 1 else float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return {"val_pairs": len(preds), "val_pearson_vs_teacher": pearson}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="train_reranker")
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows = load_pairs(Path(args.pairs))
    rows_train, rows_val = split_by_query(rows, val_fraction=args.val_fraction, seed=args.seed)
    print(f"train pairs: {len(rows_train)}, val pairs: {len(rows_val)}")
    metrics = train(
        rows_train,
        rows_val,
        out_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )
    meta = {
        "base_model": BASE_MODEL,
        "pairs_file": args.pairs,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        **metrics,
    }
    (Path(args.out) / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, light-import check, lint**

Run: `py -3.13 -m pytest tests/scripts/test_train_reranker.py -v`
Expected: 5 passed
Run: `py -3.13 -c "import scripts.train_reranker; import sys; assert 'torch' not in sys.modules; print('light import OK')"`
Expected: `light import OK`
Run: `py -3.13 -m ruff check scripts/train_reranker.py tests/scripts/test_train_reranker.py && py -3.13 -m black --check scripts/train_reranker.py tests/scripts/test_train_reranker.py`
Expected: clean

- [ ] **Step 5: Commit**

```powershell
git add scripts/train_reranker.py tests/scripts/test_train_reranker.py
git commit -m "feat(rerank-distill): rubert-tiny2 distillation trainer (BCE on soft teacher labels)"
```

---

### Task 4: Latency bench — `scripts/bench_reranker.py`

**Files:**
- Create: `scripts/bench_reranker.py`
- Test: `tests/scripts/test_bench_reranker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scripts/test_bench_reranker.py
"""Tests for the CPU-latency gate helpers (fake scorer — no model load)."""

import json

import pytest

from scripts.bench_reranker import group_queries, measure, percentile


def test_group_queries_collects_texts_per_query(tmp_path):
    rows = [
        {"query": "a", "text": "t1", "teacher_score": 0.1},
        {"query": "a", "text": "t2", "teacher_score": 0.2},
        {"query": "b", "text": "t3", "teacher_score": 0.3},
    ]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert group_queries(p) == {"a": ["t1", "t2"], "b": ["t3"]}


def test_measure_calls_scorer_once_per_query_with_capped_candidates():
    calls: list[list[tuple[str, str]]] = []

    def fake_score(pairs):
        calls.append(list(pairs))
        return [0.0] * len(pairs)

    timings = measure(fake_score, [("q", ["t1", "t2", "t3"])], candidates=2)
    assert len(timings) == 1
    assert calls == [[("q", "t1"), ("q", "t2")]]
    assert timings[0] >= 0.0


def test_percentile_p95_and_median():
    timings = [float(v) for v in range(1, 101)]
    assert percentile(timings, 0.50) == pytest.approx(50.0, abs=1.0)
    assert percentile(timings, 0.95) == pytest.approx(95.0, abs=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_bench_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_reranker'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/bench_reranker.py
"""CPU-latency gate for the distilled reranker (spec 2026-06-10 §4.2).

Reranking 20 candidates must fit the tier-B budget (default 200 ms p95).
Measures end-to-end ``CrossEncoder.predict`` wall time per query over real
pairs from the training set. Exit code 1 on budget violation, so the run is
recordable as a pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_PAIRS = Path("var/data/rerank/pairs.jsonl")
DEFAULT_MODEL = "var/models/kbai-reranker-ru"

ScoreFn = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


def group_queries(pairs_path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(row["query"], []).append(row["text"])
    return grouped


def measure(
    score_fn: ScoreFn, queries: Sequence[tuple[str, list[str]]], *, candidates: int
) -> list[float]:
    timings: list[float] = []
    for query, texts in queries:
        batch = [(query, text) for text in texts[:candidates]]
        started = time.perf_counter()
        score_fn(batch)
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def percentile(timings: Sequence[float], q: float) -> float:
    ordered = sorted(timings)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bench_reranker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    args = parser.parse_args(argv)

    grouped = group_queries(Path(args.pairs))
    sample = [(q, t) for q, t in grouped.items() if len(t) >= args.candidates][: args.queries]
    if not sample:
        raise SystemExit("No queries with enough candidates in the pairs file.")

    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(args.model, max_length=384)
    measure(encoder.predict, sample[:2], candidates=args.candidates)  # warm-up
    timings = measure(encoder.predict, sample, candidates=args.candidates)
    p50 = percentile(timings, 0.50)
    p95 = percentile(timings, 0.95)
    print(
        f"rerank {args.candidates} candidates x {len(timings)} queries: "
        f"p50={p50:.0f}ms p95={p95:.0f}ms (budget {args.budget_ms:.0f}ms)"
    )
    if p95 > args.budget_ms:
        raise SystemExit(f"FAIL: p95 {p95:.0f}ms > budget {args.budget_ms:.0f}ms")
    print("PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, light-import check, lint**

Run: `py -3.13 -m pytest tests/scripts/test_bench_reranker.py -v`
Expected: 3 passed
Run: `py -3.13 -c "import scripts.bench_reranker; import sys; assert 'sentence_transformers' not in sys.modules; print('light import OK')"`
Expected: `light import OK`
Run: `py -3.13 -m ruff check scripts/bench_reranker.py tests/scripts/test_bench_reranker.py && py -3.13 -m black --check scripts/bench_reranker.py tests/scripts/test_bench_reranker.py`
Expected: clean

- [ ] **Step 5: Commit**

```powershell
git add scripts/bench_reranker.py tests/scripts/test_bench_reranker.py
git commit -m "feat(rerank-distill): CPU latency gate (p95 budget) for the student reranker"
```

---

### Task 5: Full-suite hygiene

**Files:** none new.

- [ ] **Step 1: Run the full Python suite (mirrors CI)**

Run: `py -3.13 -m pytest -q --ignore=backend; echo "EXIT=$LASTEXITCODE"`
Expected: `EXIT=0` (judge by exit code — piping drops the summary line; memory: pytest-piped-summary)

- [ ] **Step 2: Lint + style + types**

Run: `py -3.13 -m ruff check . && py -3.13 -m black --check .`
Expected: clean
Run: `py -3.13 -m mypy app`
Expected: no NEW errors (baseline 244 pre-existing; this plan does not touch `app/`, so the total must not grow)

- [ ] **Step 3: Commit anything the formatters changed**

```powershell
git status --short
# if formatters touched files:
git add -u && git commit -m "style(rerank-distill): formatter pass"
```

---

### Task 6: Manual gate run (local, heavy — not CI)

**Files:**
- Output (not committed): `var/data/rerank/pairs.jsonl`, `var/models/kbai-reranker-ru/`, `var/data/eval/run_{base,student,teacher}.json`

All commands in PowerShell from the repo root. The store/env must match the one
the public golden was generated against (`golden_public.sig.json` is checked by
`eval_rag run` — a mismatch aborts with "Corpus signature mismatch").

- [ ] **Step 1: Ingest the public corpus into a dedicated store**

```powershell
$env:KB_MVP_DB_PATH = "var/data/eval/public.sqlite3"
py -3.13 -m scripts.build_public_corpus ingest
```
Expected: ingest count printed; idempotent re-run OK.

- [ ] **Step 2: Smoke-run the dataset builder (minutes, not hours)**

```powershell
py -3.13 -m scripts.build_rerank_dataset --limit-chunks 5 --rounds 1 --out var/data/rerank/pairs_smoke.jsonl
Get-Content var/data/rerank/pairs_smoke.jsonl -TotalCount 3
```
Expected: a few dozen pairs; each line has `query`/`chunk_key`/`text`/`teacher_score` with scores in [0,1]. **Eyeball the queries** — they must read like real user questions (spec §5: manual sample check).

- [ ] **Step 3: Full dataset build — detached (hours: GGUF query-gen + CPU teacher)**

```powershell
New-Item -ItemType Directory -Force var/log | Out-Null
Start-Process -FilePath "py" -ArgumentList "-3.13","-m","scripts.build_rerank_dataset","--rounds","3","--candidates","20" `
  -RedirectStandardOutput "var/log/rerank_dataset.out.log" -RedirectStandardError "var/log/rerank_dataset.err.log" -NoNewWindow
```
Monitor `var/log/rerank_dataset.out.log` (persistent monitor + dead-man rule; memory: detached-long-runs). Done when the final `Wrote N pairs` line appears. Expected N ≈ 15k–40k (≈300–500 chunks × 3 rounds, deduped, × 20 candidates).

- [ ] **Step 4: Anti-leak spot check**

```powershell
py -3.13 -m scripts.check_rerank_leak
```
Create `scripts/check_rerank_leak.py` now with exactly this content (it is committed in Task 7 Step 3):

```python
"""One-off: verify zero golden->train query overlap (spec §3.4)."""
import json
import sys
from pathlib import Path

from app.eval.dataset import load_golden
from scripts.build_rerank_dataset import GOLDEN_PUBLIC, normalize_question

golden = {normalize_question(item.question) for item in load_golden(GOLDEN_PUBLIC)}
train = {
    normalize_question(json.loads(line)["query"])
    for line in Path("var/data/rerank/pairs.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
overlap = golden & train
print(f"leak overlap: {len(overlap)}")
sys.exit(1 if overlap else 0)
```
Expected: `leak overlap: 0`, exit 0.

- [ ] **Step 5: Train (CPU; detached if it runs past ~30 min)**

```powershell
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/pairs.jsonl --out var/models/kbai-reranker-ru
```
Expected: falling loss; final JSON with `val_pearson_vs_teacher` — **≥ 0.8 is healthy**; < 0.6 means the student did not learn the teacher (stop and investigate the data before evaluating).

- [ ] **Step 6: Latency gate**

```powershell
py -3.13 -m scripts.bench_reranker --model var/models/kbai-reranker-ru
```
Expected: `PASS` with p95 ≤ 200 ms. (Also run once with `--model BAAI/bge-reranker-v2-m3` to record the teacher's CPU latency in the runbook.)

- [ ] **Step 7: Three-way quality gate on golden_public**

```powershell
$env:KB_MVP_DB_PATH = "var/data/eval/public.sqlite3"
Remove-Item Env:KB_RERANK_MODEL -ErrorAction SilentlyContinue
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_public.jsonl --out var/data/eval/run_base.json

$env:KB_RERANK_MODEL = "var/models/kbai-reranker-ru"
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_public.jsonl --rerank --out var/data/eval/run_student.json

$env:KB_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_public.jsonl --rerank --out var/data/eval/run_teacher.json

py -3.13 -m scripts.eval_rag compare var/data/eval/run_base.json var/data/eval/run_student.json
py -3.13 -m scripts.eval_rag compare var/data/eval/run_teacher.json var/data/eval/run_student.json
```

**Gate (spec §4):** student vs base — `hit@5` +2 п.п. OR `mrr@5` +0.02, AND not worse on any of `hit@5`/`recall@5`/`mrr@5`. Stretch: student ≥ 90% of teacher's metrics.
If the gate fails: the model does NOT ship; record results in the runbook anyway (the pipeline is the asset) and iterate (more rounds, `--candidates 30`, epoch/lr sweep) — do not silently relax the gate.

---

### Task 7: Runbook + spec status

**Files:**
- Create: `docs/superpowers/runbooks/2026-06-10-own-reranker-training.md`
- Modify: `docs/superpowers/specs/2026-06-10-own-reranker-distillation-design.md` (Status line)

- [ ] **Step 1: Write the runbook**

Content: the exact Task 6 command sequence as actually executed, plus a results table filled with the REAL numbers from the gate run:

```markdown
# Runbook: training & gating kbai-reranker-ru

## Commands
(фактически выполненные команды Task 6, скопированные из терминала)

## Results (YYYY-MM-DD, commit <sha>)
| run | hit@5 | recall@5 | mrr@5 | p95 latency (20 cand) |
|---|---|---|---|---|
| base (no rerank) | … | … | … | — |
| student kbai-reranker-ru | … | … | … | … ms |
| teacher bge-reranker-v2-m3 | … | … | … | … ms |

Gate: PASS/FAIL (criteria: spec §4). val_pearson_vs_teacher = …

## Model card (kbai-reranker-ru v1)
- base: cointegrated/rubert-tiny2 (~29M); teacher: BAAI/bge-reranker-v2-m3
- data: N pairs from corpus_public (9 RU docs), synthetic queries, golden excluded
- limitation: домен RU-документов проекта; не универсальная модель (spec §5)
- distribution: private, product bundle only (spec decision 2026-06-10)
```

- [ ] **Step 2: Update spec status**

In the spec header change `**Status:** Design approved …` to `**Status:** Implemented; gate result — see runbooks/2026-06-10-own-reranker-training.md`.

- [ ] **Step 3: Commit**

```powershell
git add docs/superpowers/runbooks/2026-06-10-own-reranker-training.md docs/superpowers/specs/2026-06-10-own-reranker-distillation-design.md scripts/check_rerank_leak.py
git commit -m "docs(rerank-distill): training runbook with gate results + spec status"
```

---

## Out of scope (do not do here)

- No production-code changes (`app/**` untouched; `KB_RERANK_MODEL` already supports a local dir via `CrossEncoder(path)` — if that assumption breaks at Task 6 Step 6, a minimal path-resolve fix in `app/retriever/rerank.py` is allowed as a separate commit with a test).
- No default-on flip of `KB_RERANK_ENABLED` (separate follow-up after the gate).
- No embedder distillation, no per-client tuning, no HF publishing (spec non-goals).
- Committed artifacts are code + docs only: `var/` outputs (pairs, model weights, run JSONs) stay untracked.
