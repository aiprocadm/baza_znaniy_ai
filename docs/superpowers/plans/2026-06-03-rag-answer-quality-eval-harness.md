# RAG Answer-Quality Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, offline-testable eval harness that scores RAG retrieval (and then generation) quality on the MVP surface, producing a baseline report — the measurement foundation for the gated quick-wins.

**Architecture:** A new I/O-free `app/eval/` package (pure metrics, dataset schema, retriever adapter, scorers, reporting) plus a `scripts/eval_rag.py` CLI. It reuses the existing `app/services/synthetic_qa.py` (golden generation) and the live `KnowledgeBaseStore`. Retrieval scoring uses a globally-unique chunk identity (`kb_chunks.id`); generation scoring uses an injected LLM-judge.

**Tech Stack:** Python 3.12 (Windows `py -3` launcher, **no venv**), pytest, SQLite (`KnowledgeBaseStore`), OpenAI-compatible LLM via `app/services/kb_llm.py`. Spec: `docs/superpowers/specs/2026-06-03-rag-answer-quality-eval-design.md`.

---

## Scope of this plan

This plan covers **PR1–PR3** of the spec (the harness + baseline). It ends at the **baseline checkpoint**. The measured quick-wins (PR4–PR8) are a deliberately separate, data-dependent plan written *after* the baseline numbers exist (see "Phase 2 — deferred" at the bottom). This plan produces working, testable software on its own: `eval_rag.py run` emitting a baseline report.

### Two deviations from the spec (decided during planning, confirm if you disagree)

1. **MVP-first, v1 deferred.** v1/Qdrant hits live in a different chunk-identity space than the MVP `kb_chunks.id` that golden labels come from, so id-based scoring is only sound on MVP. v1 retrieval scoring (content-match) moves to Phase 2. The headline baseline gap this plan exposes is therefore **MVP-on-hashing vs MVP-on-real-embedder** (toggle `KB_EMBEDDINGS_BACKEND`), which is the actual quality cliff and needs no cross-index identity.
2. **Chunk identity = `kb_chunks.id`.** `SearchHit` exposes `(document_id, chunk_index)` but not the row id, and `chunk_index` is only unique within a document. The adapter resolves each hit to its global `kb_chunks.id` via a one-time map, matching `synthetic_qa.iter_chunks`/`QAPair.source_chunk_id`.

## Verified reuse interfaces (do not re-derive)

- `app.services.kb_store`: `KnowledgeBaseStore(db_path, *, embedder=None)`; `.embedder` property; `.search(query, *, top_k=5) -> List[SearchHit]`; `get_store()` (cached default, reads `KB_MVP_DB_PATH`/`DATA_DIR`); `._connect()` (sqlite conn ctx mgr). `SearchHit` fields: `document_id, document_title, chunk_index, text, score, source, filename, page, has_original`. Table `kb_chunks(id, document_id, chunk_index, text, embedding, dim, page_number)`; table `kb_documents(id, ...)`.
- `app.services.kb_embeddings.get_embedder()`; embedder exposes `.name` (`"hash"` for the fallback) and `.dimension`.
- `app.services.kb_llm.select_provider() -> provider | None`; provider exposes `.name`, `.model`, `.generate(prompt, *, system=None, max_tokens=None, temperature=None) -> LLMResponse(text, provider, model, elapsed_ms, raw_usage)`.
- `app.services.synthetic_qa`: `QAPair(instruction, input, output, source_chunk_id)`; `iter_chunks(store) -> Iterator[(chunk_id, text)]` (chunk_id = `kb_chunks.id`); `SyntheticQAGenerator(provider=...)` with `.generate_for_chunk(chunks=[...], chunk_ids=[...], mode=GenerationMode.SINGLE) -> list[QAPair]`; `GenerationMode`; `estimate_total_cost_usd(provider=, model=, mode=, chunk_chars=[...]) -> float | None`; `is_refusal(text) -> bool`.
- `app.api.kb_mvp._RAG_SYSTEM_PROMPT` (the production MVP system prompt — pinned by a drift test, not imported at runtime).

---

## PR1 — Retrieval baseline

### Task 1: Pure retrieval metrics

**Files:**
- Create: `app/eval/__init__.py` (empty)
- Create: `app/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py
from app.eval.metrics import hit_at_k, recall_at_k, mrr_at_k, score_item, aggregate, RETRIEVAL_KS


def test_hit_at_k_true_only_within_k():
    assert hit_at_k({7}, [3, 7, 1], 3) == 1.0
    assert hit_at_k({7}, [3, 1, 7], 2) == 0.0


def test_recall_at_k_fraction_of_relevant_found():
    assert recall_at_k({7, 9}, [7, 1, 2], 3) == 0.5
    assert recall_at_k({7, 9}, [7, 9, 2], 3) == 1.0


def test_mrr_at_k_uses_first_relevant_rank():
    assert mrr_at_k({9}, [1, 9, 3], 5) == 0.5
    assert mrr_at_k({9}, [1, 2, 3], 2) == 0.0


def test_empty_relevant_scores_zero():
    assert hit_at_k(set(), [1, 2], 3) == 0.0
    assert recall_at_k(set(), [1, 2], 3) == 0.0


def test_score_item_and_aggregate_keys():
    row = score_item({7}, [7, 1, 2])
    assert row["hit@1"] == 1.0 and row["mrr@3"] == 1.0
    assert set(row) == {f"{m}@{k}" for m in ("hit", "recall", "mrr") for k in RETRIEVAL_KS}
    agg = aggregate([score_item({7}, [7]), score_item({7}, [1, 7])])
    assert agg["hit@1"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/__init__.py
```
(empty file)

```python
# app/eval/metrics.py
"""Pure retrieval-quality metrics. No I/O, no env, no globals."""
from __future__ import annotations

from typing import Collection, Sequence

RETRIEVAL_KS: tuple[int, ...] = (1, 3, 5, 10)


def hit_at_k(relevant: Collection[int], retrieved: Sequence[int], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def recall_at_k(relevant: Collection[int], retrieved: Sequence[int], k: int) -> float:
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    topk = set(retrieved[:k])
    return sum(1 for cid in rel if cid in topk) / len(rel)


def mrr_at_k(relevant: Collection[int], retrieved: Sequence[int], k: int) -> float:
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def score_item(
    relevant: Collection[int],
    retrieved: Sequence[int],
    ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(relevant, retrieved, k)
        out[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        out[f"mrr@{k}"] = mrr_at_k(relevant, retrieved, k)
    return out


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {key: sum(r[key] for r in rows) / n for key in rows[0]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/__init__.py app/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): pure retrieval metrics (hit/recall/mrr@k)"
```

---

### Task 2: Golden-set schema + JSONL I/O

**Files:**
- Create: `app/eval/dataset.py`
- Test: `tests/test_eval_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_dataset.py
import json
from app.eval.dataset import (
    GoldenItem, CorpusSignature, load_golden, save_golden,
    write_signature, read_signature,
)


def test_goldenitem_roundtrip():
    item = GoldenItem(question="Что такое отпуск?", relevant_chunk_ids=(7, 12),
                      reference_answer="Это перерыв.", expect_refusal=False, source="curated")
    back = GoldenItem.from_jsonl_line(item.to_jsonl_line())
    assert back == item


def test_reads_plain_qapair_line_back_compat():
    # A line emitted by synthetic_qa.QAPair (only source_chunk_id, no relevant_chunk_ids)
    line = json.dumps({"instruction": "Q?", "input": "", "output": "A [doc_chunk:5]",
                       "meta": {"source_chunk_id": 5}}, ensure_ascii=False)
    item = GoldenItem.from_jsonl_line(line)
    assert item.relevant_chunk_ids == (5,)
    assert item.reference_answer == "A [doc_chunk:5]"
    assert item.expect_refusal is False and item.source == "auto"


def test_save_and_load_golden(tmp_path):
    items = [GoldenItem("Q1", (1,), "A1"), GoldenItem("Q2", (), "", expect_refusal=True, source="curated")]
    path = tmp_path / "golden.jsonl"
    save_golden(path, items)
    assert load_golden(path) == items


def test_signature_sidecar_roundtrip(tmp_path):
    path = tmp_path / "golden.jsonl"
    sig = CorpusSignature(doc_count=3, max_chunk_id=42, embedder_name="ollama", dim=384)
    write_signature(path, sig)
    assert read_signature(path) == sig
    assert (tmp_path / "golden.sig.json").exists()


def test_read_signature_missing_returns_none(tmp_path):
    assert read_signature(tmp_path / "nope.jsonl") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_dataset.py -v`
Expected: FAIL with `ImportError: cannot import name 'GoldenItem'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/dataset.py
"""Golden-set schema + JSONL I/O. Back-compatible with synthetic_qa QAPair lines.

A golden line is a superset of the synthetic_qa QAPair layout: top-level
instruction/input/output is preserved (so scripts/validate_dataset.py stays
happy) and eval-specific fields live under ``meta``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GoldenItem:
    question: str
    relevant_chunk_ids: tuple[int, ...]
    reference_answer: str = ""
    expect_refusal: bool = False
    source: str = "auto"  # "auto" | "curated"

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.question,
            "input": "",
            "output": self.reference_answer,
            "meta": {
                "relevant_chunk_ids": [int(c) for c in self.relevant_chunk_ids],
                "source_chunk_id": int(self.relevant_chunk_ids[0]) if self.relevant_chunk_ids else 0,
                "expect_refusal": bool(self.expect_refusal),
                "source": self.source,
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl_line(cls, line: str) -> "GoldenItem":
        data = json.loads(line)
        meta = data.get("meta") or {}
        ids = meta.get("relevant_chunk_ids")
        if not ids:
            sid = meta.get("source_chunk_id")
            ids = [int(sid)] if sid is not None else []
        return cls(
            question=str(data["instruction"]),
            relevant_chunk_ids=tuple(int(c) for c in ids),
            reference_answer=str(data.get("output", "")),
            expect_refusal=bool(meta.get("expect_refusal", False)),
            source=str(meta.get("source", "auto")),
        )


@dataclass(frozen=True, slots=True)
class CorpusSignature:
    doc_count: int
    max_chunk_id: int
    embedder_name: str
    dim: int

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_count": self.doc_count,
            "max_chunk_id": self.max_chunk_id,
            "embedder_name": self.embedder_name,
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusSignature":
        return cls(
            doc_count=int(data["doc_count"]),
            max_chunk_id=int(data["max_chunk_id"]),
            embedder_name=str(data["embedder_name"]),
            dim=int(data["dim"]),
        )


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(GoldenItem.from_jsonl_line(line))
    return items


def save_golden(path: Path, items: Iterable[GoldenItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.to_jsonl_line())


def _sig_path(path: Path) -> Path:
    return Path(path).with_suffix(".sig.json")


def write_signature(path: Path, sig: CorpusSignature) -> None:
    _sig_path(path).write_text(
        json.dumps(sig.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_signature(path: Path) -> CorpusSignature | None:
    sp = _sig_path(path)
    if not sp.exists():
        return None
    return CorpusSignature.from_dict(json.loads(sp.read_text(encoding="utf-8")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_dataset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/dataset.py tests/test_eval_dataset.py
git commit -m "feat(eval): golden-set schema + JSONL/signature I/O"
```

---

### Task 3: Retriever adapter (chunk-id resolution)

**Files:**
- Create: `app/eval/adapter.py`
- Test: `tests/test_eval_adapter.py`

Note: `make_retriever` (pure, injected `search` + `id_map`) is unit-tested here. The store-touching wrappers `make_mvp_retriever` / `compute_signature` are smoke-tested in Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_adapter.py
from dataclasses import dataclass
from app.eval.adapter import EvalHit, make_retriever


@dataclass
class _Hit:
    document_id: int
    chunk_index: int
    text: str
    filename: str = ""
    document_title: str = "doc"


def test_make_retriever_resolves_global_chunk_id():
    # chunk_index repeats across documents; the (doc_id, chunk_index)->id map disambiguates.
    id_map = {(1, 0): 100, (1, 1): 101, (2, 0): 200}
    hits = [_Hit(2, 0, "from doc2", filename="b.pdf"), _Hit(1, 1, "from doc1")]
    retriever = make_retriever(lambda q, k: hits[:k], id_map)
    out = retriever("q", 5)
    assert out == [EvalHit(chunk_id=200, text="from doc2", title="b.pdf"),
                   EvalHit(chunk_id=101, text="from doc1", title="doc")]


def test_make_retriever_skips_unmapped_hits():
    retriever = make_retriever(lambda q, k: [_Hit(9, 9, "orphan")], {(1, 0): 1})
    assert retriever("q", 5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_adapter.py -v`
Expected: FAIL with `ImportError: cannot import name 'EvalHit'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/adapter.py
"""Bridge the eval to the live MVP retriever using a stable global chunk id.

The eval's canonical chunk identity is ``kb_chunks.id`` — the same id
``synthetic_qa.iter_chunks`` stamps onto ``QAPair.source_chunk_id``. The MVP
``SearchHit`` exposes ``(document_id, chunk_index)`` but NOT the row id, and
``chunk_index`` is only unique *within* a document, so each hit is resolved to
its global id via a one-time map built from the store.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class EvalHit:
    chunk_id: int
    text: str
    title: str = ""


Retriever = Callable[[str, int], Sequence[EvalHit]]


def make_retriever(
    search: Callable[[str, int], Sequence[object]],
    id_map: Mapping[tuple[int, int], int],
) -> Retriever:
    def _retrieve(query: str, top_k: int) -> list[EvalHit]:
        out: list[EvalHit] = []
        for h in search(query, top_k):
            cid = id_map.get((int(h.document_id), int(h.chunk_index)))
            if cid is None:
                continue
            title = getattr(h, "filename", "") or getattr(h, "document_title", "") or ""
            out.append(EvalHit(chunk_id=cid, text=h.text, title=title))
        return out

    return _retrieve


def _build_id_map(store) -> dict[tuple[int, int], int]:
    with store._connect() as conn:  # noqa: SLF001 — reuse store connection conventions
        rows = conn.execute("SELECT id, document_id, chunk_index FROM kb_chunks").fetchall()
    return {(int(doc_id), int(idx)): int(cid) for cid, doc_id, idx in rows}


def make_mvp_retriever(store) -> Retriever:
    """Wrap a live ``KnowledgeBaseStore`` as an eval Retriever."""
    return make_retriever(lambda q, k: store.search(q, top_k=k), _build_id_map(store))


def compute_signature(store):
    """Snapshot the live corpus for golden-set pinning.

    NOTE: ``embedder.dimension`` may trigger a one-time probe for remote
    embedders; acceptable here (run/generate already perform LLM calls).
    """
    from app.eval.dataset import CorpusSignature

    with store._connect() as conn:  # noqa: SLF001
        doc_count = int(conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0])
        row = conn.execute("SELECT MAX(id) FROM kb_chunks").fetchone()
        max_chunk_id = int(row[0]) if row and row[0] is not None else 0
    embedder = store.embedder
    return CorpusSignature(
        doc_count=doc_count,
        max_chunk_id=max_chunk_id,
        embedder_name=str(getattr(embedder, "name", "unknown")),
        dim=int(getattr(embedder, "dimension", 0)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_adapter.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/adapter.py tests/test_eval_adapter.py
git commit -m "feat(eval): retriever adapter with global chunk-id resolution"
```

---

### Task 4: Retrieval evaluator

**Files:**
- Create: `app/eval/retrieval_eval.py`
- Test: `tests/test_eval_retrieval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_retrieval.py
from app.eval.adapter import EvalHit
from app.eval.dataset import GoldenItem
from app.eval.retrieval_eval import evaluate


def _retriever(mapping):
    return lambda q, k: [EvalHit(cid, "t") for cid in mapping.get(q, [])][:k]


def test_evaluate_aggregates_over_items():
    items = [GoldenItem("q1", (7,)), GoldenItem("q2", (9,))]
    retriever = _retriever({"q1": [7, 1, 2], "q2": [1, 2, 3]})  # q1 hits, q2 misses
    result = evaluate(items, retriever)
    assert result["n"] == 2
    assert result["aggregate"]["hit@1"] == 0.5   # only q1 hits at rank 1
    assert result["aggregate"]["mrr@5"] == 0.5   # (1.0 + 0.0) / 2
    assert len(result["per_item"]) == 2


def test_evaluate_empty_items():
    assert evaluate([], _retriever({}))["aggregate"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval.retrieval_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/retrieval_eval.py
"""Run a retriever over golden items and score retrieval quality."""
from __future__ import annotations

from typing import Sequence

from app.eval.adapter import Retriever
from app.eval.dataset import GoldenItem
from app.eval.metrics import RETRIEVAL_KS, aggregate, score_item


def evaluate_item(item: GoldenItem, retriever: Retriever, *, max_k: int) -> dict[str, float]:
    hits = retriever(item.question, max_k)
    retrieved_ids = [h.chunk_id for h in hits]
    return score_item(item.relevant_chunk_ids, retrieved_ids)


def evaluate(
    items: Sequence[GoldenItem],
    retriever: Retriever,
    ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[str, object]:
    max_k = max(ks) if ks else 0
    per_item = [evaluate_item(it, retriever, max_k=max_k) for it in items]
    return {"n": len(items), "per_item": per_item, "aggregate": aggregate(per_item)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_retrieval.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/retrieval_eval.py tests/test_eval_retrieval.py
git commit -m "feat(eval): retrieval evaluator over golden items"
```

---

### Task 5: Report (JSON + Markdown + compare)

**Files:**
- Create: `app/eval/report.py`
- Test: `tests/test_eval_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_report.py
import json
from app.eval.report import build_report, to_markdown, save_report, compare


def _retrieval():
    return {"n": 2, "aggregate": {"hit@1": 0.5, "mrr@5": 0.75}}


def test_build_and_markdown():
    rep = build_report(surface="mvp",
                       signature={"embedder_name": "hash", "dim": 256, "doc_count": 3},
                       retrieval=_retrieval())
    assert rep["surface"] == "mvp" and rep["n"] == 2
    assert rep["retrieval"]["hit@1"] == 0.5
    md = to_markdown(rep)
    assert "hash" in md and "hit@1" in md


def test_save_writes_json_and_md(tmp_path):
    rep = build_report(surface="mvp", signature={"embedder_name": "ollama", "dim": 384, "doc_count": 1},
                       retrieval=_retrieval())
    out = tmp_path / "run.json"
    save_report(out, rep)
    assert json.loads(out.read_text(encoding="utf-8"))["surface"] == "mvp"
    assert (tmp_path / "run.md").exists()


def test_compare_emits_delta():
    a = build_report(surface="mvp", signature={}, retrieval={"aggregate": {"hit@1": 0.4}})
    b = build_report(surface="mvp", signature={}, retrieval={"aggregate": {"hit@1": 0.6}})
    out = compare(a, b)
    assert "+0.200" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/report.py
"""Assemble eval reports (JSON + Markdown) and diff two runs."""
from __future__ import annotations

import json
from pathlib import Path


def build_report(
    *,
    surface: str,
    signature: dict,
    retrieval: dict,
    generation: dict | None = None,
) -> dict:
    report: dict = {
        "surface": surface,
        "signature": signature,
        "n": retrieval.get("n", 0),
        "retrieval": retrieval.get("aggregate", {}),
    }
    if generation is not None:
        report["generation"] = generation.get("aggregate", {})
        report["generation_n"] = {
            "answerable": generation.get("n_answerable", 0),
            "refusal": generation.get("n_refusal", 0),
        }
    return report


def _metric_table(metrics: dict) -> list[str]:
    lines = ["| metric | value |", "|---|---|"]
    for key, val in metrics.items():
        lines.append(f"| {key} | {val:.3f} |")
    return lines


def to_markdown(report: dict) -> str:
    sig = report.get("signature", {})
    lines = [
        f"# RAG eval — surface `{report.get('surface', '?')}` (n={report.get('n', 0)})",
        "",
        f"- embedder: `{sig.get('embedder_name', '?')}` "
        f"(dim {sig.get('dim', '?')}), docs {sig.get('doc_count', '?')}",
        "",
        "## Retrieval",
        *_metric_table(report.get("retrieval", {})),
    ]
    if "generation" in report:
        lines += ["", "## Generation", *_metric_table(report["generation"])]
    return "\n".join(lines) + "\n"


def save_report(path: Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")


def compare(run_a: dict, run_b: dict) -> str:
    a_metrics = {**run_a.get("retrieval", {}), **run_a.get("generation", {})}
    b_metrics = {**run_b.get("retrieval", {}), **run_b.get("generation", {})}
    lines = ["# Compare", "", "| metric | A | B | Δ |", "|---|---|---|---|"]
    for key in sorted(set(a_metrics) | set(b_metrics)):
        a, b = a_metrics.get(key), b_metrics.get(key)
        if a is None or b is None:
            a_s = "—" if a is None else f"{a:.3f}"
            b_s = "—" if b is None else f"{b:.3f}"
            lines.append(f"| {key} | {a_s} | {b_s} | — |")
        else:
            lines.append(f"| {key} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_report.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/report.py tests/test_eval_report.py
git commit -m "feat(eval): JSON+Markdown report and compare diff"
```

---

### Task 6: CLI `run` (retrieval baseline) + guards

**Files:**
- Create: `scripts/eval_rag.py`
- Test: `tests/test_eval_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_cli.py
import pytest
from app.eval.adapter import EvalHit, make_mvp_retriever, compute_signature
from app.eval.dataset import GoldenItem
from app.services.kb_store import KnowledgeBaseStore
from app.services.kb_embeddings import HashingEmbedder


def _store_with_chunk(tmp_path):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=HashingEmbedder())
    doc_id = store.add_document(title="Doc", text="Отпуск — это оплачиваемый перерыв в работе сотрудника.")
    return store, doc_id


def test_mvp_retriever_and_signature_on_real_store(tmp_path):
    store, _ = _store_with_chunk(tmp_path)
    sig = compute_signature(store)
    assert sig.doc_count == 1 and sig.embedder_name == "hash" and sig.max_chunk_id >= 1
    retriever = make_mvp_retriever(store)
    hits = retriever("Что такое отпуск?", 5)
    assert hits and isinstance(hits[0], EvalHit)
    assert hits[0].chunk_id == sig.max_chunk_id  # single chunk -> its global id


def test_run_refuses_hashing_without_flag(tmp_path, monkeypatch):
    import scripts.eval_rag as cli
    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)
    golden = tmp_path / "g.jsonl"
    golden.write_text(GoldenItem("q", (1,)).to_jsonl_line(), encoding="utf-8")
    with pytest.raises(SystemExit, match="hashing"):
        cli.cmd_run(cli.build_parser().parse_args(
            ["run", "--golden", str(golden), "--out", str(tmp_path / "run.json")]))
```

> Verify at implementation time that `KnowledgeBaseStore.add_document(title=, text=)` is the correct ingest signature (grep `def add_document` in `app/services/kb_store.py`). If the public method differs (e.g. `add_text`/`ingest_text`), use that and keep the test's intent: one document → one or more chunks.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_rag'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/eval_rag.py
"""CLI for the RAG answer-quality eval harness.

Subcommands:
  run      — score retrieval (and, with --judge, generation) on the MVP surface
  generate — build a golden set from the corpus (added in a later task)
  compare  — diff two saved run JSONs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.eval import report as report_mod
from app.eval import retrieval_eval
from app.eval.adapter import compute_signature, make_mvp_retriever
from app.eval.dataset import load_golden, read_signature
from app.services.kb_store import get_store


def cmd_run(args: argparse.Namespace) -> None:
    store = get_store()
    sig = compute_signature(store)
    if sig.embedder_name == "hash" and not args.allow_hashing:
        raise SystemExit(
            "Refusing to produce a baseline on the hashing embedder (near-random "
            "results). Configure KB_EMBEDDINGS_BACKEND=ollama|api (+ model/base), or "
            "pass --allow-hashing for a throwaway smoke run."
        )
    golden_path = Path(args.golden)
    golden = load_golden(golden_path)
    gold_sig = read_signature(golden_path)
    if gold_sig is not None and gold_sig != sig:
        raise SystemExit(
            f"Corpus signature mismatch — golden was built against {gold_sig.to_dict()} "
            f"but the live corpus is {sig.to_dict()}. Regenerate the golden set."
        )
    retriever = make_mvp_retriever(store)
    retrieval = retrieval_eval.evaluate(golden, retriever)
    rep = report_mod.build_report(surface="mvp", signature=sig.to_dict(), retrieval=retrieval)
    report_mod.save_report(Path(args.out), rep)
    print(report_mod.to_markdown(rep))


def cmd_compare(args: argparse.Namespace) -> None:
    run_a = json.loads(Path(args.run_a).read_text(encoding="utf-8"))
    run_b = json.loads(Path(args.run_b).read_text(encoding="utf-8"))
    print(report_mod.compare(run_a, run_b))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval_rag")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="score retrieval on the MVP surface")
    run.add_argument("--golden", default="data/eval/golden_curated.jsonl")
    run.add_argument("--out", default="var/data/eval/run.json")
    run.add_argument("--allow-hashing", action="store_true")
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="diff two run JSONs")
    cmp.add_argument("run_a")
    cmp.add_argument("run_b")
    cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_cli.py -v`
Expected: PASS (2 tests). If `add_document` signature differed, you adjusted the test per the note.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_rag.py tests/test_eval_cli.py
git commit -m "feat(eval): eval_rag CLI 'run' (retrieval) + hashing/drift guards"
```

---

### Checkpoint A — first retrieval signal

- [ ] Author a tiny throwaway golden file against your real MVP DB (or generate one in Task 10) and run a smoke baseline:

```bash
py -3 -m scripts.eval_rag run --golden data/eval/golden_curated.jsonl --out var/data/eval/baseline_mvp.json --allow-hashing
```

- [ ] Confirm the Markdown table prints and the JSON is written. Do NOT trust the numbers yet if `embedder_name=hash` — that is exactly the cliff the guard warns about. This checkpoint only verifies the pipeline runs end-to-end.

---

## PR2 — Generation scoring (LLM-judge + deterministic refusal)

### Task 7: Judge (prompt + robust verdict parsing)

**Files:**
- Create: `app/eval/judge.py`
- Test: `tests/test_eval_judge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_judge.py
from app.eval.judge import build_judge_prompt, parse_verdict, Verdict


def test_build_prompt_includes_sections():
    p = build_judge_prompt(question="Q?", answer="A [1]", context="[1] текст", reference="эталон")
    assert "Q?" in p and "A [1]" in p and "[1] текст" in p and "эталон" in p


def test_parse_verdict_plain_json():
    v = parse_verdict('{"faithfulness":5,"relevance":4,"completeness":3,"citation":2,"rationale":"ok"}')
    assert v == Verdict(5, 4, 3, 2, "ok")
    assert v.normalized()["faithfulness"] == 1.0 and v.normalized()["citation"] == 0.25


def test_parse_verdict_tolerates_fence_and_prose():
    raw = "Вот оценка:\n```json\n{\"faithfulness\":1,\"relevance\":1,\"completeness\":1,\"citation\":1}\n```"
    v = parse_verdict(raw)
    assert v is not None and v.faithfulness == 1


def test_parse_verdict_clamps_out_of_range():
    v = parse_verdict('{"faithfulness":9,"relevance":0,"completeness":3,"citation":3}')
    assert v.faithfulness == 5 and v.relevance == 1


def test_parse_verdict_malformed_returns_none():
    assert parse_verdict("no json here") is None
    assert parse_verdict("") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval.judge'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/judge.py
"""LLM-as-judge: prompt construction + robust verdict parsing.

Parsing mirrors the tolerance of ``synthetic_qa.parse_qa_response`` (markdown
fences, surrounding prose). Scores are 1–5, normalized to [0,1] for aggregation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

JUDGE_SYSTEM = (
    "Ты — строгий оценщик ответов RAG-системы по корпоративным документам. "
    "Оцениваешь ответ только по предоставленному контексту. Возвращаешь строго JSON."
)

SCORE_KEYS: tuple[str, ...] = ("faithfulness", "relevance", "completeness", "citation")


def build_judge_prompt(*, question: str, answer: str, context: str, reference: str = "") -> str:
    ref_block = f"\nЭталонный ответ (для оценки полноты):\n{reference}\n" if reference else ""
    return (
        "Оцени ответ системы по шкале 1–5 по каждому критерию:\n"
        "- faithfulness: каждое утверждение подтверждается контекстом, нет выдумок;\n"
        "- relevance: ответ по существу вопроса;\n"
        "- completeness: ответ полон относительно эталона (если он дан);\n"
        "- citation: ссылки вида [N] соответствуют использованным фрагментам.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Контекст (фрагменты):\n{context}\n{ref_block}\n"
        f"Ответ системы:\n{answer}\n\n"
        'Верни строго JSON без пояснений: '
        '{"faithfulness":N,"relevance":N,"completeness":N,"citation":N,"rationale":"кратко"}'
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    faithfulness: int
    relevance: int
    completeness: int
    citation: int
    rationale: str = ""

    def normalized(self) -> dict[str, float]:
        return {k: (getattr(self, k) - 1) / 4.0 for k in SCORE_KEYS}


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _clamp(value: object) -> int:
    try:
        n = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, n))


def parse_verdict(raw: str) -> Verdict | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return Verdict(
        faithfulness=_clamp(data.get("faithfulness")),
        relevance=_clamp(data.get("relevance")),
        completeness=_clamp(data.get("completeness")),
        citation=_clamp(data.get("citation")),
        rationale=str(data.get("rationale", "")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_judge.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/eval/judge.py tests/test_eval_judge.py
git commit -m "feat(eval): LLM-judge prompt + robust verdict parsing"
```

---

### Task 8: Generation evaluator (answer → refusal/judge)

**Files:**
- Create: `app/eval/generation_eval.py`
- Test: `tests/test_eval_generation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_generation.py
from dataclasses import dataclass
from app.eval.adapter import EvalHit
from app.eval.dataset import GoldenItem
from app.eval.generation_eval import (
    RAG_SYSTEM_PROMPT, looks_like_refusal, evaluate_generation,
)


@dataclass
class _Resp:
    text: str


class _Provider:
    """Returns a canned text; records the last prompt it saw."""
    def __init__(self, text):
        self.text = text
        self.last_prompt = None

    name = "fake"
    model = "fake"

    def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
        self.last_prompt = prompt
        return _Resp(self.text)


def test_system_prompt_matches_production():
    # Drift guard: the eval must answer with the SAME system prompt as the MVP path.
    from app.api.kb_mvp import _RAG_SYSTEM_PROMPT
    assert RAG_SYSTEM_PROMPT == _RAG_SYSTEM_PROMPT


def test_looks_like_refusal():
    assert looks_like_refusal("Не удалось найти в документах информацию для ответа.")
    assert looks_like_refusal("Извините, я не могу ответить.")
    assert not looks_like_refusal("Отпуск — это перерыв [1].")


def test_refusal_item_scored_deterministically():
    items = [GoldenItem("Кто выиграл матч?", (), expect_refusal=True)]
    retriever = lambda q, k: [EvalHit(1, "нерелевантный текст")]
    gen = _Provider("Не удалось найти в документах информацию для ответа.")
    judge = _Provider("{}")  # must not be consulted for refusal items
    out = evaluate_generation(items, retriever, gen_provider=gen, judge_provider=judge, top_k=5)
    assert out["aggregate"]["refusal_correct"] == 1.0
    assert out["n_refusal"] == 1 and out["n_answerable"] == 0
    assert judge.last_prompt is None


def test_answerable_item_uses_judge():
    items = [GoldenItem("Что такое отпуск?", (7,), reference_answer="перерыв")]
    retriever = lambda q, k: [EvalHit(7, "Отпуск — перерыв.")]
    gen = _Provider("Отпуск — это перерыв [1].")
    judge = _Provider('{"faithfulness":5,"relevance":5,"completeness":4,"citation":5}')
    out = evaluate_generation(items, retriever, gen_provider=gen, judge_provider=judge, top_k=5)
    assert out["aggregate"]["faithfulness"] == 1.0
    assert out["n_answerable"] == 1
    # The judge saw the generated answer and the retrieved context.
    assert "Отпуск — это перерыв [1]." in judge.last_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_generation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval.generation_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/eval/generation_eval.py
"""Score end-to-end answers: deterministic refusal-correctness + LLM-judge.

The answer prompt mirrors the production MVP path (``kb_mvp._build_rag_prompt``
+ ``_RAG_SYSTEM_PROMPT``). ``RAG_SYSTEM_PROMPT`` is pinned equal to the
production constant by a drift test rather than imported at runtime, so this
module stays free of the FastAPI import chain.
"""
from __future__ import annotations

from typing import Sequence

from app.eval.adapter import EvalHit, Retriever
from app.eval.dataset import GoldenItem
from app.eval.judge import JUDGE_SYSTEM, build_judge_prompt, parse_verdict
from app.eval.metrics import aggregate
from app.services.synthetic_qa import LLMProvider, is_refusal

# MUST stay byte-identical to app.api.kb_mvp._RAG_SYSTEM_PROMPT (drift-tested).
RAG_SYSTEM_PROMPT = (
    "Ты — помощник корпоративной базы знаний. Отвечай на русском. "
    "Используй ТОЛЬКО фрагменты из контекста, не выдумывай факты. "
    "Если данных недостаточно — честно сообщи об этом. "
    "В ответе ссылайся на фрагменты в формате [1], [2] там, где они уместны."
)

_CANONICAL_REFUSAL = "не удалось найти"


def looks_like_refusal(text: str) -> bool:
    return is_refusal(text) or _CANONICAL_REFUSAL in (text or "").lower()


def format_context(hits: Sequence[EvalHit]) -> str:
    # Mirrors kb_mvp._format_context: "[i] <source>\n<text>" joined by separators.
    parts = []
    for i, h in enumerate(hits, start=1):
        label = h.title or "фрагмент"
        parts.append(f"[{i}] {label}\n{h.text}")
    return "\n\n---\n\n".join(parts)


def _build_answer_prompt(question: str, context: str) -> str:
    return f"Фрагменты базы знаний:\n{context}\n\nВопрос пользователя: {question}\nОтвет:"


def _generate(provider: LLMProvider, prompt: str, system: str) -> str:
    resp = provider.generate(prompt, system=system)
    return getattr(resp, "text", "") or ""


def evaluate_generation_item(
    item: GoldenItem,
    hits: Sequence[EvalHit],
    *,
    gen_provider: LLMProvider,
    judge_provider: LLMProvider,
) -> dict[str, float]:
    context = format_context(hits)
    answer = _generate(gen_provider, _build_answer_prompt(item.question, context), RAG_SYSTEM_PROMPT)
    if item.expect_refusal:
        return {"refusal_correct": 1.0 if looks_like_refusal(answer) else 0.0}
    jprompt = build_judge_prompt(
        question=item.question, answer=answer, context=context, reference=item.reference_answer
    )
    verdict = parse_verdict(_generate(judge_provider, jprompt, JUDGE_SYSTEM))
    return verdict.normalized() if verdict else {}


def evaluate_generation(
    items: Sequence[GoldenItem],
    retriever: Retriever,
    *,
    gen_provider: LLMProvider,
    judge_provider: LLMProvider,
    top_k: int,
) -> dict[str, object]:
    judge_rows: list[dict[str, float]] = []
    refusal_rows: list[dict[str, float]] = []
    per_item: list[dict[str, object]] = []
    for item in items:
        hits = list(retriever(item.question, top_k))
        row = evaluate_generation_item(
            item, hits, gen_provider=gen_provider, judge_provider=judge_provider
        )
        per_item.append({"question": item.question, "expect_refusal": item.expect_refusal, **row})
        if item.expect_refusal:
            refusal_rows.append(row)
        elif row:
            judge_rows.append(row)

    agg: dict[str, float] = {}
    if judge_rows:
        agg.update(aggregate(judge_rows))
    if refusal_rows:
        agg["refusal_correct"] = aggregate(refusal_rows)["refusal_correct"]

    return {
        "n_answerable": sum(1 for i in items if not i.expect_refusal),
        "n_refusal": sum(1 for i in items if i.expect_refusal),
        "per_item": per_item,
        "aggregate": agg,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_generation.py -v`
Expected: PASS (4 tests). If `test_system_prompt_matches_production` fails, copy the exact current text of `app.api.kb_mvp._RAG_SYSTEM_PROMPT` into `RAG_SYSTEM_PROMPT`.

- [ ] **Step 5: Commit**

```bash
git add app/eval/generation_eval.py tests/test_eval_generation.py
git commit -m "feat(eval): generation evaluator (refusal + LLM-judge)"
```

---

### Task 9: Wire generation into CLI `run` + `--judge`

**Files:**
- Modify: `scripts/eval_rag.py` (extend `cmd_run` and `build_parser`)
- Test: `tests/test_eval_cli.py` (add a case)

- [ ] **Step 1: Write the failing test (append to tests/test_eval_cli.py)**

```python
def test_run_includes_generation_when_judge_enabled(tmp_path, monkeypatch):
    import scripts.eval_rag as cli
    from app.eval.dataset import GoldenItem

    # Reuse the real-store helper from earlier in this file.
    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    class _Resp:
        def __init__(self, t): self.text = t
    class _Prov:
        name = model = "fake"
        def __init__(self, t): self._t = t
        def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
            return _Resp(self._t)

    # gen answers; judge returns top marks
    monkeypatch.setattr(cli, "_gen_provider", lambda: _Prov("Отпуск — перерыв [1]."))
    monkeypatch.setattr(cli, "_judge_provider", lambda: _Prov(
        '{"faithfulness":5,"relevance":5,"completeness":5,"citation":5}'))

    golden = tmp_path / "g.jsonl"
    golden.write_text(GoldenItem("Что такое отпуск?", (1,), "перерыв").to_jsonl_line(), encoding="utf-8")
    out = tmp_path / "run.json"
    cli.cmd_run(cli.build_parser().parse_args(
        ["run", "--golden", str(golden), "--out", str(out), "--allow-hashing", "--judge"]))
    import json
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["generation"]["faithfulness"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_cli.py::test_run_includes_generation_when_judge_enabled -v`
Expected: FAIL (`AttributeError: module 'scripts.eval_rag' has no attribute '_gen_provider'` or `unrecognized arguments: --judge`)

- [ ] **Step 3: Modify implementation**

In `scripts/eval_rag.py`, add provider seams + generation wiring. Add these imports near the top:

```python
from app.eval import generation_eval
from app.services.kb_llm import select_provider
```

Add seam helpers (kept as module functions so tests can monkeypatch them):

```python
def _gen_provider():
    provider = select_provider()
    if provider is None:
        raise SystemExit("No LLM provider configured for generation (set DEEPSEEK_API_KEY etc.).")
    return provider


def _judge_provider():
    # Same provider family by default; override via env in a later PR if needed.
    return _gen_provider()
```

Replace the tail of `cmd_run` (after `retrieval = retrieval_eval.evaluate(...)`) with:

```python
    generation = None
    if getattr(args, "judge", False):
        generation = generation_eval.evaluate_generation(
            golden, retriever,
            gen_provider=_gen_provider(),
            judge_provider=_judge_provider(),
            top_k=max(retrieval_eval.RETRIEVAL_KS) if hasattr(retrieval_eval, "RETRIEVAL_KS") else 10,
        )
    rep = report_mod.build_report(
        surface="mvp", signature=sig.to_dict(), retrieval=retrieval, generation=generation
    )
    report_mod.save_report(Path(args.out), rep)
    print(report_mod.to_markdown(rep))
```

> `top_k` for generation: import `RETRIEVAL_KS` from `app.eval.metrics` instead of the guarded `hasattr`. Add `from app.eval.metrics import RETRIEVAL_KS` and use `top_k=max(RETRIEVAL_KS)`. (Simplify the line above accordingly.)

Add the `--judge` flag in `build_parser`, inside the `run` subparser:

```python
    run.add_argument("--judge", action="store_true", help="also score generation via LLM-judge")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_cli.py -v`
Expected: PASS (all cases)

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_rag.py tests/test_eval_cli.py
git commit -m "feat(eval): wire generation scoring into 'run --judge'"
```

---

## PR3 — Golden-set generation + curated set

### Task 10: CLI `generate` + curated golden file

**Files:**
- Modify: `scripts/eval_rag.py` (add `cmd_generate` + subparser)
- Create: `data/eval/golden_curated.jsonl` (committed)
- Test: `tests/test_eval_cli.py` (add a generate case with a fake teacher)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_generate_builds_golden_from_corpus(tmp_path, monkeypatch):
    import scripts.eval_rag as cli
    from app.services import synthetic_qa as sq

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    class _Resp:
        def __init__(self, t): self.text = t
    class _Teacher:
        name = "deepseek"; model = "deepseek-chat"
        def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
            return _Resp('{"instruction":"Что такое отпуск?","input":"","output":"перерыв [doc_chunk:1]"}')

    monkeypatch.setattr(cli, "_gen_provider", lambda: _Teacher())
    # Disable self-consistency double-call for a deterministic single-shot test.
    monkeypatch.setattr(sq, "SyntheticQAGenerator",
                        lambda provider: sq.SyntheticQAGenerator(provider=provider, check_self_consistency=False))

    out = tmp_path / "golden_auto.jsonl"
    cli.cmd_generate(cli.build_parser().parse_args(
        ["generate", "--out", str(out), "--limit", "5", "--budget-usd", "100", "--yes"]))

    from app.eval.dataset import load_golden, read_signature
    items = load_golden(out)
    assert items and items[0].question == "Что такое отпуск?"
    assert items[0].source == "auto" and items[0].relevant_chunk_ids  # tied to a real chunk id
    assert read_signature(out) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_cli.py::test_generate_builds_golden_from_corpus -v`
Expected: FAIL (`unrecognized arguments: generate` / no `cmd_generate`)

- [ ] **Step 3: Modify implementation**

Add to `scripts/eval_rag.py`:

```python
from app.eval.adapter import compute_signature  # already imported; keep single import
from app.eval.dataset import GoldenItem, save_golden, write_signature
from app.services import synthetic_qa as sq


def cmd_generate(args: argparse.Namespace) -> None:
    store = get_store()
    provider = _gen_provider()
    chunks = list(sq.iter_chunks(store))
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        raise SystemExit("Corpus is empty — ingest documents before generating a golden set.")

    cost = sq.estimate_total_cost_usd(
        provider=provider.name, model=provider.model,
        mode=sq.GenerationMode.SINGLE, chunk_chars=[len(t) for _, t in chunks],
    )
    if cost is None:
        print(f"WARNING: no pricing for ({provider.name}, {provider.model}); cost guard disabled.")
    elif cost > args.budget_usd and not args.yes:
        raise SystemExit(
            f"Estimated cost ${cost:.2f} exceeds --budget-usd ${args.budget_usd:.2f}. "
            f"Re-run with --yes to proceed."
        )

    generator = sq.SyntheticQAGenerator(provider=provider)
    items: list[GoldenItem] = []
    for chunk_id, text in chunks:
        for pair in generator.generate_for_chunk(
            chunks=[text], chunk_ids=[chunk_id], mode=sq.GenerationMode.SINGLE
        ):
            items.append(GoldenItem(
                question=pair.instruction,
                relevant_chunk_ids=(pair.source_chunk_id,),
                reference_answer=pair.output,
                expect_refusal=False,
                source="auto",
            ))

    out = Path(args.out)
    save_golden(out, items)
    write_signature(out, compute_signature(store))
    print(f"Wrote {len(items)} golden items + signature to {out}")
```

Add the subparser in `build_parser`:

```python
    gen = sub.add_parser("generate", help="build a golden set from the corpus")
    gen.add_argument("--out", default="var/data/eval/golden_auto.jsonl")
    gen.add_argument("--limit", type=int, default=0, help="max chunks to process (0 = all)")
    gen.add_argument("--budget-usd", type=float, default=5.0)
    gen.add_argument("--yes", action="store_true", help="proceed past the budget guard")
    gen.set_defaults(func=cmd_generate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_cli.py -v`
Expected: PASS (all cases)

- [ ] **Step 5: Create the committed curated golden file**

Create `data/eval/golden_curated.jsonl`. The `expect_refusal` items below are corpus-independent and committable as-is. **Replace the positive example's `relevant_chunk_ids` with real `kb_chunks.id` values from your corpus** — find them with:
`py -3 -c "from app.services.kb_store import get_store; [print(i,t[:60]) for i,t in __import__('app.services.synthetic_qa',fromlist=['iter_chunks']).iter_chunks(get_store())]"`

```jsonl
{"instruction": "Сколько дней основного оплачиваемого отпуска положено сотруднику?", "input": "", "output": "ЗАМЕНИ: эталонный ответ из вашего корпуса с [N].", "meta": {"relevant_chunk_ids": [1], "expect_refusal": false, "source": "curated"}}
{"instruction": "Какая температура на поверхности Венеры?", "input": "", "output": "", "meta": {"relevant_chunk_ids": [], "expect_refusal": true, "source": "curated"}}
{"instruction": "Кто победил в матче вчера вечером?", "input": "", "output": "", "meta": {"relevant_chunk_ids": [], "expect_refusal": true, "source": "curated"}}
{"instruction": "Назови рецепт борща из нашей базы знаний.", "input": "", "output": "", "meta": {"relevant_chunk_ids": [], "expect_refusal": true, "source": "curated"}}
```

> The curated file must contain ≥3 `expect_refusal` items (above) and ≥1 real positive item authored against the corpus. Add more positive items by picking real questions and looking up their chunk id(s).

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_rag.py tests/test_eval_cli.py data/eval/golden_curated.jsonl
git commit -m "feat(eval): 'generate' golden set from corpus + curated seed file"
```

---

### Checkpoint B — baseline + STOP for review

- [ ] Ensure a **real** embedder is configured (`KB_EMBEDDINGS_BACKEND=ollama` + `OLLAMA_EMBED_MODEL`, or `=api` + `EMBEDDINGS_API_BASE_URL`). Reindex if you changed it (CLAUDE.md: `py -3 -m scripts.kb_cli reindex`).
- [ ] Generate the auto golden set and run the full baseline:

```bash
py -3 -m scripts.eval_rag generate --out var/data/eval/golden_auto.jsonl --limit 200
py -3 -m scripts.eval_rag run --golden var/data/eval/golden_auto.jsonl --out var/data/eval/baseline_auto.json --judge
py -3 -m scripts.eval_rag run --golden data/eval/golden_curated.jsonl --out var/data/eval/baseline_curated.json --judge
```

- [ ] **Read the reports.** Record `hit@k`/`recall@k`/`mrr@k`, the four judge metrics, and `refusal_correct`.
- [ ] **Do not start Phase 2 yet.** The exact quick-win parameters (e.g. which `top_k`, whether e5-prefix helps, latency of `bge-reranker-v2-m3`) depend on these numbers. Bring the baseline back and write the Phase 2 plan against it.

---

## Phase 2 — deferred (PR4–PR8, planned after the baseline)

These modify production retrieval and are **data-dependent**; they get their own detailed TDD plan once Checkpoint B numbers exist. Captured here as procedure + gate so nothing is lost:

- **PR4 — Embedder reality guard.** Already partially shipped in Task 6 (`run` refuses hashing). Promote to a reusable check and document real-embedder setup. *Gate: prerequisite.*
- **PR5 — e5 `query:`/`passage:` prefixes.** Add a `VECTOR_E5_PREFIX` flag; prefix queries with `"query: "` and passages with `"passage: "` in `app/retriever/qdrant.py:_batched_encode` (`:131`) / `faiss.py` (`:105`) + the v1 ingest path. **Requires reindex.** Also build the **v1 content-match adapter** here (Deviation 1). *Gate: keep iff `recall@k`/`mrr@k` improve on a reindexed snapshot (`compare`).*
- **PR6 — Russian reranker.** Default `KB_RERANK_MODEL`/v1 reranker → `BAAI/bge-reranker-v2-m3`; align `KB_RERANK_ENABLED`/`RERANK_ENABLED`. Query-time only, no reindex. *Gate: keep iff `mrr@k`/`hit@5` improve and latency acceptable.*
- **PR7 — top_k / context-budget tuning.** Sweep `top_k ∈ {5,8,10,12}` and the v1 3000-token budget. *Gate: completeness↑ without faithfulness↓.*
- **PR8 — Prompt tightening.** Sharpen grounding + citation discipline in `kb_mvp._RAG_SYSTEM_PROMPT` (`:406`) and the v1 orchestrator prompt; re-pin the `generation_eval.RAG_SYSTEM_PROMPT` drift test. *Gate: `faithfulness` + `refusal_correct`.*

---

## Self-review (completed)

- **Spec coverage:** §4 layout → Tasks 1–6,7–10; §5 golden → Tasks 2,10; §6 metrics → Tasks 1,7,8; §7 runner/report → Tasks 5,6,9; §8 quick-wins → Phase 2; §9 guardrails → Tasks 6 (hashing/drift), 8 (offline fakes), 10 (cost guard); §10 PR staging → PR1/PR2/PR3 headers + Phase 2.
- **Placeholders:** none in code steps; the curated `.jsonl` positive line is explicitly a data-entry task with a lookup command (corpus-specific by nature), and its refusal items are real/committable.
- **Type consistency:** `EvalHit(chunk_id, text, title)`, `GoldenItem(question, relevant_chunk_ids, reference_answer, expect_refusal, source)`, `Verdict(faithfulness, relevance, completeness, citation, rationale)`, `CorpusSignature(doc_count, max_chunk_id, embedder_name, dim)`, and `Retriever = Callable[[str,int], Sequence[EvalHit]]` are used identically across all tasks. `RETRIEVAL_KS` is imported from `app.eval.metrics` (the Task 9 note removes the `hasattr` shortcut). `provider.generate(prompt, *, system=...)` matches the verified `kb_llm` signature.
