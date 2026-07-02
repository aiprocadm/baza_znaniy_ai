# Pravo Reranker Candidate/Latency Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how reranker quality and CPU latency vary with `KB_RERANK_CANDIDATES` on `golden_pravo_natural`, and record a revised, achievable p95 budget that preserves the Track A quality win.

**Architecture:** One instrumented pass captures each query's top-20 bi-encoder shortlist (chunk keys) + per-candidate bge score into a committed fixture. Quality for any k≤20 is then reconstructed offline (deterministic, no model) by re-sorting `shortlist[:k]` by teacher score, mirroring the production `candidates=k` semantics and the `rerank.py` tie-break. Latency is measured live with the real bge model, single-process.

**Tech Stack:** Python, pytest, existing `app/eval/*` harness (`metrics.py`, `dataset.py`, `adapter.py`), `app/services/kb_store`, `sentence-transformers` `CrossEncoder` (`BAAI/bge-reranker-v2-m3`, offline/local only), reused latency helpers from `scripts/bench_reranker.py`.

**Spec:** `docs/superpowers/specs/2026-07-02-pravo-reranker-candidate-latency-sweep-design.md`

**Conventions (this repo, Windows):** run Python via `py -3.13` (not bare `py -3`). No venv. Lint/style: `py -3.13 -m ruff check .` and `py -3.13 -m black --check .`. Comments in code in Russian; chat explanations in Russian.

---

## File Structure

- Create: `app/eval/candidate_sweep.py` — pure sweep logic (reconstruct "rerank top-k", aggregate base/teacher metrics per candidate count). No I/O, no model.
- Create: `tests/test_candidate_sweep.py` — deterministic unit tests for the pure logic.
- Create: `scripts/sweep_rerank_candidates.py` — instrumented pass (store + bge) → writes fixture + prints quality table; `--latency` adds a single-process p50/p95 table.
- Create: `tests/scripts/test_sweep_rerank_candidates.py` — unit tests for the script's pure helpers (`_parse_ks`, `_fixture_items`).
- Create (generated): `data/eval/rerank_sweep_pravo.json` — committed fixture (keys + teacher scores + relevant per query).
- Modify: `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md` — append the Track B Pareto section + revised budget.
- Modify (conditional, data-driven): `scripts/bench_reranker.py:58` (budget default) and/or `app/services/kb_rerank.py:33` (`DEFAULT_CANDIDATES`) — only if the measured plateau justifies it (Task 6).

---

## Task 0: Environment prep

**Files:** none (setup only — `var/` is untracked).

The sibling worktree holds a ready pravo store matching the golden signature
(6141 docs / 14231 chunks / embedder `st` / dim 384). Copy it in so the script's
default `--store var/data/pravo_public.sqlite` works without an absolute path.

- [ ] **Step 1: Copy the store into this worktree**

Run (Git Bash):
```bash
mkdir -p var/data
cp "D:/Кодинг/1. База знаний/baza_znaniy_ai-main/var/data/pravo_public.sqlite" var/data/pravo_public.sqlite
```
Expected: `var/data/pravo_public.sqlite` present (~72 MB).

- [ ] **Step 2: Verify the store signature matches the golden**

Run:
```bash
py -3.13 -c "import sqlite3; c=sqlite3.connect('file:var/data/pravo_public.sqlite?mode=ro',uri=True); print('docs',c.execute('SELECT COUNT(*) FROM kb_documents').fetchone()[0],'chunks',c.execute('SELECT COUNT(*) FROM kb_chunks').fetchone()[0])"
```
Expected: `docs 6141 chunks 14231`.

> If counts differ, STOP — the store does not match `golden_pravo_natural` and the
> sweep would be invalid. Re-locate the correct store before proceeding.

---

## Task 1: Pure sweep logic module

**Files:**
- Create: `app/eval/candidate_sweep.py`
- Test: `tests/test_candidate_sweep.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_candidate_sweep.py`:
```python
"""Юнит-тесты чистой логики свипа кандидатов (без модели, детерминированно)."""

from __future__ import annotations

from app.eval.candidate_sweep import base_topk, rerank_topk, sweep_quality


def test_base_topk_truncates_in_order() -> None:
    keys = ["a", "b", "c", "d"]
    assert base_topk(keys, 2) == ["a", "b"]
    assert base_topk(keys, 10) == ["a", "b", "c", "d"]  # k>len → весь список
    assert base_topk(keys, 0) == []


def test_rerank_topk_sorts_by_teacher_score_desc() -> None:
    # би-энкодер: a,b,c ; teacher поднимает c (0.9) над a (0.5), b (0.1)
    keys = ["a", "b", "c"]
    scores = [0.5, 0.1, 0.9]
    assert rerank_topk(keys, scores, 3) == ["c", "a", "b"]
    # реранк только top-2 → c не в шорт-листе, порядок среди {a,b} по скору
    assert rerank_topk(keys, scores, 2) == ["a", "b"]
    # k=1 → ровно один ключ (одиночку реранк не меняет)
    assert rerank_topk(keys, scores, 1) == ["a"]


def test_rerank_topk_tie_break_by_original_position() -> None:
    # равные скоры → меньший исходный индекс раньше (зеркалит rerank.py:141)
    keys = ["a", "b", "c"]
    scores = [0.5, 0.5, 0.5]
    assert rerank_topk(keys, scores, 3) == ["a", "b", "c"]


def test_sweep_quality_shows_teacher_lift_at_full_k() -> None:
    # 2 вопроса; base ставит золото на позицию 2, teacher — на 1
    items = [
        {"relevant": ["g1"], "shortlist_keys": ["x", "g1", "y"], "teacher_scores": [0.1, 0.9, 0.2]},
        {"relevant": ["g2"], "shortlist_keys": ["z", "g2", "w"], "teacher_scores": [0.1, 0.9, 0.2]},
    ]
    table = sweep_quality(items, [1, 3])
    # k=3: teacher поднял золото на 1 → hit@1 teacher=1.0, base=0.0
    assert table[3]["teacher"]["hit@1"] == 1.0
    assert table[3]["base"]["hit@1"] == 0.0
    # k=1: шорт-лист = [x]/[z] — золота нет вообще → оба hit@1 = 0.0
    assert table[1]["teacher"]["hit@1"] == 0.0
    assert table[1]["base"]["hit@1"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/test_candidate_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.eval.candidate_sweep'`.

- [ ] **Step 3: Write the module**

Create `app/eval/candidate_sweep.py`:
```python
"""Чистая логика свипа «число кандидатов reranker'а ↔ качество» (без модели, без I/O).

Реконструирует ранжирование «реранк только top-k кандидатов» из уже захваченного
top-N шорт-листа би-энкодера и teacher-скор (bge) на кандидата, поэтому весь свип по
числу кандидатов исполняется детерминированно, не загружая модель. Зеркалит
прод-семантику ``KB_RERANK_CANDIDATES`` (``app/services/kb_rerank.py``: би-энкодер
достаёт ровно ``candidates`` хитов, cross-encoder скорит их все) и тай-брейк реранка
(``app/retriever/rerank.py``: sort по ``(score, -index)`` убыв.).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.eval.metrics import RETRIEVAL_KS, aggregate, score_item


def base_topk(shortlist_keys: Sequence[str], k: int) -> list[str]:
    """Ранжирование без реранка: первые ``k`` ключей в порядке би-энкодера."""
    return list(shortlist_keys[: max(0, k)])


def rerank_topk(
    shortlist_keys: Sequence[str],
    teacher_scores: Sequence[float],
    k: int,
) -> list[str]:
    """Ранжирование «реранк top-k»: ``shortlist_keys[:k]``, отсортированный по
    соответствующему teacher-скору убыв., тай-брейк по исходной позиции.

    Зеркалит ``rerank.py``: ``sort(key=(score, -index), reverse=True)`` — при равных
    скорах меньший исходный индекс идёт раньше. Кандидаты за позицией ``k`` в прод
    вообще не достаются (би-энкодер тянет ровно ``candidates``), поэтому они не
    участвуют.
    """
    cut = max(0, k)
    keys = list(shortlist_keys[:cut])
    scores = list(teacher_scores[:cut])
    order = sorted(range(len(keys)), key=lambda i: (scores[i], -i), reverse=True)
    return [keys[i] for i in order]


def sweep_quality(
    items: Sequence[Mapping[str, Any]],
    candidate_ks: Sequence[int],
    metric_ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[int, dict[str, dict[str, float]]]:
    """На каждое число кандидатов из ``candidate_ks`` — агрегированные метрики base и teacher.

    ``item`` содержит: ``relevant`` (релевантные ключи), ``shortlist_keys`` (top-N
    порядок би-энкодера), ``teacher_scores`` (скор bge на кандидата, той же длины и
    порядка, что ``shortlist_keys``). Возврат: ``{k: {"base": {...}, "teacher": {...}}}``,
    где значения — усреднённые по вопросам метрики из ``app.eval.metrics``.
    """
    out: dict[int, dict[str, dict[str, float]]] = {}
    for k in candidate_ks:
        base_rows: list[dict[str, float]] = []
        teacher_rows: list[dict[str, float]] = []
        for it in items:
            relevant = it["relevant"]
            keys = it["shortlist_keys"]
            scores = it["teacher_scores"]
            base_rows.append(score_item(relevant, base_topk(keys, k), metric_ks))
            teacher_rows.append(score_item(relevant, rerank_topk(keys, scores, k), metric_ks))
        out[k] = {"base": aggregate(base_rows), "teacher": aggregate(teacher_rows)}
    return out


__all__ = ["base_topk", "rerank_topk", "sweep_quality"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_candidate_sweep.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint + commit**

Run: `py -3.13 -m ruff check app/eval/candidate_sweep.py tests/test_candidate_sweep.py && py -3.13 -m black app/eval/candidate_sweep.py tests/test_candidate_sweep.py`
Then:
```bash
git add app/eval/candidate_sweep.py tests/test_candidate_sweep.py
git commit -m "feat(eval): pure candidate-sweep logic (rerank top-k reconstruction)"
```

---

## Task 2: Instrumented sweep script

**Files:**
- Create: `scripts/sweep_rerank_candidates.py`
- Test: `tests/scripts/test_sweep_rerank_candidates.py`

- [ ] **Step 1: Write the failing tests for the pure helpers**

Create `tests/scripts/test_sweep_rerank_candidates.py`:
```python
"""Юнит-тесты чистых хелперов свип-скрипта (без модели/стора)."""

from __future__ import annotations

from scripts.sweep_rerank_candidates import _fixture_items, _parse_ks


def test_parse_ks_splits_and_ints() -> None:
    assert _parse_ks("1,2,3,5,8,10,12,16,20") == [1, 2, 3, 5, 8, 10, 12, 16, 20]
    assert _parse_ks("5") == [5]
    assert _parse_ks("1, ,2 ,") == [1, 2]  # пустые токены игнорируются


def test_fixture_items_drops_question_and_texts() -> None:
    items = [
        {
            "question": "q",
            "relevant": ["g1"],
            "shortlist_keys": ["a", "g1"],
            "teacher_scores": [0.1, 0.9],
            "_texts": ["ta", "tg1"],
        }
    ]
    out = _fixture_items(items)
    assert out == [{"relevant": ["g1"], "shortlist_keys": ["a", "g1"], "teacher_scores": [0.1, 0.9]}]
    assert "question" not in out[0] and "_texts" not in out[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_sweep_rerank_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.sweep_rerank_candidates'`.

- [ ] **Step 3: Write the script**

Create `scripts/sweep_rerank_candidates.py`:
```python
"""Свип «число кандидатов reranker'а ↔ качество/латентность» на корпусе права.

Один инструментированный проход захватывает top-N шорт-лист би-энкодера и teacher-скор
(bge) на каждого кандидата → пишет коммитимый фикстур ``data/eval/rerank_sweep_pravo.json``.
Качество «реранк top-k» реконструируется офлайн (``app/eval/candidate_sweep.py``). С
флагом ``--latency`` дополнительно мерит p50/p95 реального bge на CPU по каждому k,
single-process, с прогревом (переиспользуя ``scripts.bench_reranker``).

Запуск (Windows; эмбеддер st обязателен — хэш-эмбеддер даёт мусор):
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.sweep_rerank_candidates
    KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.sweep_rerank_candidates --latency
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from app.eval.candidate_sweep import sweep_quality

DEFAULT_STORE = "var/data/pravo_public.sqlite"
DEFAULT_GOLDEN = "data/eval/golden_pravo_natural.jsonl"
DEFAULT_OUT = "data/eval/rerank_sweep_pravo.json"
DEFAULT_KS = "1,2,3,5,8,10,12,16,20"
DEFAULT_SHORTLIST = 20
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_HEADLINE = ("hit@1", "hit@3", "hit@5", "mrr@5", "recall@10")


def _parse_ks(text: str) -> list[int]:
    """'1,2, ,5' -> [1, 2, 5] (пустые токены игнорируются)."""
    return [int(tok) for tok in text.split(",") if tok.strip()]


def _fixture_items(items: Sequence[dict]) -> list[dict]:
    """Проекция для коммита: только поля, нужные для офлайн-качества."""
    return [
        {
            "relevant": it["relevant"],
            "shortlist_keys": it["shortlist_keys"],
            "teacher_scores": it["teacher_scores"],
        }
        for it in items
    ]


def capture_items(store: Any, golden: Sequence[Any], model: Any, shortlist: int) -> list[dict]:
    """Один проход модели: на вопрос — top-N шорт-лист (ключи+тексты) + teacher-скор.

    Тексты остаются в items в памяти (для ``--latency``), но в фикстур не пишутся.
    """
    from app.eval.adapter import make_mvp_retriever

    base = make_mvp_retriever(store)
    items: list[dict] = []
    for it in golden:
        hits = base(it.question, shortlist)
        texts = [h.text for h in hits]
        scores = [float(s) for s in model.predict([(it.question, t) for t in texts])]
        items.append(
            {
                "question": it.question,
                "relevant": list(it.relevant_chunks),
                "shortlist_keys": [h.chunk_key for h in hits],
                "teacher_scores": scores,
                "_texts": texts,
            }
        )
    return items


def print_quality(items: Sequence[dict], candidate_ks: Sequence[int]) -> None:
    table = sweep_quality(items, candidate_ks)
    print(f"\nquality sweep (n={len(items)}):")
    print("  " + f"{'k':>4} | " + " ".join(f"{m:>10}" for m in _HEADLINE))
    for k in candidate_ks:
        t = table[k]["teacher"]
        print("  " + f"{k:>4} | " + " ".join(f"{t[m]:>10.3f}" for m in _HEADLINE))
    base_row = table[candidate_ks[-1]]["base"]
    print("  " + f"{'base':>4} | " + " ".join(f"{base_row[m]:>10.3f}" for m in _HEADLINE))


def print_latency(
    items: Sequence[dict],
    candidate_ks: Sequence[int],
    model: Any,
    budget_ms: float,
    warmup: int,
) -> None:
    from scripts.bench_reranker import measure, percentile

    queries = [(it["question"], it["_texts"]) for it in items if it["_texts"]]
    warm = queries[:warmup]
    timed = queries[warmup:] or queries
    print(f"\nlatency sweep (n={len(timed)}, budget {budget_ms:.0f}ms, single-process):")
    print("  " + f"{'k':>4} | {'p50(ms)':>9} {'p95(ms)':>9}  verdict")
    for k in candidate_ks:
        if warm:
            measure(model.predict, warm, candidates=k)  # прогрев, отбрасывается
        timings = measure(model.predict, timed, candidates=k)
        p50 = percentile(timings, 0.50)
        p95 = percentile(timings, 0.95)
        verdict = "PASS" if p95 <= budget_ms else "FAIL"
        print("  " + f"{k:>4} | {p50:>9.0f} {p95:>9.0f}  {verdict}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=os.environ.get("KB_RERANK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--ks", default=DEFAULT_KS)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--budget-ms", type=float, default=200.0)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)

    candidate_ks = _parse_ks(args.ks)

    # Стор выбирается через env до первого get_store(); сбрасываем кэш стора.
    os.environ["KB_MVP_DB_PATH"] = str(Path(args.store))
    from app.services.kb_store import get_store, reset_default_store

    reset_default_store()
    store = get_store()

    from app.eval.adapter import compute_signature
    from app.eval.dataset import load_golden

    golden = load_golden(Path(args.golden))
    if not golden:
        raise SystemExit(f"empty golden: {args.golden}")

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.model)

    items = capture_items(store, golden, model, args.shortlist)

    fixture = {
        "_sig": compute_signature(store).to_dict(),
        "_measured": {"shortlist": args.shortlist, "ks": candidate_ks},
        "items": _fixture_items(items),
    }
    Path(args.out).write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}  n={len(items)}  store={args.store}")

    print_quality(items, candidate_ks)
    if args.latency:
        print_latency(items, candidate_ks, model, args.budget_ms, args.warmup)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/scripts/test_sweep_rerank_candidates.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint + commit**

Run: `py -3.13 -m ruff check scripts/sweep_rerank_candidates.py tests/scripts/test_sweep_rerank_candidates.py && py -3.13 -m black scripts/sweep_rerank_candidates.py tests/scripts/test_sweep_rerank_candidates.py`
Then:
```bash
git add scripts/sweep_rerank_candidates.py tests/scripts/test_sweep_rerank_candidates.py
git commit -m "feat(eval): instrumented reranker candidate/latency sweep script"
```

---

## Task 3: Run the capture → fixture + quality numbers

**Files:**
- Generated: `data/eval/rerank_sweep_pravo.json`

This runs the REAL bge model once (offline, local). 36 queries × 20 candidates =
720 cross-encoder forwards + model load ≈ a couple of minutes on CPU. Run detached/
background if the foreground window is tight; the script is idempotent (overwrites
the fixture).

- [ ] **Step 1: Run the capture (quality only)**

Run (Git Bash; env inline):
```bash
KB_EMBEDDINGS_BACKEND=st ST_EMBED_MODEL=intfloat/multilingual-e5-small VECTOR_E5_PREFIX=1 \
KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3 \
py -3.13 -m scripts.sweep_rerank_candidates
```
Expected: `wrote data/eval/rerank_sweep_pravo.json  n=36  store=...` followed by a
quality table. Teacher `hit@1` should rise from very low at k=1 toward ~0.89 by
k=10–20; the `base` row (k=20) should read hit@1≈0.75, recall@10≈0.68 (matching the
Track A `_measured` base). Teacher hit@1 at k≥8 should sit near its k=20 value.

> If teacher hit@1 does not exceed base, or every row is identical, STOP — the
> reranker is not actually scoring (check `KB_EMBEDDINGS_BACKEND=st`, that
> `BAAI/bge-reranker-v2-m3` loaded, and that `store.search` returned texts). Do not
> commit a degenerate fixture.

- [ ] **Step 2: Sanity-check the fixture sig against the golden sig**

Run:
```bash
py -3.13 -c "import json; f=json.load(open('data/eval/rerank_sweep_pravo.json',encoding='utf-8')); g=json.load(open('data/eval/golden_pravo_natural.sig.json',encoding='utf-8')); print('SIG MATCH' if f['_sig']==g else ('MISMATCH', f['_sig'], g))"
```
Expected: `SIG MATCH`.

> If MISMATCH, the store differs from the corpus the golden was built against — the
> sweep is meaningless. Re-check Task 0 before committing.

- [ ] **Step 3: Commit the fixture**

```bash
git add data/eval/rerank_sweep_pravo.json
git commit -m "feat(eval): frozen candidate-sweep fixture (top-20 shortlist + bge scores)"
```

---

## Task 4: Measure latency per candidate count

**Files:** none (measurement only; numbers land in the runbook in Task 5).

- [ ] **Step 1: Run the latency sweep (single-process)**

Run:
```bash
KB_EMBEDDINGS_BACKEND=st ST_EMBED_MODEL=intfloat/multilingual-e5-small VECTOR_E5_PREFIX=1 \
KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3 \
py -3.13 -m scripts.sweep_rerank_candidates --latency
```
Expected: the same quality table, then a latency table with a `p50(ms)`/`p95(ms)`
row per k. p95 should grow ~linearly with k (each candidate ≈ one cross-encoder
forward); at k=20 expect p95 in the ~1000–1300 ms range (matching the runbook's
"~1.2 s for 20 candidates"), and PASS only at very small k against the 200 ms budget.

- [ ] **Step 2: Record the raw table**

Copy the printed quality + latency tables verbatim into a scratch note (used in Task
5). No commit here.

> Do NOT parallelize this measurement. Two torch processes on one CPU corrupt the
> p95. If the foreground run risks the ~10-min background-task kill, it is still one
> process — just let it finish; each k prints as it completes, so a partial table
> survives an interrupt.

---

## Task 5: Runbook — Pareto table + revised budget

**Files:**
- Modify: `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md`

- [ ] **Step 1: Derive the knee and the revised budget from the measured tables**

Apply these concrete rules to Task 3/4 output:
- **Knee k\*** = the smallest k where teacher `hit@1` AND teacher `mrr@5` are each
  within 0.01 (1 pp) of their k=20 values (quality plateau).
- **Revised p95 budget** = the measured p95 at k\*, rounded UP to a clean ceiling
  with ~20% headroom (e.g. measured 470 ms → budget 600 ms).
- **Quality cost line** = teacher hit@1/mrr@5 at k\* vs at k=20 (should be ≤1 pp).

- [ ] **Step 2: Append the runbook section**

Append to the end of `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md`
(fill the bracketed numbers from the measured tables — every bracket is a real number
from Task 3/4, not a placeholder to ship):
```markdown

## Candidate/latency sweep (Track B, 2026-07-02)

Quality vs CPU latency as a function of `KB_RERANK_CANDIDATES` (bge teacher over the
e5 base) on `golden_pravo_natural` (n=36). Quality is reconstructed offline from
`data/eval/rerank_sweep_pravo.json` (top-20 shortlist + per-candidate bge score);
latency is measured single-process with the real bge on this CPU.

| candidates (k) | hit@1 | hit@3 | mrr@5 | recall@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| 1  | [..] | [..] | [..] | [..] | [..] | [..] |
| 3  | [..] | [..] | [..] | [..] | [..] | [..] |
| 5  | [..] | [..] | [..] | [..] | [..] | [..] |
| 8  | [..] | [..] | [..] | [..] | [..] | [..] |
| 10 | [..] | [..] | [..] | [..] | [..] | [..] |
| 20 | [..] | [..] | [..] | [..] | [..] | [..] |
| base (no rerank) | [..] | [..] | [..] | [..] | 0 | 0 |

- **Knee:** at k=[k\*] the teacher holds hit@1=[..] / mrr@5=[..] — within [≤1] pp of
  the k=20 values ([..] / [..]) — i.e. reranking beyond [k\*] candidates buys ~0
  quality (base recall is near-ceiling), only latency.
- **Revised p95 budget:** the original 200 ms is unreachable for the 568M bge on CPU
  (p95 ≈ [..] ms even at k=[..]). Priority is to keep the Track A quality win, so the
  honest budget is **p95 ≤ [budget] ms at k=[k\*]** — this holds the full +[..] pp
  hit@1 domain gain at [p95 at k\*] ms. Sub-200 ms remains reserved for the distilled
  student (Phase 1 GPU run), which is the only structural path to the original budget.
- **`KB_RERANK_CANDIDATES`:** [kept at 20 — quality still climbing to k=20] /
  [lowered to [k\*] — the plateau makes 20 pure latency cost]. (Pick the branch the
  data supports; state which and why.)
- **Reproduce:** `KB_EMBEDDINGS_BACKEND=st py -3.13 -m scripts.sweep_rerank_candidates --latency`.
```

- [ ] **Step 3: Commit the runbook**

```bash
git add docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md
git commit -m "docs(eval): Track B candidate/latency Pareto + revised p95 budget"
```

---

## Task 6: Conditional default/budget adjustment (data-driven)

**Files (conditional):**
- Modify: `scripts/bench_reranker.py:58`
- Modify: `app/services/kb_rerank.py:33`

Only if Task 5 concluded a change is warranted. Two independent decisions:

- [ ] **Step 1: Revise the bench budget default (if the runbook set a new budget)**

If the revised budget differs from 200 ms, update the `bench_reranker.py` argument
default so the gate reflects the honest target. Change line 58 from:
```python
    parser.add_argument("--budget-ms", type=float, default=200.0)
```
to (example — use the runbook's `[budget]`):
```python
    # Реалистичный CPU-бюджет teacher-reranker'а (Track B, runbook 2026-06-15):
    # оригинальные 200 мс структурно недостижимы для 568M bge на CPU.
    parser.add_argument("--budget-ms", type=float, default=<budget>)
```

- [ ] **Step 2: Lower `DEFAULT_CANDIDATES` (ONLY if the plateau is strictly below 20)**

If and only if Task 5 found the knee k\* < 20 with ≤1 pp quality cost, lower the MVP
default so serving stops paying latency for candidates that add no quality. Change
`app/services/kb_rerank.py:33` from:
```python
DEFAULT_CANDIDATES = 20
```
to (example — use the runbook's `[k*]`):
```python
# Точка перегиба качества (Track B, runbook 2026-06-15): реранк сверх этого числа
# кандидатов не улучшает качество на golden_pravo_natural, только растит латентность.
DEFAULT_CANDIDATES = <k*>
```

> If the data shows quality still climbing at k=20, DO NOT change this — keep 20 and
> only revise the budget (Step 1). Record the "kept at 20" decision in the runbook.

- [ ] **Step 3: If either file changed, run its tests + commit**

Run: `py -3.13 -m pytest tests/scripts/test_bench_reranker.py tests/test_kb_rerank.py -v`
Expected: pass (or "no tests ran" for a file without a suite — that's acceptable).
Then (only if a file changed):
```bash
git add scripts/bench_reranker.py app/services/kb_rerank.py
git commit -m "chore(rerank): align default candidates/budget with Track B measurement"
```

> If Task 5 concluded no change (budget stays 200 for the gate, candidates stay 20),
> skip this whole task — the runbook already records the rationale.

---

## Task 7: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Lint + style on all new/changed files**

Run: `py -3.13 -m ruff check app/eval/candidate_sweep.py scripts/sweep_rerank_candidates.py tests/test_candidate_sweep.py tests/scripts/test_sweep_rerank_candidates.py && py -3.13 -m black --check app/eval/candidate_sweep.py scripts/sweep_rerank_candidates.py tests/test_candidate_sweep.py tests/scripts/test_sweep_rerank_candidates.py`
Expected: no errors; black reports "would reformat 0 files".

- [ ] **Step 2: Run the new pure-logic tests**

Run: `py -3.13 -m pytest tests/test_candidate_sweep.py tests/scripts/test_sweep_rerank_candidates.py -v`
Expected: 6 passed.

- [ ] **Step 3: Confirm no regression in the eval slice**

Run: `py -3.13 -m pytest tests/ -k "eval or frozen or rerank or sweep" -m "not integration"; echo "EXIT=$?"`
Expected: `EXIT=0` (piping drops the summary line; the echo confirms the exit code).

- [ ] **Step 4: mypy on the new pure module**

Run: `py -3.13 -m mypy app/eval/candidate_sweep.py`
Expected: no NEW errors on this file's own lines (the repo carries a pre-existing
baseline; judge by errors attributable to the touched file).

- [ ] **Step 5: Final summary (no commit)**

Confirm the branch contains: pure sweep logic + tests, instrumented script + helper
tests, committed fixture, runbook Pareto section + revised budget, and (if the data
warranted) the default/budget adjustment. Spec acceptance §2 items 1–5 are met.

---

## Self-Review (filled by plan author)

- **Spec coverage:** §2.1 (committed fixture with keys+scores+relevant, sig-matched) → Task 3.
  §2.2 (pure offline reconstruction, unit-tested) → Task 1. §2.3 (runbook Pareto table,
  quality+latency, knee) → Tasks 4–5. §2.4 (revised p95 budget; default changed only if
  plateau<20) → Tasks 5–6. §2.5 (ruff/black/mypy/pytest green) → Task 7. ✓
- **Placeholder scan:** the only bracketed values are in the runbook template (Task 5),
  and each is explicitly a real number transcribed from the Task 3/4 measured tables —
  the measurement is a plan step, not a shipped placeholder. All code blocks are complete. ✓
- **Type consistency:** `base_topk(keys, k)`, `rerank_topk(keys, scores, k)`,
  `sweep_quality(items, candidate_ks, metric_ks)` signatures identical across module,
  unit tests, and the script's `sweep_quality`/`print_quality` call sites. Item dict keys
  (`relevant`, `shortlist_keys`, `teacher_scores`, plus script-only `question`/`_texts`)
  consistent between `capture_items` (writer), `_fixture_items` (projector), and the
  pure logic (reader). `score_item`/`aggregate` used exactly as in `pravo_gate.py`. ✓
