# RAG Answer-Quality Evaluation Harness + Measured Quick-Wins — Design

**Date:** 2026-06-03
**Author scope:** technical design for measuring and then improving RAG answer quality across the shared retriever layer.
**Status:** Design document. Subordinate to `2026-05-22-project-vision-design.md`. Sibling to `2026-05-29-retrieval-degradation-visibility-design.md` (this spec *scores* quality; that one makes *degradations* loud).
**Decision context:** Today nothing measures answer quality. The repo tracks infrastructure degradations (Qdrant down, hashing embedder, dim mismatch) but never scores whether an answer is relevant, grounded, or complete. Any retrieval/prompt tuning is therefore a guess. This spec builds a measurement foundation first, then applies cheap, high-confidence improvements — each gated by the metric it claims to move.

> Этот документ описывает **что и как сделать**, без привязки к срокам. Реализация стадируется по PR (см. §10).

---

## 1. Context and problem

KB.AI answers questions over a customer corpus through **two parallel retrieval paths** (`docs/architecture.md`) that share an embedder/reranker layer but are configured differently. The quality-relevant defaults today:

| Concern | MVP `/api/kb/*` | v1 `/api/v1/*` | Source |
|---|---|---|---|
| Embedder (default, no `.env`) | **hashing**, dim 256 (near-random) | `intfloat/multilingual-e5-small`, dim 384 | `kb_embeddings.py:169` (`_build_from_env`); `.env.example:45,86,87` |
| e5 query/passage prefixes | n/a | **absent** — raw text encoded | `qdrant.py:273`, `faiss.py:229` (`_batched_encode([query])`) |
| Reranker enabled | `false` | `true` | `.env.example:65,90` |
| Reranker model | `ms-marco-MiniLM-L-6-v2` (**English**) | same | `.env.example:68` |
| `top_k` default | 4–5 | 10 | `kb_mvp.py:921` (`ask`), `kb_store.py:514`; `.env.example:89` |
| LLM gen params | temp 0.2, max_tokens 1024 | same | `kb_llm.py:27-29` |
| Grounding check | none — prompt instructs "use only context" but answer is never verified | none | `kb_mvp.py:406` (`_RAG_SYSTEM_PROMPT`) |

**Three latent quality problems fall out of the table** and motivate the quick-wins (§8):

1. **Hashing cliff.** With no `.env`, the MVP path embeds with the hashing fallback → near-random semantic search while the LLM still answers confidently. The same query gets decent results on v1 and garbage on MVP.
2. **e5 without prefixes.** `multilingual-e5` is trained to receive `"query: "` / `"passage: "` prefixes; omitting them measurably lowers recall.
3. **English reranker on a Russian corpus.** The default cross-encoder is English; `.env.example:67` itself recommends `BAAI/bge-reranker-v2-m3` for RU/multilingual, yet ships the English one and disables it on MVP entirely.

**The gap is measurement.** None of the above can be confirmed or safely changed without an answer-quality eval. This spec builds that eval, establishes a baseline, then fixes the three problems above (plus `top_k`/prompt tuning) as **measured** changes.

## 2. Goals and non-goals

**Goals.**
- A reusable eval harness over the **shared retriever layer** that scores both retrieval and generation quality, runnable against **both** HTTP surfaces from one entrypoint, with no server/auth required for the retrieval loop.
- A committed, regression-guarding **golden set** (auto-generated for breadth + hand-curated for hard cases), pinned to a corpus snapshot.
- A reproducible **baseline report** that exposes the MVP-hashing vs v1-e5 gap.
- A sequence of **gated quick-wins**, each shipped only if the harness proves it improves the metric it targets.

**Non-goals (explicitly deferred — Approach C).**
- **Not** building hybrid search (BM25+vector fusion), MMR diversity, query expansion, or an answer-time faithfulness guard. These are a data-driven follow-on once the baseline shows where the ceiling is.
- **Not** unifying the two HTTP surfaces (anti-pattern per `docs/architecture.md`).
- **Not** retraining/altering LoRA adapters or DPO. We *reuse* the synthetic-QA toolkit those workstreams produced; we do not modify their training paths.
- **Not** adding heavy deps to the light MVP install. The eval lives in its own package and CLI; nothing it adds is imported by request-handling code.

## 3. What already exists (do not rebuild)

The W1/W3 ML-strengthening workstreams left an I/O-free toolkit whose `QAPair` (question + known source chunk) is exactly a retrieval-eval gold label.

- **Synthetic Q&A generation** — `app/services/synthetic_qa.py`:
  - `SyntheticQAGenerator` (`:408`): prompt → teacher LLM → `parse_qa_response` → refusal/length filter → optional self-consistency.
  - `QAPair` (`:16`) with `to_jsonl_line`/`from_jsonl_line` — the on-disk schema we extend.
  - `iter_chunks(store)` (`:497`): yields `(chunk_id, text)` from `kb_chunks`.
  - `estimate_total_cost_usd` (`:371`) + per-provider pricing — the cost guard we reuse.
  - `is_refusal` (`:98`): RU/EN refusal detector — reused for deterministic refusal scoring.
  - `load_processed_chunk_ids` (`:464`): resume support.
- **RAG-dataset assembly** — `app/services/rag_dataset.py`:
  - `Retriever = Callable[[str, int], Sequence[hit]]` (`:143`) — the injection seam the eval calls.
  - `build_relevant_sample` (`:160`): retrieves and checks `seed.source_chunk_id in retrieved_chunk_ids` — this membership test **is hit@k**.
- **Degradation contract** — `app/observability/retrieval_health.py`: the eval report echoes its reasons (e.g. hashing-embedder active) rather than re-detecting.
- **LLM transport** — `app/services/kb_llm.py`: `select_provider()` + the `LLMProvider` protocol (`synthetic_qa.py:389`) — the judge reuses this.
- **Retriever entrypoints** — `kb_store.KnowledgeBaseStore.search` (`:514`, returns `SearchHit` with `.chunk_index`/`.text`) and `services/vectorstore.search` (`:200`).

## 4. Architecture and layout

A new, focused package plus one CLI. The core is I/O-free (matching `synthetic_qa`/`rag_dataset` style) so it is deterministic in tests.

```
app/eval/
  __init__.py
  dataset.py          # GoldenItem + JSONL load/save; back-compat with QAPair JSONL
  metrics.py          # pure: hit_at_k, recall_at_k, mrr_at_k, aggregation
  adapter.py          # one (query, top_k) -> hits adapter over MVP + v1 retrievers
  retrieval_eval.py   # run Retriever over golden items -> retrieval metrics
  judge.py            # LLM-judge prompts + robust verdict parsing
  generation_eval.py  # end-to-end answer -> faithfulness/relevance/completeness/citation + refusal
  report.py           # JSON + Markdown report; compare(run_a, run_b) diff
scripts/
  eval_rag.py         # CLI: generate | run | compare   (mirrors scripts/eval_lora.py, kb_cli.py)
data/eval/
  golden_curated.jsonl   # committed: hand-written hard cases + corpus signature header
var/data/eval/           # gitignored: generated golden, run reports, judge/retrieval caches
tests/
  test_eval_metrics.py, test_eval_retrieval.py, test_eval_judge.py,
  test_eval_dataset.py, test_eval_report.py     # TDD, fake retriever/provider (cf. test_rag_dataset.py)
```

**Reuse map:** golden generation = `SyntheticQAGenerator` + `iter_chunks`; schema = `QAPair`(+meta); retrieval call = `rag_dataset.Retriever`; judge transport = `kb_llm` + `LLMProvider`; cost guard = `estimate_total_cost_usd`; refusal = `is_refusal`.

**Surface coverage.** `adapter.py` exposes `make_retriever(surface)` returning a `Retriever`:
- `surface="mvp"` → wraps `kb_store.search` (SQLite cosine; no server).
- `surface="v1"` → wraps `vectorstore.search` (Qdrant/FAISS).
One golden set, run against both, surfaces the hashing-vs-e5 gap in a single report.

## 5. Golden-set design

**Schema — `GoldenItem`** (superset of `QAPair`, serialized to JSONL; loader accepts plain `QAPair` lines too):

```json
{"instruction": "<question>", "input": "", "output": "<reference answer or \"\">",
 "meta": {"source_chunk_id": 7, "relevant_chunk_ids": [7, 12],
          "expect_refusal": false, "source": "auto|curated"}}
```

- `relevant_chunk_ids` is the retrieval ground truth (defaults to `[source_chunk_id]` for legacy `QAPair` lines).
- `output` is the reference answer for completeness scoring (may be empty for auto items where we only score retrieval).
- `expect_refusal=true` marks unanswerable questions — the system *should* decline; scored deterministically.

**Auto-generated set (breadth).** `eval_rag.py generate` runs `SyntheticQAGenerator` over `iter_chunks` in `SINGLE` mode (clean single-chunk ground truth → clean hit@k) plus a smaller `MULTI_HOP` portion (multi-chunk relevance). Target ~100–300 items, bounded by `estimate_total_cost_usd` with a `--budget-usd` guard and `--yes` confirmation above threshold. Resume via `load_processed_chunk_ids`.

**Curated set (precision).** ~20–30 hand-written items in `data/eval/golden_curated.jsonl`, committed. Must include: multi-chunk questions, paraphrase/synonym questions, and several `expect_refusal=true` unanswerable questions. This file is the **regression guard** — small, stable, reviewed.

**Corpus pinning.** Ground-truth `chunk_id`s are only valid against a fixed index. The golden file header records a **corpus signature** (`{doc_count, max_chunk_id, embedder_name, dim}`). `run` recomputes it and **refuses with a clear error** on mismatch — preventing stale-label silent failure (same ethos as the degradation-visibility spec).

## 6. Metrics

**Retrieval (deterministic, pure functions in `metrics.py`).** For k ∈ {1, 3, 5, 10}, computed from `relevant_chunk_ids ∩ retrieved_ids`:
- `hit@k` — any relevant chunk in top-k.
- `recall@k` — fraction of relevant chunks retrieved.
- `mrr@k` — reciprocal rank of the first relevant chunk.
Aggregated as mean across items, reported overall and per surface.

**Generation (LLM-judge + deterministic).** For each item, run the real answer path, then score:
- `faithfulness` — every claim supported by retrieved context (the anti-hallucination metric).
- `relevance` — does it answer the question.
- `completeness` — coverage vs `output` reference (only when present).
- `citation_correctness` — do `[N]` markers map to chunks that actually support the claim (heuristic + judge).
- `refusal_correctness` — **deterministic**: for `expect_refusal` items, did the answer decline (`is_refusal` + `IRRELEVANT_REFUSAL` marker)?

Judge runs at **temperature 0**; the report records judge `provider`/`model`. Scores are 1–5, normalized to [0,1] in aggregation.

## 7. Runner, caching, report

- `retrieval_eval` and `generation_eval` consume the `adapter` retriever and `kb_llm` judge.
- **Caching** (`var/data/eval/cache/`): keyed by `(surface, query, top_k)` for retrieval and by a hash of the judge prompt for verdicts, so reruns and `compare` are cheap and fair. No `random`/timestamps in keys.
- **Report** (`report.py`): a machine-readable JSON run + a human Markdown summary with per-surface metric tables and a **degradations** block echoing `retrieval_health` reasons (e.g. flags loudly if the run executed on a hashing embedder — a baseline on hashing is not a valid baseline).
- `compare(run_a, run_b)` emits a metric delta table — the artifact pasted into each quick-win PR as evidence.

## 8. Quick-wins sequence (each a gated experiment)

Ordered by ROI and safety. Each lands as its own ~≤400 LoC PR with a `compare` report; kept only if it moves its target metric without unacceptable latency cost.

1. **Embedder reality guard (step 0).** `run` refuses to emit a "baseline" on the hashing embedder (loud error + remediation text). Document configuring a real MVP embedder (`KB_EMBEDDINGS_BACKEND=ollama|api`). *No reindex if MVP was never truly indexed; switching a populated index requires `kb-cli reindex`.* Gate: prerequisite, not metric-gated.
2. **e5 `query:`/`passage:` prefixes.** Add prefix handling in the shared embed path (`qdrant.py`/`faiss.py` `_batched_encode` + v1 ingest), behind a `VECTOR_E5_PREFIX` flag. **Requires reindex** (passages embedded with `passage:`; queries with `query:` must match). Gate: keep iff `recall@k`/`mrr@k` improve on a reindexed snapshot.
3. **Russian reranker on both surfaces.** Default `KB_RERANK_MODEL`/v1 reranker → `BAAI/bge-reranker-v2-m3`; align `KB_RERANK_ENABLED`/`RERANK_ENABLED`. Query-time only — *no reindex*. Gate: keep iff `mrr@k`/`hit@5` improve and added latency is acceptable.
4. **`top_k` / context-budget tuning.** Sweep `top_k ∈ {5,8,10,12}` and the v1 3000-token context budget; pick by completeness↑ without faithfulness↓. *No reindex.*
5. **Prompt tightening.** Sharpen grounding + citation discipline in `_RAG_SYSTEM_PROMPT` and the v1 orchestrator prompt. Gate via `faithfulness` + `refusal_correctness`. *No reindex.*

## 9. Error handling, guardrails, testing

- **Hashing guard** (§8.1) and **corpus-drift guard** (§5) are hard, loud failures — never silent.
- **Offline/CI.** `metrics.py` + `retrieval_eval` tests use fake retrievers (the `_FakeHit` pattern from `tests/test_rag_dataset.py`); `judge` tests use a fake `LLMProvider`. No network in unit tests. Reuse/extend `tests/stubs/` for `sentence_transformers` etc.
- **Cost control.** `generate`/`run` print a forecast via `estimate_total_cost_usd` and require `--yes` above a configurable USD threshold; unknown (provider, model) disables the guard with a warning (existing behavior).
- **CI gating.** CI runs only the **deterministic** metrics (retrieval + `refusal_correctness`) as a pass/fail gate; LLM-judge generation scores are reported and regression-flagged within a tolerance band, never hard-failing the build (judge nondeterminism).
- **TDD order:** `metrics` → `dataset` → `retrieval_eval` → `judge` → `generation_eval` → `report` → CLI. Run with `py -3 -m pytest` (Windows: `py -3` launcher, no venv).

## 10. PR staging

| PR | Scope | Deliverable |
|---|---|---|
| PR1 | `metrics.py` + `dataset.py` + `adapter.py` + `retrieval_eval.py` + `eval_rag.py run` (retrieval only) | Retrieval baseline on both surfaces; exposes hashing-vs-e5 gap |
| PR2 | `judge.py` + `generation_eval.py` + `report.py` (+ `compare`) | Full baseline report (retrieval + generation) |
| PR3 | `eval_rag.py generate` wiring + `data/eval/golden_curated.jsonl` + corpus signature | Committed golden set (auto + curated), pinned |
| PR4 | Embedder reality guard (§8.1) | `run` refuses hashing baselines |
| PR5 | e5 prefixes (§8.2) | Reindexed snapshot + recall delta report |
| PR6 | RU reranker (§8.3) | mrr/hit@5 delta report |
| PR7 | `top_k`/context tuning (§8.4) | completeness/faithfulness trade-off report |
| PR8 | Prompt tightening (§8.5) | faithfulness/refusal delta report |

PR1–PR3 build the harness (Approach A's content); PR4–PR8 are the measured quick-wins (the rest of Approach B). PR5–PR8 each gate on their `compare` report.

## 11. Acceptance criteria

- `eval_rag.py run --surface mvp|v1` produces a JSON+Markdown report with `hit@k`, `recall@k`, `mrr@k` (k∈{1,3,5,10}) and, with a judge configured, generation scores.
- The report flags loudly when run on a hashing embedder or a drifted corpus.
- `golden_curated.jsonl` is committed, includes ≥3 `expect_refusal` items, and carries a corpus signature.
- `compare` outputs a metric delta table between two runs.
- Each of PR5–PR8 is merged only with a `compare` report showing its target metric improved.
- Unit tests pass offline (no network) via fakes/stubs; CI gates deterministic metrics.

## 12. Risks and open questions

- **LLM-judge cost & noise.** Mitigated by caching, temp 0, deterministic CI gate, and judge-model recording. Open: whether a cheap judge (deepseek-chat) is discriminating enough — validate against the curated set in PR2.
- **Reindex coupling (PR5).** e5-prefix correctness depends on re-embedding passages. Open: run on a dedicated eval corpus snapshot to avoid disturbing any live index.
- **Auto-golden quality.** Synthetic seeds can be trivial or leak the answer. Mitigated by the self-consistency filter and the curated subset; open: whether to add a difficulty filter.
- **Ground-truth granularity.** `source_chunk_id` assumes the answer lives in one chunk; chunk boundaries (900/140) may split it. Mitigated by `relevant_chunk_ids` (multi) on curated items.
