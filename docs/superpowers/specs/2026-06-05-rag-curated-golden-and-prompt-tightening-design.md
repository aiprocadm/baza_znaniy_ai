# Infra-Free RAG Quality Wins — Curated Golden + Prompt Tightening — Design

**Date:** 2026-06-05
**Author scope:** technical design for improving RAG answer quality *without* standing up a real embedder/LLM on this machine.
**Status:** Design document. Subordinate to `2026-05-22-project-vision-design.md`. Direct continuation of `2026-06-03-rag-answer-quality-eval-design.md` (the eval harness) and its Phase-2 handoff `docs/superpowers/plans/2026-06-04-rag-quick-wins-phase2-resume.md`.
**Decision context:** Phase-2 quick-wins are blocked on a real baseline, which needs a real embedder + LLM (an ~8 h Ollama download on a ~44 KB/s link, or API credentials). The user chose to stay **infra-free for now**. This spec scopes the subset of quality work that is defensible and verifiable **without** a judge-based eval: a real curated golden set (so the eval becomes trustworthy the moment models exist) plus the one prompt change whose prior is overwhelming and whose consistency is unit-testable. Everything that genuinely needs measurement stays queued behind a turnkey runbook.

> Этот документ описывает **что и как сделать** в infra-free режиме. Сроки не фиксируются.

---

## 1. Context and problem

"Measurably better RAG answers" currently has **three** blockers, not one (verified 2026-06-05):

| Blocker | State | Evidence |
|---|---|---|
| Real embedder/LLM | absent | Ollama not installed/running; no API keys in env |
| Eval corpus | **1 doc / 48 chunks** (one outsourcing contract «Русконструкт») | `var/data/kb_mvp.sqlite`: `kb_documents=1`, `kb_chunks=48`; v1 `kb.sqlite` empty |
| Curated golden | **stub** | `data/eval/golden_curated.jsonl`: 1 answerable item with literal `"ЗАМЕНИ…"` + `relevant_chunk_ids:[1]` (unaligned), 3 generic refusals; **no `.sig.json` sidecar** |

The corpus is, if anything, the deeper blocker than the models: on 48 chunks with a stub golden, any judge metric is noise and any keep/revert gate decision is a coin flip. So the highest-leverage infra-free move is to make the **measurement substrate** real, and to land only the quality change that does not need a judge to justify.

**Chosen approach (Approach C of three considered):** ship a real curated golden (PR-G) + prompt tightening (PR-E); queue the measurement-gated wins (RU reranker, top_k sweep, e5) behind a runbook. Rejected: A ("substrate only" — no quality change lands now) and B ("ship E + reranker now" — the reranker ships blind against its own latency gate and adds ~600 MB to the light MVP install).

## 2. Goals and non-goals

**Goals.**
- Replace the stub `golden_curated.jsonl` with a real, reviewed, corpus-pinned regression guard (~18–22 hand-labelled items) so a credible baseline is one command away once models exist.
- Sharpen the MVP system prompt for per-claim `[N]` citations and a **canonical** refusal phrase, with unit-tested consistency and zero added install weight or latency.
- Consolidate the (currently stale, scattered) "stand up models → baseline → gates" steps into one committed turnkey runbook.

**Non-goals (explicitly queued or out of scope).**
- **Not** shipping the RU reranker default (Task C), top_k pick (Task D steps 3–4), or e5-on-v1 (Task B) — these need real measurement; they stay queued (§7).
- **Not** touching the v1 `chat_orchestrator` prompt now — separate symmetric sibling PR; v1 corpus is empty, nothing to measure (two-path design preserved).
- **Not** standing up infra, downloading models, or adding heavy deps to the MVP install.
- **Not** unifying the two HTTP surfaces; **not** altering LoRA/DPO paths.
- **Not** expanding the corpus with new source documents (a useful later step, but retrieval metrics over new docs need a real embedder).

## 3. What already exists (do not rebuild)

- **Golden schema + I/O** — `app/eval/dataset.py`: `GoldenItem(question, relevant_chunk_ids, reference_answer, expect_refusal, source)`, `load_golden`/`save_golden`, QAPair-superset JSONL.
- **Corpus signature** — `app/eval/dataset.py:60` (`CorpusSignature{doc_count,max_chunk_id,embedder_name,dim}`), `write_signature`/`read_signature` (`.sig.json` sidecar); `app/eval/adapter.py:62` `compute_signature(store)` (returns `embedder_name="hash"` today).
- **Signature guard** — `scripts/eval_rag.py`: `run` WARNS on missing sidecar (`:50`), ERRORS on mismatch (`:56`); `generate` writes the sidecar (`:125`).
- **Hashing guard** — `app/eval/guards.py:9` `ensure_real_embedder` refuses to emit a baseline on `embedder_name=="hash"`.
- **Deterministic refusal scoring** — `app/eval/generation_eval.py:63` scores `refusal_correct` with **no judge** (`looks_like_refusal`, `:30`); canonical substring `_CANONICAL_REFUSAL="не удалось найти"` (`:27`); `is_refusal` (`app/services/synthetic_qa.py:98`).
- **Canonical refusal string** — `IRRELEVANT_REFUSAL = "Не удалось найти в документах информацию для ответа."` (`app/services/rag_dataset.py:214`).
- **Prompt drift test** — `test_system_prompt_matches_production` (`tests/test_eval_generation.py:31`) pins `generation_eval.RAG_SYSTEM_PROMPT == kb_mvp._RAG_SYSTEM_PROMPT`.
- **Queued-gate tooling** — `scripts/eval_rag.py` (`generate|run|compare`, `--top-k`, `--judge`), `scripts/eval_sweep.py` (top_k sweep).

## 4. Reindex-stability of chunk-id labels (the load-bearing fact)

`reindex` (`scripts/cli/reindex.py`) re-embeds chunks **in place** (UPDATE by `doc_id`+`chunk_idx`); it does not renumber them. Therefore `relevant_chunk_ids` authored now — by **reading chunk text**, which is embedder-independent — remain valid after a future real reindex. Only the signature's `embedder_name`/`dim` flip (`hash/256` → e.g. `ollama/1024`). Consequence: we author labels now, and refresh only the `.sig.json` sidecar after the first real reindex (one command; §5.3).

## 5. Deliverable 1 — Real curated golden set (PR-G)

### 5.1 Authoring
Read all 48 chunks of `kb_mvp.sqlite` and hand-label **~18–22 `GoldenItem`s**:
- **~13–17 answerable** (incl. the multi-hop and paraphrase items below), each with real `relevant_chunk_ids` and a draft `reference_answer` derived from the chunk text. Include **2–3 multi-hop** (answer spans several chunks → multi-id `relevant_chunk_ids`) and **2–3 paraphrase/synonym** items (retrieval-robustness).
- **~5 `expect_refusal=true`**: keep the 3 existing generic out-of-corpus refusals, add **2 "plausible-but-out-of-corpus"** items (sound like the contract but have no answer in it — the highest-value probe for catching confident hallucination in a legal corpus).

### 5.2 Reference-answer ownership
The draft `reference_answer`s are ground truth *derived from reading chunks* — legitimate curator work, but in a legal domain the user is the final arbiter. The implementation plan includes an explicit **user-review step** of the ~18 reference answers before commit. (Per-item escape hatch: leave `reference_answer=""` → that item is scored retrieval-only.)

### 5.3 Signature handling
Write `data/eval/golden_curated.sig.json` via `write_signature`, recording the current corpus (`doc_count=1, max_chunk_id=48, embedder_name="hash", dim=256`). This already guards against corpus **content** drift (doc/chunk counts). Document, in the eval README / runbook, that after the first real reindex the sidecar is refreshed — chunk-ids unchanged, only embedder/dim flip — so `run`'s mismatch guard passes. There is no existing CLI to rewrite *only* the sidecar (`eval_rag generate` would regenerate the items too), so the refresh is a documented one-liner `write_signature(p, compute_signature(store))`, with a small `eval_rag sig` subcommand as an optional convenience in PR-R.

### 5.4 Files
- Modify: `data/eval/golden_curated.jsonl`.
- Add: `data/eval/golden_curated.sig.json`.
- **No production code touched.** Reuse `GoldenItem`/`save_golden`/`write_signature`.

## 6. Deliverable 2 — Prompt tightening (PR-E)

### 6.1 Sites (changed together)
`app/api/kb_mvp.py:406` (`_RAG_SYSTEM_PROMPT`) **and** `app/eval/generation_eval.py:20` (`RAG_SYSTEM_PROMPT`), bound by `test_system_prompt_matches_production` — both edited in one commit so the drift test stays green.

### 6.2 Two changes
1. **Per-claim, mandatory citation.** Replace «…ссылайся на фрагменты в формате [1], [2] **там, где они уместны**» with an instruction that *every* claim carries a `[N]` pointing at the supporting fragment. Lever: faithfulness / citation_correctness.
2. **Canonical refusal.** Instruct the model to answer **exactly** `Не удалось найти в документах информацию для ответа.` when context is insufficient. This string equals `IRRELEVANT_REFUSAL` and is caught by `looks_like_refusal` ("не удалось найти"), so the deterministic `refusal_correct` metric becomes meaningful later **without a judge**, and production refusals become uniform.

To avoid a second copy of the string and avoid pulling heavy imports into the light `kb_mvp`, the literal is **inlined** in the prompt and a new unit test pins it equal to `IRRELEVANT_REFUSAL` (idiom mirrors the existing drift test).

### 6.3 Scope boundaries
- **MVP prompt only.** The customer-facing surface per vision. v1 `chat_orchestrator` prompt is a queued symmetric sibling, not this PR.
- **No-hits path untouched.** `kb_mvp.py:437` returns a distinct hardcoded message when there are zero hits; that is an orthogonal path the refusal probes (which *do* retrieve irrelevant hits) never exercise.

### 6.4 Honest boundary
E's *metric* gains (refusal_correct↑, faithfulness↑) can only be measured once an LLM exists; this PR ships on **overwhelming prior + reversibility + unit-tested consistency**, not a `compare` report. This is a conscious, scoped relaxation of the spec's "merged only with a compare report" rule, justified because (a) the change is a reversible string, (b) it adds no install weight or latency, and (c) its two behaviours are deterministically checkable later. The reranker (Task C), whose tradeoff is genuinely measurable, is **not** granted the same relaxation.

## 7. Deliverable 3 — Turnkey runbook (PR-R)

Add `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md`: a single, de-staled, parameterized "models exist → one paste" sequence:
1. `reindex` under the real embedder → regenerate `golden_curated.sig.json`;
2. `generate` auto-golden + `run --judge` → `baseline.json`;
3. run the queue — **C** (RU reranker), **D** (`eval_sweep` top_k), **B** (e5 on v1) — each with `compare` vs baseline;
4. rule: keep iff the target metric improves (and, for C, latency acceptable), else revert.

No new maintained code (existing `eval_rag.py`/`eval_sweep.py` suffice). An optional thin `scripts/eval_baseline.ps1` orchestrator is a stretch goal only if literal one-command is wanted.

## 8. Testing, error handling, guardrails

**Tests (all infra-free, no network):**
- PR-G: new `tests/test_golden_curated.py` — `load_golden` parses N items; ≥3 `expect_refusal`; every answerable item has non-empty `relevant_chunk_ids` ⊆ `[1, max_chunk_id]`; `read_signature` ≠ None. This is the anti-re-stub guard.
- PR-E: `test_system_prompt_matches_production` stays green (both constants edited together) + new test "prescribed refusal == `IRRELEVANT_REFUSAL`" + test "prompt contains the per-claim citation instruction".

**Error handling (reuse existing, loud-by-design):** signature mismatch already raises a clear error (`eval_rag.py:56`); the hashing guard already refuses hash baselines (`guards.py:9`). The committed `hash/256` sidecar means `run` will loudly require a sidecar refresh after the first real reindex — documented, never silent.

**Verification:** `py -3 -m pytest tests/test_golden_curated.py tests/test_eval_generation.py tests/test_eval_dataset.py -q` → green, offline (Windows `py -3` launcher, no venv).

## 9. PR staging

| PR | Scope | Deliverable | Est. LoC |
|---|---|---|---|
| PR-G | curated golden + sidecar + `test_golden_curated.py` | real regression guard, corpus-pinned | ~120 (mostly data) |
| PR-E | MVP prompt tightening + eval pin + 2 tests | per-claim citations + canonical refusal | ~40 |
| PR-R | `eval-baseline-and-gates.md` runbook | turnkey queued-gate sequence | docs only |

Order: PR-G → PR-E (independent; G first so the substrate exists). PR-R rides with either. Each well under the ~400 LoC norm.

## 10. Acceptance criteria

- `data/eval/golden_curated.jsonl` holds ~18–22 reviewed items (no `"ЗАМЕНИ…"` placeholder), ≥3 `expect_refusal` (incl. ≥2 plausible-but-out-of-corpus), with a committed `.sig.json`; `tests/test_golden_curated.py` green.
- The MVP prompt mandates per-claim `[N]` and the canonical `IRRELEVANT_REFUSAL` phrase; `test_system_prompt_matches_production` + the new pin/instruction tests green.
- The runbook reproduces the full baseline→gates sequence from a configured embedder+LLM with no stale commands.
- The targeted pytest subset passes offline.
- No production behaviour beyond the MVP system prompt changes; no new heavy deps; two-path design preserved.

## 11. Risks and open questions

- **Single-doc corpus limits breadth.** ~18–22 curated items over one contract is a precision set, not breadth. Mitigation: that is exactly the curated set's role per the harness spec; auto-golden breadth comes later (runbook step 2). Open: whether to ingest 2–3 more representative docs before the first real reindex (deferred — needs the user's docs + a real embedder to be useful).
- **Author-derived reference answers.** Mitigated by the explicit user-review gate (§5.2) and the `reference_answer=""` escape hatch.
- **Shipping E without a `compare` report.** Mitigated as in §6.4 (reversible, weightless, deterministically checkable later); explicitly *not* extended to the reranker.
- **Sidecar refresh friction after reindex.** One documented command; chunk-ids stable so labels survive.

## 12. Explicitly queued (post-infra) and out of scope

Queued behind the runbook, each gated by its `compare` report once models exist: **C** RU reranker default (`kb_rerank.py:32` + `retriever/rerank.py:18` → `BAAI/bge-reranker-v2-m3`; gate mrr/hit@5↑ **and** latency), **D** top_k pick (`eval_sweep`), **B** e5 on the v1 path. Out of scope entirely: two-path merge, hybrid search, LoRA/DPO changes, corpus expansion with new documents, v1 orchestrator prompt (separate sibling).
