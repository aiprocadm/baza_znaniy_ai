# Hybrid eval corpus (public CI-guard + private fidelity) + measurable gates — Design

**Date:** 2026-06-06
**Author scope:** technical design for making the RAG answer-quality eval *trustworthy and reproducible*, then running the queued measurement gates on a stable foundation.
**Status:** Design document. Subordinate to [`2026-05-22-project-vision-design.md`](2026-05-22-project-vision-design.md). Direct continuation of [`2026-06-03-rag-answer-quality-eval-design.md`](2026-06-03-rag-answer-quality-eval-design.md) (the harness) and [`2026-06-06-local-keyless-eval-stack-design.md`](2026-06-06-local-keyless-eval-stack-design.md) (the in-process model stack). Executes the queued gates in [`docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md`](../runbooks/2026-06-05-eval-baseline-and-gates.md).

> **TL;DR (ru).** Eval-харнесс уже написан, но стоит на «игрушечном» корпусе из 1 документа / 48 чанков, который живёт только локально (gitignored `var/data/kb_mvp.sqlite`) и не воспроизводится из чистого clone. Чиним фундамент: **гибридный корпус** — публичная синтетическая половина коммитится и гейтит CI на каждом PR, приватная (ваши реальные документы) остаётся локальной и даёт доверенные абсолютные числа. Затем на стабильном фундаменте прогоняем уже существующие гейты B (e5) и D (top_k) и снимаем первый judge-baseline. Каждый PR ≤400 LoC, TDD.

---

## 1. Context and problem

The answer-quality eval harness (`app/eval/*`, `scripts/eval_rag.py`, `scripts/eval_sweep.py`) is built and tested. Its measurement foundation has three defects that make every downstream gate decision untrustworthy:

1. **Toy corpus.** The eval corpus is **1 real RU services contract, 48 chunks**. The runbook flags twice that metrics on it are noisy and that more representative documents are needed before trusting deltas.
2. **Not reproducible.** The corpus lives only in the operator's gitignored `./var/data/kb_mvp.sqlite` ([`kb_store.py:998`](../../../app/services/kb_store.py) `_default_db_path`). Only the *golden* (questions + chunk-ids + reference answers) is committed ([`scripts/build_curated_golden.py`](../../../scripts/build_curated_golden.py)). A fresh clone cannot reproduce the chunk-ids the golden references, so CI cannot run the eval and the "committed regression-guard" promised in the harness spec (§5) does not actually guard anything off the author's machine.
3. **No real committed baseline.** `data/eval/golden_curated.sig.json` currently pins to the **hashing** embedder (`embedder_name="hash", dim 256`) — the very configuration the harness guard rejects as a valid baseline.

There is also a **latent fragility**: the golden binds to the global autoincrement `kb_chunks.id`. Adding documents renumbers ids, silently invalidating every label. Any "grow the corpus" change must first fix the labeling scheme.

**The fix is foundational, not a new feature.** It respects the binding anti-roadmap (`ROADMAP.md`): no new product surface, no API merge, no deferred feature. It makes the existing quality programme trustworthy.

## 2. Goals and non-goals

**Goals.**
- A **reproducible** committed eval corpus so the regression-guard works from a clean clone and in CI.
- A **stable chunk-identity scheme** that survives re-ingestion and corpus growth.
- A **CI gate** that runs deterministic retrieval metrics on every relevant PR **without** downloading multi-GB models.
- A **private** path for trustworthy absolute numbers and generation/judge scoring on real documents, kept out of git.
- Run the queued gates **B** (e5) and **D** (top_k) and produce the first judge-scored generation baseline on the stable foundation.

**Non-goals (explicitly out of scope).**
- **Not** changing eval metrics, the judge prompt, or `RAG_SYSTEM_PROMPT` (drift-tested).
- **Not** building hybrid sparse+dense search, MMR, query expansion, or a runtime faithfulness guard (deferred per the harness spec §2 and `ROADMAP.md`).
- **Not** solving reranker CPU latency (a separate direction; gate C already decided — model swapped, `KB_RERANK_ENABLED` stays default-false).
- **Not** unifying the `/api/kb/*` and `/api/v1/*` surfaces (binding anti-pattern).
- **Not** vendoring model weights into git (we commit the *choice* and a small *derived* artifact, not the bytes).

## 3. What already exists (reuse, do not rebuild)

- **Harness:** `app/eval/{metrics,dataset,adapter,retrieval_eval,judge,generation_eval,report,guards}.py` — metrics, golden schema, the `(document_id, chunk_index) → global_id` resolution in [`adapter._build_id_map`](../../../app/eval/adapter.py), the corpus-signature pin, the hashing/ drift guards.
- **CLIs:** `scripts/eval_rag.py` (`generate`/`run`/`compare`, `--judge`, `--rerank`), `scripts/eval_sweep.py` (top_k sweep), `scripts/build_curated_golden.py`.
- **Local keyless model stack** (`2026-06-06-local-keyless-eval-stack`): in-process `st` embedder (bge-m3) + GGUF eval provider (Qwen2.5-3B) — lets `generate`/`run --judge` run offline, keyless, no daemon. This is what makes the private-half baseline runnable at all.
- **Synthetic generation:** `app/services/synthetic_qa.py` (`SyntheticQAGenerator`, `iter_chunks`, cost guard, `is_refusal`) — reused to author the public corpus's Q&A breadth set.
- **Fixture hygiene:** `tests/conftest.py::_protect_committed_fixtures` — extend to cover new committed fixtures.

## 4. Architecture — two halves of one harness

```
               ┌─ public half (committed) ─────────────────────┐
 eval_rag run ─┤  synthetic RU docs (text fixtures)             │→ CI guard,
   (one code   │  + golden_public.jsonl + frozen embeddings     │   reproducible
    path)       └─ private half (local, gitignored) ────────────┘   from clone
                  real documents (KB_EVAL_CORPUS=private)        → trustworthy
                                                                    absolute numbers
                                                                    + judge/generation
```

One golden format, one metric set, one CLI. The halves differ **only** in (a) which corpus is loaded and (b) what is committed. Selection via `KB_EVAL_CORPUS=public|private` (default `public`).

## 5. Stable chunk identity (the key refactor — PR1)

**Problem.** Golden binds to the global autoincrement `kb_chunks.id`; adding/reordering documents renumbers ids → labels silently drift.

**Decision.** Identify a chunk by the composite key **`"<doc_key>:<chunk_index>"`** where:
- `doc_key` = `kb_documents.filename` (the public-corpus builder guarantees unique, stable filenames; private corpora document the same requirement).
- `chunk_index` = the existing 0-based within-document index (`kb_chunks.chunk_index`, already in the schema).

The adapter already builds `(document_id, chunk_index) → global_id`; extend `_build_id_map` to key on `(filename, chunk_index)` via a join with `kb_documents`. The golden stores the composite string keys; the adapter resolves them to whatever global ids the local ingest produced. **No schema migration** (filename already exists).

**Golden schema change.** `GoldenItem.relevant_chunk_ids: tuple[int, ...]` → `relevant_chunks: tuple[str, ...]` (composite keys). JSONL `meta` carries `relevant_chunks`; a back-compat loader still accepts legacy integer `relevant_chunk_ids`/`source_chunk_id` lines (resolved against a provided id-map or kept as-is for old private goldens). The hit@k membership test compares resolved global ids exactly as today.

**Alternatives considered.**
- *Global autoincrement id* — the current fragility; rejected.
- *`sha256(chunk_text)`* — robust to reordering but **breaks when chunk size/overlap is tuned**, and chunk-size is a thing we may tune; rejected.
- *New `doc_key`/content-hash column* — a migration for no gain over `filename`; rejected (YAGNI).

**Boundary.** `chunk_index` is stable only while chunk size/overlap (900/140) is unchanged. The queued gates B/D do **not** re-chunk, so labels are stable across all planned work. If chunk size is ever tuned, labels are rebuilt from the corpus — documented exactly like a reindex is.

## 6. Public half — synthetic corpus (PR2)

- **Generator:** `scripts/build_public_corpus.py` uses the local GGUF (or any configured teacher-LLM) to author **~8–15 realistic RU documents** spanning the dog-food document types: services contract, NDA, internal regulation (регламент), procedure (процедура), and an NPA-style normative text. Target **~300–500 chunks** — an order of magnitude more stable than 48.
- **Determinism.** LLM generation is a **one-time authoring step**. Git stores its *output* — plain-text fixtures under `data/eval/corpus_public/*.md` — **not** the generator's randomness. CI never runs an LLM. Re-running the generator is for *extending* the corpus, a deliberate reviewed change.
- **Review gate.** Before commit, generated docs and the auto-golden are reviewed for triviality and answer-leakage (same discipline as the curated golden's domain-owner review).
- **Golden:** `data/eval/golden_public.jsonl` = auto breadth (`SyntheticQAGenerator` over the fixtures) + curated hard cases (multi-hop, paraphrase, ≥3 `expect_refusal`). Carries a corpus signature sidecar.

## 7. Private half (PR4)

- Real documents stay in the operator's local `var/data/...` (gitignored, as today). Nothing private enters git: `golden_private.jsonl` and its corpus are local-only.
- Selection: `KB_EVAL_CORPUS=private` points the harness at the private store/golden.
- Tests touching the private half are `@pytest.mark.integration` and **skip loudly** (with a reason) when the corpus is absent — never a silent pass.

## 8. CI gate without model downloads (PR3)

**Problem.** Retrieval metrics in CI require embedding the *query* → loading bge-m3 (~2 GB). Unacceptable per-CI-run.

**Decision — frozen embeddings.** Once, locally, on the pinned bge-m3, precompute the public corpus's **passage vectors** and the public golden's **query vectors**; commit them as a compact `.npz` (`data/eval/corpus_public/frozen_bge-m3.npz`, ~1–2 MB). A new `make_frozen_retriever(npz)` in the adapter does pure cosine ranking with **no model loaded**. CI is pure numpy — deterministic, seconds.

- **What it gates:** retrieval metrics `hit@k` / `recall@k` / `mrr@k` on the **answerable** items of `golden_public`, against a committed threshold; a regression fails the build. These need no LLM — pure numpy over the frozen vectors. `expect_refusal` items (empty relevant set) are excluded from the retrieval aggregate.
- **What runs only locally / integration (not in the no-model CI gate):** `refusal_correctness` and all generation/judge scores require the in-process LLM to *produce* an answer (the scoring step is then deterministic, but generation is not free), so they are report-only on the private half and the integration smoke test. This **narrows** the harness spec §9 ("retrieval + refusal_correctness") to retrieval-only for CI — a deliberate consequence of the keyless/no-download CI constraint.
- **What it does NOT gate:** the embedder weights themselves (covered by one `@pytest.mark.integration` test that loads real bge-m3) and LLM-judge scores (nondeterministic).
- **CI wiring:** a new job in `.github/workflows/ci.yml`, path-scoped to `app/eval/**`, `app/retriever/**`, `app/services/kb_store.py`, `data/eval/**`, mirroring the existing path-classifier pattern.

**Alternative considered.** Download a small embedder (e5-small, ~120 MB) and embed live in CI — simpler conceptually, but (1) adds a flaky/slow per-run download and (2) would gate *e5-small* retrieval rather than the production *bge-m3* config. Frozen embeddings are faster, deterministic, and guard the production embedder's vectors. Rejected.

**Why this is sound (design note).** Frozen embeddings *isolate the layer under test*: the CI guard verifies exactly the logic PRs change (ranking, reranking, filters, golden-label correctness) and is independent of multi-GB weights and network. The weights are verified separately by one integration test — the same "one gate, one responsibility" discipline that split gate C into a quality gate and a latency gate.

## 9. Gate execution (private half, after the foundation — PR4)

The harness already supports every step:
- **Baseline:** `eval_rag run --judge` on bge-m3 + GGUF (local keyless stack) → keep `baseline.json`.
- **Gate B (e5 on v1):** reindex under an e5 model with `VECTOR_E5_PREFIX=true` (no-op for bge-m3) → `run` → `compare`. Gate: `recall@k` / `mrr@k` ↑.
- **Gate D (top_k):** `eval_sweep --golden <private> --values 5,8,10,12 --judge` → argmax `completeness` without dropping `faithfulness`; set MVP `ask` `top_k` (and/or v1 `RETRIEVE_TOPK`).
- Each gate: keep iff its target metric improves, else revert; paste the `compare` delta table into the PR (the gate-C discipline).

## 10. Error handling and guardrails

- **Corpus-absent:** loud `@pytest.mark.integration` skip with a reason; never a silent pass.
- **Signature drift:** the existing `run` signature-mismatch refusal is extended to the `doc_key` scheme — stale labels fail loudly.
- **Fixture hygiene:** the new committed fixtures (`corpus_public/`, `golden_public.jsonl` + sidecar, `frozen_*.npz`) are added to `tests/conftest.py::_protect_committed_fixtures` so a stray test write fails loudly instead of dirtying tracked bytes.
- **Hashing guard:** retained; the frozen public path is never on the hashing embedder.

## 11. Testing strategy (TDD)

- **Unit:** dependency injection everywhere — fake retriever/provider/model; no real weights. Order: `metrics` → `dataset` (new key scheme + back-compat) → `adapter` (`_build_id_map` join + `make_frozen_retriever`) → `report`.
- **Integration (`@pytest.mark.integration`, not `skip`):** one real-bge-m3 smoke test over the public corpus, verifying the frozen `.npz` matches a live encode within tolerance (so a stale `.npz` is caught).
- **Stubs:** the frozen path loads no heavy library; `tests/stubs/` need not change. If an integration-stub path later needs them, fix `DummySentenceTransformer.encode` separately (out of scope).
- **No new drift surface:** `RAG_SYSTEM_PROMPT` stays byte-identical (existing drift test).

## 12. PR staging (each ≤400 LoC)

| PR | Scope | Deliverable |
|---|---|---|
| PR1 | Stable chunk identity: `GoldenItem` key scheme + `adapter._build_id_map` join + migrate the current 21 curated items + back-compat loader | Pure refactor; metrics unchanged; the single contract still scores identically |
| PR2 | Synthetic public corpus: `build_public_corpus.py` + `corpus_public/*.md` fixtures + `golden_public.jsonl` (auto + curated) + frozen `.npz` | Committed, reproducible public corpus + golden |
| PR3 | CI eval-gate: `make_frozen_retriever` + threshold file + path-scoped workflow job | Every relevant PR gated on deterministic public metrics |
| PR4 | Private half: `KB_EVAL_CORPUS` switch + integration markers + runbook update; run gates B/D + judge-baseline | Trustworthy absolute numbers + recorded gate results |

## 13. Open decision (resolve at spec review)

The current 21 curated questions describe a real contract. In the hybrid, that contract logically moves to the **private** half and the committed guard becomes the synthetic public golden. The contract's facts are already in git history, so this does not "un-leak" them — but keeping the guard on synthetic data going forward is cleaner. **Decision needed:** retire the contract Q&A from the committed set (move to private) vs. keep it committed as an extra example. Recommended: retire to private.

## 14. Risks and mitigations

- **Synthetic realism.** Synthetic docs may miss real scan/layout messiness. Mitigation: that fidelity is the *private* half's job; the public half only needs to be a stable, representative *guard*.
- **Auto-golden quality.** Synthetic Q&A can be trivial or leak answers. Mitigation: self-consistency filter + the curated subset + the pre-commit review gate.
- **Frozen `.npz` staleness.** If the corpus text changes but the `.npz` is not rebuilt, CI would guard stale vectors. Mitigation: the integration smoke test re-encodes and compares to the committed `.npz` within tolerance; the signature sidecar pins the embedder/dim.
- **Local stack required for PR2/PR4.** Authoring the corpus and running the baseline need the in-process model stack installed (`pip install` of the already-declared heavy deps). PR1 and the unit tests need none of it (DI + lazy imports).
- **CPU latency.** bge-m3 + 3B GGUF on CPU are slow per call but trivial over a few-hundred-chunk corpus; flagged for anyone who grows the corpus much further.
