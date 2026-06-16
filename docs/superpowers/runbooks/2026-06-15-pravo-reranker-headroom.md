# Runbook: pravo reranker headroom probe (Phase 0)

**Date:** 2026-06-16. **Spec:** ../specs/2026-06-15-pravo-reranker-headroom-design.md.
**Plan:** ../plans/2026-06-15-pravo-reranker-headroom-phase0.md.

**Store:** `var/data/pravo_public.sqlite` — 6141 docs / 14231 chunks, embedder
**e5-small** (`KB_EMBEDDINGS_BACKEND=st`, `ST_EMBED_MODEL=intfloat/multilingual-e5-small`,
`VECTOR_E5_PREFIX=1`; sig `st`/dim 384).
**Golden:** `data/eval/golden_pravo.jsonl` — 77 structural held-out queries
(article heading topic → query, article chunks → relevant; stride 80 over 6141).

## Env (all four commands)

```powershell
$env:KB_MVP_DB_PATH = "var/data/pravo_public.sqlite"
$env:KB_EMBEDDINGS_BACKEND = "st"
$env:ST_EMBED_MODEL = "intfloat/multilingual-e5-small"
$env:VECTOR_E5_PREFIX = "1"
```

## Commands as executed

```powershell
# ingest (one article = one document; resumable; run detached/background)
py -3.13 -m scripts.ingest_pravo --resume          # -> 6141 docs / 14231 chunks
# golden (structural held-out)
py -3.13 -m scripts.build_pravo_golden             # -> 77 items + sig
# base (bi-encoder, no rerank)
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_pravo.jsonl --out var/data/eval/pravo_base.json
# teacher (cross-encoder rerank)
$env:KB_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_pravo.jsonl --rerank --out var/data/eval/pravo_teacher.json
```

## Results (2026-06-16, golden_pravo, 77 items)

| run | hit@1 | hit@3 | hit@5 | recall@5 | mrr@5 | hit@10 |
|---|---|---|---|---|---|---|
| base (e5-small bi-encoder) | 0.532 | 0.623 | 0.649 | 0.474 | 0.580 | 0.662 |
| teacher (bge-reranker-v2-m3) | 0.597 | 0.662 | 0.688 | 0.513 | 0.634 | 0.701 |
| **Δ (teacher − base)** | **+0.065** | **+0.039** | **+0.039** | **+0.039** | **+0.054** | **+0.039** |

## Gate (spec §4): Δhit@5 ≥ +0.10 → GO / else NO-GO

**Δhit@5 = +0.039 (+3.9 pp) → below the +10 pp threshold → NO-GO (provisional).**

## Verdict: NO-GO (provisional) — but materially different from the prior corpus

Two things changed for the better vs the 9-doc corpus (runbook 2026-06-10):

1. **The base is no longer near-ceiling.** Base hit@5 = 0.649 here vs 0.833 on the
   9-doc corpus. The large, confusable legal corpus genuinely confuses the
   bi-encoder — the "no headroom" failure condition did **not** reproduce.
2. **The teacher helps consistently on every metric** (+3.9…+6.5 pp), unlike the
   9-doc corpus where it added **+0** to hit@5. hit@1 +6.5 pp and mrr@5 +5.4 pp
   are real reordering gains.

But the measured headroom (+3.9 pp hit@5) is still **below** the strict +10 pp
gate the spec set. Per spec §4/§5 this is **provisional**, not a final close,
because two factors plausibly **understate** the true headroom:

- **Structural-golden bias (spec §5, risk #1).** The query is the article heading
  topic, whose words recur in the article body — so the e5 bi-encoder matches
  heading→body unusually easily, inflating base and shrinking the measured delta.
  Natural user questions (which don't echo the heading) would give the reranker
  more to fix.
- **10k-chunk search cap.** `kb_store.search` scans at most 10000 chunks; the
  corpus has 14231, so ~30% of chunks (the higher-index tail) are unreachable by
  **both** retrievers. Golden items whose gold chunk is in that tail are
  unwinnable for base and teacher alike — the teacher cannot rerank what the
  shortlist never contained. This caps the achievable delta. (Input for Phase 1:
  a corpus this size needs Qdrant, not the brute-force in-memory MVP store.)

## Decision

**Do NOT close the reranker yet.** Before a final verdict, run the **manual
control lens** (plan Task 8): ~20 hand-written *natural* legal questions with
known gold articles, re-run base vs teacher on them.

- If the manual lens also shows Δhit@5 < +10 pp → close `kbai-reranker-ru`
  (final, "verified on 6141 articles"); the learned reranker is not worth it for
  this product on this corpus.
- If it flips to ≥ +10 pp (likely, given the structural-golden bias) → proceed to
  **Phase 1** (structural miner, label source A), and additionally fix the
  retrieval cap (Qdrant) so the gold is reachable.

The pipeline (resumable ingest, structural golden, two-way harness) is reproducible
and banked regardless of the decision.
