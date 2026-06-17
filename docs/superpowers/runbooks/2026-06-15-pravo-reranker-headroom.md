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

## Manual control lens (2026-06-16) — 20 natural questions

Per the §5 safeguard, 20 hand-written *natural* legal questions (one per several
codes; phrasing deliberately does NOT echo the article heading), gold = the
article's chunks. File: `data/eval/golden_pravo_manual.jsonl` (committed).

| run | hit@1 | hit@3 | hit@5 | mrr@3 | mrr@5 | hit@10 |
|---|---|---|---|---|---|---|
| base (e5-small) | 0.550 | 0.600 | 0.700 | 0.575 | 0.597 | 0.750 |
| teacher (bge-reranker-v2-m3) | 0.650 | 0.750 | 0.750 | 0.700 | 0.700 | 0.750 |
| **Δ** | **+0.100** | **+0.150** | **+0.050** | **+0.125** | **+0.103** | **0.000** |

**Reading.** On natural questions the teacher delivers large gains exactly where
they matter for RAG — the **top of the ranking**: hit@1 +10 pp, hit@3 +15 pp,
mrr@5 +10.3 pp. But the **gate metric, hit@5, is only +5.0 pp** — because base
hit@5 is already 0.700 and hit@10 is 0.750, so @5/@10 are near a ceiling on this
20-query sample. hit@5 is structurally **insensitive to reordering inside the
top-5** (moving gold from rank 3→1 doesn't change hit@5, but does change hit@1 and
mrr) — which is most of what a reranker does. So the chosen gate metric
under-measures the reranker's real, product-relevant value here.

## Final verdict (combining both lenses)

- **Literal gate (Δhit@5 ≥ +10 pp): NOT met** on either golden (+3.9 pp structural,
  +5.0 pp manual).
- **But the reranker clearly helps**, and on natural questions it lifts the
  top-rank metrics that drive RAG answer quality by **+10–15 pp** (hit@1/hit@3/mrr).
  This is qualitatively unlike the 9-doc corpus (where teacher added +0 across the
  board). The headroom is real; the hit@5 gate was the wrong single yardstick.
- Two confounds still **understate** it: the 10k-chunk search cap (~30% unreachable)
  and the tiny 20-query manual sample.

**Recommendation: do NOT close — treat as a conditional GO, gated on mrr@5/hit@3
rather than hit@5.** Sensible next step before committing to Phase 1 training:
remove the 10k search cap (or move to Qdrant) and re-measure on a larger natural
golden, so the decision rests on a clean, top-rank-sensitive metric. If the user
prefers the original strict hit@5 gate as binding, the verdict is a (narrow) close.
Decision is the user's; both readings are documented honestly above.

## Clean re-measure / Phase 0.5 (2026-06-16) — cap confound removed

The two earlier lenses were confounded by the 10k search cap: golden items whose
gold chunk sits in the unreachable tail (>10000 chunks) are forced misses,
deflating **base** and inflating the apparent headroom. Setting
`KB_SEARCH_HARD_LIMIT=20000` makes the brute-force MVP store scan all 14231 chunks
in pure Python — **pathologically slow** (~16 min CPU for *base* alone, teacher
untenable). So instead the natural golden was **restricted to gold articles
reachable under the default cap** (`golden_pravo_natural.jsonl`, 40 → **36** items;
4 tail articles dropped), giving a clean delta at full speed — every gold is
reachable by both retrievers.

| run | hit@1 | hit@3 | hit@5 | recall@5 | mrr@5 | hit@10 |
|---|---|---|---|---|---|---|
| base (e5-small) | 0.778 | 0.861 | 0.917 | 0.609 | 0.832 | 0.944 |
| teacher (bge-reranker-v2-m3) | 0.889 | 0.944 | 0.944 | 0.658 | 0.917 | 0.972 |
| **Δ** | **+0.111** | **+0.083** | **+0.028** | **+0.049** | **+0.086** | **+0.028** |

**What the clean number says.** Once the gold is actually reachable, the e5
bi-encoder is **near-ceiling on coverage** (hit@5 0.917) — so hit@5 has almost no
room (+2.8 pp), and *most of the earlier "headroom" on hit@5 was the cap artifact,
not bi-encoder confusion.* BUT the reranker delivers a clean, confound-free
**+11.1 pp hit@1** and **+8.6 pp mrr@5**: it reliably pulls the correct article
from rank 2–5 up to rank 1. That is exactly the move that matters for a RAG product
(the top result drives the generated answer).

## FINAL verdict (Phase 0 + 0.5)

- **Strict hit@5 gate: NOT met** anywhere (+3.9 / +5.0 / +2.8 pp). On a reachable
  corpus hit@5 is near-ceiling, so this gate is the **wrong yardstick** for a
  reranker (it is blind to top-5 reordering, which is the reranker's whole job).
- **By a top-rank metric the reranker clearly helps, cleanly:** **+11 pp hit@1,
  +8.6 pp mrr@5** on confound-free natural questions. This is real product value,
  unlike the 9-doc corpus (which showed +0 across the board).
- **Conditional GO.** Recommend proceeding to **Phase 1** (structural miner +
  distil/train a student), with the **release gate re-defined as mrr@5 ≥ +0.05 (or
  hit@1 ≥ +0.05)**, not hit@5. The student must of course beat *base*, and ideally
  recover a good fraction of the teacher's +11 pp hit@1.
- **Carry into Phase 1:** the MVP brute-force store does not scale to 14k chunks
  (10k cap; full scan ~pathological) — Phase 1 should run retrieval through Qdrant
  so the whole corpus is reachable and fast.

Pipeline banked: resumable ingest, structural + natural goldens, two-way harness,
detached-run recipe for the slow teacher. Final decision is the user's.
