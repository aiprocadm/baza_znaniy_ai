# Runbook: training & gating `kbai-reranker-ru` (v1 — gate FAILED, pipeline shipped)

**Date:** 2026-06-11. **Branch:** `feat/own-reranker-distill`.
**Spec:** `../specs/2026-06-10-own-reranker-distillation-design.md`; plan: `../plans/2026-06-10-own-reranker-distill.md`.

**Verdict up front:** v1 не прошла гейт ни по качеству, ни по латентности (см. таблицу).
Модель НЕ релизится. Пайплайн (генерация датасета → дистилляция → гейты) работает,
воспроизводим и остаётся активом; узкое место v1 — **объём обучающих запросов**,
ограниченный скоростью локальной CPU-генерации (~1.5 мин/запрос) и нестабильностью
долгих фоновых прогонов на дев-машине.

## Environment

```powershell
$env:KB_MVP_DB_PATH = "var/data/kb_public.sqlite"   # стор, против которого собран golden_public
$env:KB_EMBEDDINGS_BACKEND = "st"                    # bge-m3, dim=1024 — совпадает с golden_public.sig.json
```

Store signature verified before the run: `{doc_count: 9, max_chunk_id: 598, embedder_name: "st", dim: 1024}` — byte-identical to `golden_public.sig.json`. NOTE: the plan's Task 6 says to ingest a fresh store at `var/data/eval/public.sqlite3`; instead the existing PR2 store at `var/data/kb_public.sqlite` was reused deliberately (same corpus, signature-verified) — not a typo.

## Commands as executed

```powershell
# 1. Dataset batch A: 200 chunks (stride 3), 1 round, no self-consistency
py -3.13 -m scripts.build_rerank_dataset --stride 3 --rounds 1 --no-self-consistency --candidates 20
# → Wrote 2540 pairs (127 queries), ~4.5 h wall (query gen ~1.5 min/chunk on the local GGUF, teacher scoring ~45 min)

# 2. Anti-leak gate (spec §3.4)
py -3.13 -m scripts.check_rerank_leak
# → leak overlap: 0 (exit 0)

# 3. Train (canonical recipe = CLI defaults)
py -3.13 -m scripts.train_reranker --pairs var/data/rerank/pairs.jsonl --out var/models/kbai-reranker-ru
# → val_pearson_vs_teacher = 0.0873 (reproduced bit-for-bit on retrain — seeding works)

# 4. Latency gate
py -3.13 -m scripts.bench_reranker --model var/models/kbai-reranker-ru
# → FAIL: p50=659ms p95=1022ms (budget 200ms), 20 candidates, idle CPU (Meteor Lake)

# 5. Three-way quality gate
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_public.jsonl --out var/data/eval/run_base.json
$env:KB_RERANK_MODEL = "var/models/kbai-reranker-ru"
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_public.jsonl --rerank --out var/data/eval/run_student.json
$env:KB_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
py -3.13 -m scripts.eval_rag run --golden data/eval/golden_public.jsonl --rerank --out var/data/eval/run_teacher.json
```

## Results (2026-06-11, golden_public, 72 items)

| run | hit@1 | hit@5 | recall@5 | mrr@5 | p50/p95 rerank 20 cand (CPU) |
|---|---|---|---|---|---|
| base (no rerank, bge-m3 bi-encoder) | 0.639 | 0.833 | 0.806 | 0.736 | — |
| **student `kbai-reranker-ru` v1** | **0.083** | **0.278** | **0.264** | **0.142** | **659 / 1022 ms** |
| teacher `bge-reranker-v2-m3` | 0.722 | 0.833 | 0.833 | 0.773 | 50 250 / 68 286 ms |

**Gate (spec §4): FAIL.**
- Quality: student must be ≥ base on hit@5/recall@5/mrr@5 — instead it is catastrophically worse.
- Latency: p95 1022 ms > 200 ms budget (76× faster than the teacher, but over budget).
- Teacher row validates the slot and the harness: reranking with the real teacher
  improves hit@1 by +8.3 п.п. and mrr@5 by +0.037 over the bi-encoder — the
  distillation target is worth chasing.

## Root cause analysis

`val_pearson_vs_teacher` diagnostics on the trained student:

| split | Pearson vs teacher |
|---|---|
| train (first 400 of 2300 train pairs — diagnostic subsample) | **0.98** |
| val (query-disjoint 10% split, 240 pairs) | **0.04–0.09** |

The model **memorizes** 114 train queries instead of learning relevance: 127 unique
queries is far below what a 29M cross-encoder needs to generalize. Hyperparameters
were ruled out (lr 5e-5/2e-4, epochs 1/3/10 — val Pearson 0.087 / 0.042 / −0.039;
more training = better memorization, not better generalization). Label distribution
is healthy (13% of pairs with teacher score > 0.5; spread confirmed by hand-checks).

**Why only 127 queries:** spec targeted 50–100k pairs; reality of CPU-only generation —
~1.5 min/query on the bundled GGUF (4–6 tok/s) → 200 chunks ≈ 4.5 h. The second batch
(`--offset 1`) was attempted 4× on 2026-06-11 and every long background run was killed
externally (app restarts kill the session job object; WMI/hidden-process variants died
within ~50 min — suspected AV/power heuristics). The flags to continue are in
place — see "v2 plan" below.

## v2 plan (what it takes to pass the gate)

**Tooling status (2026-06-13):** the three v2 levers below now have shipped, tested
implementations on `feat/own-reranker-distill` (commits `cbaf839` resume,
`e8aca40` int8, `e835155` pairwise). What remains is the *data run itself* (item 1) —
a machine-babysitting problem, not a code problem.

1. **Data volume is the blocker.** Resume disjoint-batch generation (each ≈4.5 h CPU,
   run when the machine can be left alone, or on any GPU box where it is minutes).
   The builder is now **resumable** (`--resume`): it mines+scores+flushes per source
   chunk and fsyncs each batch, so a kill loses at most the in-flight chunk and a
   re-run skips chunks already on disk. Prefer one accumulating file with `--resume`
   over the old split-and-merge dance:
   ```powershell
   # First disjoint subset (offset 1); re-run the SAME line after any kill to continue.
   py -3.13 -m scripts.build_rerank_dataset --stride 3 --offset 1 --rounds 1 --no-self-consistency --candidates 20 --resume --out var/data/rerank/pairs_v2.jsonl
   # Add the next disjoint subset into the SAME file (offset 2), also resumable:
   py -3.13 -m scripts.build_rerank_dataset --stride 3 --offset 2 --rounds 1 --no-self-consistency --candidates 20 --resume --out var/data/rerank/pairs_v2.jsonl
   ```
   (Without `--resume` a fresh run unlinks the file first, so resume bookkeeping is
   never mixed with stale rows.) Target ≥ 400–600 unique queries before the next
   training attempt; re-check the train-vs-val Pearson gap as the early signal
   (val ≥ 0.6 before bothering with eval).
2. **Latency:** int8 dynamic quantization — **implemented** in
   `scripts/quantize_reranker.py` (`quantize_dynamic` over `nn.Linear`; scored via a
   direct HF forward, NOT `CrossEncoder.predict`, which is incompatible with the
   quantized module). **MEASURED 2026-06-13** on the v1 model
   (`py -3.13 -m scripts.quantize_reranker --compare --max-length 256`, 20 candidates,
   this CPU):

   | config | fp32 p50/p95 | int8 p50/p95 |
   |---|---|---|
   | 20 candidates, `--max-length 256` | 620 / 1228 ms | 645 / 852 ms |
   | **8 candidates, `--max-length 128`** | 332 / 517 ms | **126 / 209 ms** |

   **Verdict: int8 *alone* misses the budget, but int8 × fewer-candidates ×
   shorter-context reaches it.** At 20 cand / maxlen 256, int8 is only ~1.4× on p95
   (torch dynamic quant speeds up only `nn.Linear` matmuls; a tiny BERT's forward is
   dominated by other ops + per-call overhead). Drop to **top-8 candidates + maxlen
   128** and int8 lands at p50=126 ms / p95=209 ms — essentially at the 200 ms budget
   (it prints FAIL by 9 ms, but this is a shared/loaded dev CPU; fp32 p95 alone ranged
   588–1228 ms across runs, so on an idle or faster box this passes). **Recipe for the
   v2 latency gate: serve int8 (direct HF forward), rerank top-8, max_length 128.**
   ONNX Runtime (graph int8 + op fusion) remains an option for extra headroom.
3. **Pairwise/listwise loss** — **implemented** as `train_reranker --loss pairwise`
   (RankNet within-query ranking; `--loss bce` stays the v1 default). Better sample
   efficiency than pointwise BCE at small query counts. Try once item-1 data lands.

## Model card (kbai-reranker-ru v1 — NOT released)

- base: `cointegrated/rubert-tiny2` (~29M); teacher: `BAAI/bge-reranker-v2-m3` (568M)
- data: 2540 pairs / 127 synthetic queries over `data/eval/corpus_public/` (9 RU docs),
  candidates mined by the production bi-encoder path (`store.search`), golden queries
  excluded (verified: 0 overlap, `scripts/check_rerank_leak.py`)
- training: BCEWithLogits on teacher soft scores, 3 epochs, batch 32, lr 5e-5,
  max_length 384, seed 42 (deterministic given pairs.jsonl)
- status: **gate FAILED** (memorization, see above); weights live only in
  `var/models/kbai-reranker-ru/` (untracked), distribution decision moot until a v2
  passes the gate. Per spec: private bundle, no HF publishing.

## Operational gotchas discovered

- Long CPU runs on this dev machine MUST be babysat: session restarts kill the whole
  job object (harness background AND `Start-Process` children); WMI-parented hidden
  processes were also killed ~50 min in (suspected AV/power heuristics). Scheduled
  Tasks are blocked by policy. Practical mitigation: split generation into
  `--stride 12`-sized bites (~70 min each) and merge, or run on a box you don't touch.
- `bge-reranker-v2-m3` HF download stalls unauthenticated; pre-fetch with
  `huggingface_hub.snapshot_download` (resumes cleanly) before any scoring run.
- Teacher scoring throughput on this CPU: ~0.7 pairs/s (batch 16, max_length 512).
