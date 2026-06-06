# Eval Baseline + Queued Gates — Runbook

> Run this the moment a real embedder + LLM exist. Until then, the curated
> golden (`data/eval/golden_curated.jsonl`) and the tightened MVP prompt have
> already shipped; everything here is gated on a trustworthy baseline.
> Status is judged from git, not checkboxes.

## 0. Stand up a real embedder + LLM (pick one)

**Ollama (local, free; ~1.3 GB download, CPU inference on this machine):**

```powershell
winget install --id Ollama.Ollama -e
ollama pull bge-m3 ; ollama pull qwen2.5:3b
$env:KB_EMBEDDINGS_BACKEND="ollama"; $env:OLLAMA_EMBED_MODEL="bge-m3"
$env:KB_LLM_PROVIDER="ollama"; $env:OLLAMA_MODEL="qwen2.5:3b"
```

**OpenAI-compatible API (fast; needs BOTH an embeddings endpoint AND an LLM key):**

```powershell
$env:KB_EMBEDDINGS_BACKEND="api"; $env:EMBEDDINGS_API_BASE_URL="https://<host>/v1"
$env:EMBEDDINGS_API_KEY="<key>"; $env:EMBEDDINGS_API_MODEL="<embed-model>"
$env:DEEPSEEK_API_KEY="<key>"   # judge/generation — a chat key is NOT an embeddings key
```

## 1. Reindex under the real embedder, refresh the curated sidecar

```powershell
py -3 -m scripts.kb_cli reindex --embedder ollama --force-yes   # or: openai-compatible
py -3 -m scripts.build_curated_golden    # rewrites .sig.json; chunk-ids unchanged
```

The reindex name must equal `embedder.name` (`ollama` or `openai-compatible`),
else it fails loudly. The builder re-emits `golden_curated.sig.json` so the
signature matches the new `embedder_name`/`dim` (the chunk-id labels are
reindex-stable and do not change).

## 2. Baseline (the HARD gate)

```powershell
py -3 -m scripts.eval_rag generate --out var/data/eval/golden_auto.jsonl --limit 200
py -3 -m scripts.eval_rag run --golden data/eval/golden_curated.jsonl `
  --out var/data/eval/baseline.json --judge
```

Keep `baseline.json` — every gate compares against it. The corpus is small
(1 doc / 48 chunks), so metrics are noisy; consider ingesting more
representative documents for a stabler baseline before trusting deltas.

## 3. Queued gates (keep iff the metric improves, else revert)

Each: apply the change, re-run `run ... --out var/data/eval/<task>.json`, then
`py -3 -m scripts.eval_rag compare var/data/eval/baseline.json var/data/eval/<task>.json`.
Paste the delta table into the PR.

- **C — Russian reranker.** `DEFAULT_MODEL_NAME` in `app/services/kb_rerank.py`
  and `app/retriever/rerank.py` → `BAAI/bge-reranker-v2-m3`; enable
  (`KB_RERANK_ENABLED=true`). Gate: `mrr@k` / `hit@5` ↑ **and** latency
  acceptable. (~600 MB model download; this is the win that genuinely needs a
  number before shipping.)
  - **RESULT (2026-06-06 — in-process bge-m3 + bge-reranker-v2-m3, curated golden
    n=21): PASS.** hit@1 0.619→0.714 (+0.095), recall@1 0.548→0.643, mrr@5
    0.659→0.738 (+0.079), recall@5 0.714→0.762 (+0.048); every k improves.
    `DEFAULT_MODEL_NAME` swapped to `bge-reranker-v2-m3` (both files) + `.env.example`.
    `KB_RERANK_ENABLED` kept **opt-in** (default false) — flip once latency is
    confirmed acceptable for the target deployment. NB: measured via a
    reranking-aware eval retriever — the stock `make_mvp_retriever` calls raw
    `store.search`, while rerank lives in `kb_mvp.ask` (search → `rerank_hits`); a
    `--rerank` flag on `eval_rag run` would make this repeatable.
- **D — top_k.** `py -3 -m scripts.eval_sweep --golden data/eval/golden_curated.jsonl --values 5,8,10,12 --judge`.
  Pick argmax `completeness` without dropping `faithfulness`; set the MVP `ask`
  `top_k` (and/or v1 `RETRIEVE_TOPK`). Commit with the table.
- **B — e5 on v1.** Reindex the v1 path under an e5 model with
  `VECTOR_E5_PREFIX=true` (no-op for MVP / bge-m3). Gate: `recall@k` / `mrr@k` ↑.

## Sidecar refresh note

There is no CLI to rewrite *only* the sidecar; `scripts/build_curated_golden.py`
re-emits it (chunk-ids are reindex-stable — only `embedder_name`/`dim` change).

## Tracked-file hygiene (what dirties git — and what doesn't)

Three committed fixtures are written **only by the manual ops scripts above**,
never by `pytest`:

- `data/eval/golden_curated.jsonl` + `data/eval/golden_curated.sig.json` ←
  `scripts/build_curated_golden.py` (step 1). With a real embedder the `.sig.json`
  flips to e.g. `embedder_name="st"`/`dim 1024`; the `.jsonl` is re-emitted
  byte-identically, so usually only the sidecar shows in `git diff`.
- `models/model_manifest.json` ← `scripts/download_model.py --target <model>`
  (the one-time GGUF fetch: nulls `model_id`/`filename`, adds `sha256`, `sort_keys`).

So after running either script, `git diff` and commit (or `git checkout --`) the
change deliberately. If you ever see those files dirty after *just* a test run,
it isn't the suite: `tests/conftest.py::_protect_committed_fixtures` (session
autouse) snapshots them and **fails loudly while restoring the bytes** if any
test writes one. Run the suite with `KB_TEST_LOCK_FIXTURES=1` to instead lock
them read-only and get a `PermissionError` traceback naming the offending test.
