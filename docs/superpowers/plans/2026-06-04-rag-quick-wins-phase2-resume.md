# RAG Quick-Wins (Phase 2) — Resume / Handoff

> Companion to [`2026-06-03-rag-quick-wins-phase2.md`](2026-06-03-rag-quick-wins-phase2.md).
> Captures exactly what shipped, what's blocked, and the turnkey sequence to finish.
> **Status is judged from git, not checkboxes** (per repo convention).

## TL;DR

The **gate-free / offline code is done and committed** (4 commits, 35 tests green,
ruff+black clean). The **measurement gates and the two behavior-changing tasks
(C, E) are blocked on a real baseline**, which needs a real embedder + LLM. On
this machine that means either an ~8 h Ollama download (link is ~44 KB/s) or
API credentials. Setup was **paused** by user choice.

## Shipped (branch `rag-eval-harness`)

| Commit | Task | Notes |
|---|---|---|
| `e264da4` | **A / PR4** — `app/eval/guards.py:ensure_real_embedder` + README note | Not metric-gated. Done. |
| `0e405b6` | **Reindex real-embedder support** (unplanned prerequisite gap) | `kb-cli reindex` delegated hash-only; now delegates non-hash names to `kb_embeddings.get_embedder()` with a loud backend-mismatch guard. **Required** for any real reindex. |
| `f70f9fb` | **B / PR5** — `app/retriever/e5.py` + `VECTOR_E5_PREFIX` wired into qdrant/faiss | Flag **off** by default. See "Task B reality" below. |
| `0ce8cbe` | **D / PR7 (steps 1–2)** — `eval_rag run --top-k` + `scripts/eval_sweep.py` | Tooling only; the sweep run + argmax pick (steps 3–4) need the baseline. |

Verify anytime:
```powershell
py -3 -m pytest tests/test_eval_guards.py tests/test_kb_cli_reindex.py `
  tests/test_retriever_e5_prefix.py tests/test_vector_stores.py `
  tests/test_eval_retrieval.py tests/test_eval_cli.py tests/test_eval_sweep.py -q
```

## Key findings (read before resuming)

1. **Reindex was hash-only.** The plan's prerequisite `kb-cli reindex --embedder <name>`
   could not produce a real index until `0e405b6`. Now: `--embedder ollama`
   (or `openai-compatible` for the API backend) re-embeds via the env-configured
   backend. It asserts `embedder.name == <name>` so a misconfigured env fails loudly.
2. **Task B is v1-path, not MVP-measurable.** `eval_rag.py` measures the **MVP
   SQLite store** (`make_mvp_retriever` → `kb_store.search` → `kb_embeddings`).
   The e5 prefix wiring lives in the **v1 stores** (`qdrant.py`/`faiss.py` →
   SentenceTransformer). The harness does **not** exercise it. Also `bge-m3`
   (the recommended MVP embedder) is **non-e5**, so the prefix is a no-op there.
   → Task B ships as correct, inert infra for e5-based **v1** deployments; its
   gate is only meaningful with an e5 model on the v1 path. Don't expect the MVP
   `compare` to move from `VECTOR_E5_PREFIX`.
3. **Machine / network reality:** 31 GB RAM, Intel Core Ultra 9 185H, **Intel Arc
   iGPU (no CUDA/ROCm → Ollama runs on CPU)**. Link ≈ **44 KB/s**; OllamaSetup.exe
   is **1.29 GB** (~8 h). Nothing cached: no `torch`, no `sentence-transformers`,
   no HF hub cache, no GGUF.
4. **No `.env` autoload.** Production reads `os.environ` directly (only the test
   stub defines `load_dotenv`). Pass env **inline** per command (examples below),
   or set user/system env vars.

## Resume — Step 0: stand up a real embedder + LLM

Pick ONE path. **bge-m3** = embeddings (1024-d, strong RU); **qwen2.5:3b** =
chat/judge (fast on CPU).

### Path A — Ollama (local, free; needs the ~1.29 GB download)
```powershell
winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
# (or download OllamaSetup.exe; a ~41 MB partial may remain in %TEMP%)
ollama serve            # if the tray service isn't already on :11434
ollama pull bge-m3
ollama pull qwen2.5:3b
$env:KB_EMBEDDINGS_BACKEND="ollama"; $env:OLLAMA_EMBED_MODEL="bge-m3"
$env:KB_LLM_PROVIDER="ollama"; $env:OLLAMA_MODEL="qwen2.5:3b"
# reindex name must equal embedder.name == "ollama"
py -3 -m scripts.kb_cli reindex --embedder ollama --force-yes
```

### Path B — OpenAI-compatible API (fast; needs an embeddings endpoint AND an LLM key)
```powershell
$env:KB_EMBEDDINGS_BACKEND="api"; $env:EMBEDDINGS_API_BASE_URL="https://<host>/v1"
$env:EMBEDDINGS_API_KEY="<key>"; $env:EMBEDDINGS_API_MODEL="<embed-model>"
$env:DEEPSEEK_API_KEY="<key>"   # or any provider in kb_llm.KNOWN_PRESETS
# OpenAICompatibleEmbedder.name == "openai-compatible"
py -3 -m scripts.kb_cli reindex --embedder openai-compatible --force-yes
```
> A chat-only key (DeepSeek/Groq/etc.) covers `generate`/`--judge` but **not**
> embeddings — embeddings need their own API or Ollama.

## Resume — Step 1: baseline (the HARD GATE)
With env set + corpus reindexed under the real embedder:
```powershell
py -3 -m scripts.eval_rag generate --out var/data/eval/golden_auto.jsonl --limit 200
py -3 -m scripts.eval_rag run --golden var/data/eval/golden_auto.jsonl `
  --out var/data/eval/baseline.json --judge
```
Corpus is small (**1 doc / 48 chunks** — "Договор аутсорсинг Русконструкт"), so
`--limit 200` yields ≤48 golden items; metrics will be noisy. Consider ingesting
more docs for a stabler baseline. Keep `baseline.json` — every gate compares to it.

## Resume — Step 2: run the gates (keep iff metric improves, else revert)

Each: re-run `run ... --out var/data/eval/<task>.json` after the change, then
`py -3 -m scripts.eval_rag compare var/data/eval/baseline.json var/data/eval/<task>.json`.

- **B / e5** — only if you reindex the **v1** path under an **e5** model with
  `VECTOR_E5_PREFIX=true` (no-op for MVP/bge-m3 — see finding #2). Gate: recall@k / mrr@k ↑.
- **D / top_k (steps 3–4, tooling ready):**
  ```powershell
  py -3 -m scripts.eval_sweep --golden var/data/eval/golden_auto.jsonl --values 5,8,10,12 --judge
  ```
  Pick argmax `completeness` **without** dropping `faithfulness`; set it as the
  MVP `top_k` default (`app/api/kb_mvp.py` `ask`) and/or v1 `RETRIEVE_TOPK`. Commit with the table.

## Resume — Tasks NOT yet implemented (held until measurable)

These change **live behavior**; per the plan ("nothing ships blind") they were
deliberately left for after the baseline.

- **C / PR6 — Russian reranker default.** `app/services/kb_rerank.py` `_DEFAULT_MODEL`
  → `BAAI/bge-reranker-v2-m3`; enable on both surfaces (`KB_RERANK_ENABLED=true`,
  v1 `RERANK_ENABLED=true`); update `tests/test_reranking.py` default + `.env.example`.
  Gate: mrr@k / hit@5 ↑ **and** latency acceptable. ⚠️ The cross-encoder needs
  `sentence-transformers` + a ~600 MB model — another download on this link, and
  confirm whether the MVP eval path actually invokes the reranker before trusting the gate.
- **E / PR8 — prompt tightening.** `app/api/kb_mvp.py:_RAG_SYSTEM_PROMPT` (+ v1
  `chat_orchestrator.py`): add per-claim `[N]` citation + explicit "say I don't
  know when context is insufficient" (keep Russian). Re-pin
  `app/eval/generation_eval.py:RAG_SYSTEM_PROMPT` so
  `test_eval_generation.py::test_system_prompt_matches_production` stays green.
  Gate: faithfulness ↑ + refusal_correct ↑ without relevance regressing.

## One-line status
Code that needs no real model: **done & green**. Everything gated on measurement:
**ready to run the moment a real embedder + LLM exist** — follow Step 0 → Step 2 above.
