# Local, keyless, in-process model stack for the RAG eval — Design

> Status: design (approved shape, 2026-06-06). The implementation plan will live
> in a sibling `docs/superpowers/plans/2026-06-06-local-keyless-eval-stack.md`
> (created next, via the writing-plans workflow).

## 1. Problem

The only open work in the eval programme is the three measurement-gated steps
in [`docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md`](../runbooks/2026-06-05-eval-baseline-and-gates.md):

- **B** — e5 on the v1 retrieval path (`recall@k` / `mrr@k`).
- **C** — Russian reranker `BAAI/bge-reranker-v2-m3` (`mrr@k` / `hit@5`).
- **D** — `top_k` sweep (`completeness` / `faithfulness`).

They are blocked **not by code but by infrastructure**: the harness refuses to
produce a baseline on the hashing embedder (near-random — [`guards.ensure_real_embedder`](../../../app/eval/guards.py)),
and today a "real" embedder/LLM means **either** the Ollama daemon (`winget install`,
`ollama pull`, ~GB, a running server) **or** an OpenAI-compatible API key. The
operator wants neither a daemon to babysit nor secrets to re-enter — the model
choices should be **wired into the code** and run offline after a one-time fetch.

## 2. Root cause (why the friction exists)

`sentence-transformers`, `torch`, `llama-cpp-python`, and `huggingface-hub` are
**already core dependencies** ([`requirements.txt`](../../../requirements.txt)).
The reranker already loads its model **in-process** through them — keyless,
one-time download ([`app/retriever/rerank.py`](../../../app/retriever/rerank.py)).
But the **embedder** ([`app/services/kb_embeddings.py`](../../../app/services/kb_embeddings.py))
has only three backends — `ollama` (needs the daemon), `api` (needs a key), and
`hash` (near-random, rejected by the guard) — **no in-process backend at all**.

The **LLM** is split across two modules. The eval calls the MVP layer
([`app/services/kb_llm.py`](../../../app/services/kb_llm.py)), which is OpenAI-HTTP
only (keyless-local = Ollama-the-daemon). But a fully in-process GGUF provider
**already exists** in [`app/llm/llama_cpp_provider.py`](../../../app/llm/llama_cpp_provider.py)
(the v1/general stack), with a matching downloader in
[`scripts/download_model.py`](../../../scripts/download_model.py) and config in
`Settings` (`llm_model_path`, `llm_ctx`, …). It is simply **not wired into the
eval's provider selection**, and its `generate(prompt, *, context) -> str` signature
differs from the eval's `generate(prompt, *, system, …) -> obj.text` contract.

So the blocker is narrower than it first looks: build **one** new in-process
embedder backend, and **adapt** the existing in-process LLM provider into the
eval's interface — no new LLM engine, no new dependency.

## 3. Goals / non-goals

**Goals**

- Run the full runbook (`reindex`, `eval_rag generate/run --judge`, `eval_sweep
  --judge`, `compare`) **fully offline, keyless, with no background daemon**.
- Model identity **pinned in code** (overridable by env for advanced/gate-B use);
  weights fetched **once** into the HuggingFace cache, reused forever.
- **Zero new dependencies.** Wiring only.
- Keep the eval scoring logic and the runbook commands **byte-identical** — only
  env defaults and two new backends change.

**Non-goals**

- No Git-LFS vendoring of weights into the repo (bloat — rejected; we commit the
  *choice*, not the *bytes*).
- No Ollama, no local HTTP server process.
- No GPU requirement (CPU path; the eval corpus is 1 doc / 48 chunks).
- No unification of the `/api/kb/*` and `/api/v1/*` surfaces (binding anti-pattern,
  see [`docs/architecture.md`](../../architecture.md)).
- No change to the eval metrics, judge prompt, or `RAG_SYSTEM_PROMPT` (drift-tested).

## 4. Chosen model tier (operator-selected: "quality ~5 GB")

| Role | Model | Size | Notes |
|---|---|---|---|
| Embedder (default) | `BAAI/bge-m3` | ~2.2 GB | 1024-dim, strong RU, **no** query/passage prefixing. Matches the runbook baseline. |
| Embedder (gate B) | `intfloat/multilingual-e5-base` | ~1.1 GB | e5 prefixing required; selected via env for the B run only. |
| Reranker (gate C) | `BAAI/bge-reranker-v2-m3` | ~600 MB | The gate's named target. |
| LLM gen + judge | `Qwen2.5-3B-Instruct` GGUF `Q4_K_M` | ~2.0 GB | Same family the runbook chose (`qwen2.5:3b`). |

One-time total ≈ 5 GB into `~/.cache/huggingface`. Keyless (public repos), offline
thereafter.

## 5. Architecture — two in-process components + thin wiring

Both components mirror the existing reranker pattern: optional-import guard,
lazy module-level cache, heavy load on first use.

### 5.1 Component A — `SentenceTransformerEmbedder`

New backend in [`app/services/kb_embeddings.py`](../../../app/services/kb_embeddings.py),
selected by `KB_EMBEDDINGS_BACKEND=st`.

- **Interface:** satisfies the existing `Embedder` Protocol — `name`,
  `dimension`, `embed(text) -> list[float]`. Adds a `model: str` attribute so the
  store can apply e5 prefixing.
- **Model:** pinned default `BAAI/bge-m3`, overridable by `ST_EMBED_MODEL`. Lazy
  `SentenceTransformer(model_name)`; `encode(text, normalize_embeddings=True)`
  (BGE/e5 want normalized vectors + cosine, consistent with the other backends'
  `_normalise`). `dimension` read from the loaded model (no network probe needed —
  weights are local after the one-time fetch).
- **`.name = "st"`.** This is what `kb-cli reindex --embedder st` must match
  ([`reindex.py:36`](../../../scripts/cli/reindex.py) asserts `embedder.name == name`).
  The corpus signature becomes `(embedder_name="st", dim=1024)`. This mirrors the
  **existing** looseness for `ollama` (the name does not encode the model — the
  model is pinned separately). Because the model is *pinned in code*, swapping it
  is a deliberate edit that the operator already knows requires a reindex
  (documented in CLAUDE.md). Residual risk: two different same-`dim` models would
  share a signature — accepted, pre-existing.
- **Guard / lazy import:** `import sentence_transformers` happens **inside** the
  lazy loader (not at module top), so importing `kb_embeddings` stays light and a
  clear error is raised only when `st` is selected *and* a real model is needed.
  Unit tests inject a fake model (DI) and never import the real library — note that
  `tests/stubs/` has **no** `sentence_transformers` stub and `install_service_stubs`
  is not global, so a non-DI unit test would pull the real 2.2 GB model. DI is
  therefore mandatory in unit tests.
- **Reindex integration:** none beyond the name match — `_make_embedder("st")`
  already routes any non-`hash` name through `get_embedder()`.

### 5.2 Component B — `GgufEvalProvider` (adapter over the existing engine)

A thin adapter in [`app/services/kb_llm.py`](../../../app/services/kb_llm.py),
selected by `KB_LLM_PROVIDER=gguf`. It does **not** re-implement llama.cpp — it
wraps the existing [`app/llm/llama_cpp_provider.LlamaCppProvider`](../../../app/llm/llama_cpp_provider.py)
and exposes the eval's interface.

- **Interface:** matches the duck-typed `LLMProvider` Protocol the eval depends on
  ([`synthetic_qa.py:389`](../../../app/services/synthetic_qa.py)) — `name="gguf"`,
  `model` (the GGUF filename), `generate(prompt, *, system=None, max_tokens=None,
  temperature=None)` returning the existing `LLMResponse` dataclass.
- **Adaptation:** the inner provider's `generate(prompt, *, context) -> str` takes
  no `system` and returns a bare string. The adapter folds system+prompt into one
  text prompt (`f"{system}\n\n{prompt}"`) and passes
  `context={"temperature": temperature if temperature is not None else 0.0,
  "max_tokens": max_tokens or 512}`, then wraps the returned string as
  `LLMResponse(text=…, provider="gguf", model=…, elapsed_ms=…)`. **Temperature 0**
  by default for stable judge verdicts; `parse_verdict` already tolerates
  fences/prose/out-of-range ([`app/eval/judge.py`](../../../app/eval/judge.py)).
- **Weights / download:** reuse [`scripts/download_model.py`](../../../scripts/download_model.py)
  (HF-hub or HTTP, skip-if-present, manifest-driven). Add a `qwen2.5-3b-instruct`
  entry to [`models/model_manifest.json`](../../../models/model_manifest.json)
  (`model_id="Qwen/Qwen2.5-3B-Instruct-GGUF"`, `filename="qwen2.5-3b-instruct-q4_k_m.gguf"`
  — exact filename confirmed against the repo at implementation). One-time
  `py -3 scripts/download_model.py --target qwen2.5-3b-instruct …`; the adapter
  points `Settings.llm_model_path` at the result via `KB_LLM_GGUF_PATH`
  (default `./models/qwen2.5-3b-instruct-q4_k_m.gguf`). The current manifest
  `default` (TinyLlama-1.1B) is too weak for an RU judge — hence a dedicated entry.
- **Selection wiring:** `select_provider` / `build_provider` in `kb_llm.py` branch
  on `provider == "gguf"` and return `GgufEvalProvider`, bypassing the
  HTTP-specific `KNOWN_PRESETS` path. Default is **explicit** (`KB_LLM_PROVIDER=gguf`);
  optional `KB_LLM_LOCAL_FALLBACK=true` auto-selects it when no key is set. (We do
  not silently load a multi-GB model by default.)
- **Guard:** the inner provider already guards `llama_cpp` import and validates the
  GGUF magic bytes. The adapter lazily constructs the inner provider so importing
  `kb_llm` stays cheap.

### 5.3 Wiring — reranker, env, e5 prefixing

- **Gate C is measurable with no code change:** run with
  `KB_RERANK_ENABLED=true KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3` (both honoured
  today by [`kb_rerank.load_config`](../../../app/services/kb_rerank.py)). The
  default-model swap (`DEFAULT_MODEL_NAME` in `kb_rerank.py` and
  `app/retriever/rerank.py`) is the **commit** that happens *only if the gate
  passes* — not a prerequisite to measure. Tests asserting the old default name
  are updated in that commit only.
- **e5 prefixing (gate B):** the store applies the existing
  [`e5_prefix()`](../../../app/retriever/e5.py) helper at its two call sites —
  `role="query"` in `KnowledgeBaseStore.search` ([`kb_store.py:524`](../../../app/services/kb_store.py)),
  `role="passage"` on the index/reindex path — guarded by `_is_e5(embedder.model)`
  and `VECTOR_E5_PREFIX`. **No-op for the bge-m3 default**, so default users pay
  nothing; it switches on only for the e5 gate-B run.
- **Env defaults:** `.env.example` gains a documented "fully-local, keyless"
  block (`KB_EMBEDDINGS_BACKEND=st`, `KB_LLM_PROVIDER=gguf`, model pins). These can
  ship as the committed default so the eval runs offline out of the box.

## 6. Data flow — one offline `run --judge`

```
eval_rag run --judge
  └─ get_store() → KnowledgeBaseStore(embedder = SentenceTransformerEmbedder["st"])
  └─ guards.ensure_real_embedder(sig)            # name "st" ≠ "hash" → passes
  └─ retriever = make_mvp_retriever(store)        # bi-encoder (bge-m3), in-process
       └─ (optional) rerank_hits(...)             # bge-reranker-v2-m3, in-process
  └─ generation_eval.evaluate_generation(
         gen_provider  = LlamaCppProvider["gguf"],  # Qwen2.5-3B, in-process
         judge_provider= LlamaCppProvider["gguf"])
  └─ report → var/data/eval/baseline.json
```

No socket opened, no key read, no daemon. First run downloads weights; every run
after is offline.

## 7. Gate execution (after the components exist)

| Gate | Code change to *measure*? | How to run (env + command) |
|---|---|---|
| Baseline | no | `KB_EMBEDDINGS_BACKEND=st KB_LLM_PROVIDER=gguf` → `eval_rag run --judge` |
| **C** reranker | no (env only) | `…KB_RERANK_ENABLED=true KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3` → `run` → `compare` |
| **D** top_k | no (env only) | `eval_sweep --golden … --values 5,8,10,12 --judge` → pick argmax `completeness` |
| **B** e5 | no (env only) | `ST_EMBED_MODEL=intfloat/multilingual-e5-base VECTOR_E5_PREFIX=true` → reindex → `run` |

Each gate: keep the change **iff** the metric improves vs `baseline.json`, else
revert (the runbook's rule). Gate B's "official" surface is v1; the MVP-store path
above is the **no-server** way to get the same `recall@k`/`mrr@k` number locally —
the in-process FAISS backend is available if a v1-surface run is wanted without
Qdrant.

## 8. Build sequence (TDD, each step independently verifiable)

0. **De-risk the platform:** confirm `import llama_cpp` and `import
   sentence_transformers` succeed under `py -3` on this Windows box. If the
   llama-cpp wheel is missing/broken, resolve at this step (prebuilt wheel) before
   building on it — this is the main platform risk.
1. `SentenceTransformerEmbedder` + stub + unit tests (fake model injected) +
   one `@pytest.mark.integration` test loading real bge-m3.
2. e5-prefix plumbing into the store's two call sites (guarded; no-op for bge-m3).
3. `LlamaCppProvider` + selection wiring + stub + unit tests (fake `Llama`) +
   one integration test loading the real GGUF.
4. Reindex under `st`/bge-m3 (`kb-cli reindex --embedder st --force-yes`); refresh
   the curated sidecar (`scripts.build_curated_golden`).
5. Produce and keep `baseline.json` (`run --judge`, offline).
6. Gate C → measure → `compare` → commit default swap iff win.
7. Gate D → `eval_sweep --judge` → set `top_k` iff win.
8. Gate B → e5 reindex → `run` → `compare` → commit iff win.
9. Docs: `.env.example` local block, README embedding/LLM sections, and a new
   "fully-local offline" path appended to the runbook.

## 9. Testing strategy

- **Unit:** dependency-injection everywhere — the embedder accepts a preloaded
  model, the adapter accepts a preloaded inner provider; the eval already takes
  `gen_provider`/`judge_provider`/`retriever` as arguments. No real weights in unit
  tests; deterministic fakes injected directly. **DI is mandatory** because
  `tests/stubs/` ships no `sentence_transformers`/`llama_cpp` stub and
  `install_service_stubs` is opt-in, so a non-DI test would load real multi-GB
  models.
- **No stub changes required:** the unit path never imports the heavy libraries
  (lazy import + DI), so the `service_stubs.py` `DummySentenceTransformer` /
  `DummyLlama` are left untouched. (If a future integration-stub test needs them,
  `DummySentenceTransformer.encode` would need a `normalize_embeddings`/`str`-input
  fix — out of scope here.)
- **Integration (`@pytest.mark.integration`, not `skip`):** real-model smoke tests
  gated off the default CI run (CONTRIBUTING.md prefers markers over `skip`).
- **No new drift surface:** `RAG_SYSTEM_PROMPT` stays byte-identical (existing
  drift test covers it).

## 10. Risks & mitigations

- **Deps declared but not installed here.** `sentence-transformers`, `torch`,
  `llama-cpp-python`, `huggingface-hub` are pinned in `requirements.txt` but this
  dev machine has only `faiss-cpu` + `numpy` installed (the rest is stubbed under
  pytest). "Zero new dependencies" means nothing is *added* to `requirements.txt`
  — but a one-time `pip install` is required before the real eval runs. The
  feature code and its unit tests run without them (DI + lazy imports), so this
  only gates Tasks 5–9, not Tasks 1–4.
- **llama-cpp-python on Windows** may need a prebuilt wheel / MSVC runtime —
  verified at Step 0; documented fallback is `python -m llama_cpp.server` +
  the `custom` HTTP provider if the in-process import proves unworkable.
- **CPU latency:** bge-m3 + a 3B GGUF on CPU are slow per call but trivial over a
  48-chunk corpus; flagged for anyone who grows the corpus.
- **3B judge quality:** approximate — good for *relative* gate deltas, not absolute
  scores. The runbook already warns the tiny corpus makes metrics noisy and
  suggests ingesting more representative documents before trusting deltas.
- **First-run download UX:** ~5 GB on first invocation; emit clear "downloading
  model…" logs (sentence-transformers / hf_hub show progress).
- **Signature looseness:** same-`dim` model swap shares a signature — pre-existing,
  accepted; mitigated by pinning the model in code.

## 11. Open decisions (resolved)

- Model tier — **resolved:** quality ~5 GB (bge-m3 / bge-reranker-v2-m3 / Qwen2.5-3B).
- LLM integration — **resolved:** in-process `llama-cpp` (not a local server, not Ollama).
- Default auto-selection of the local LLM — **resolved:** explicit
  `KB_LLM_PROVIDER=gguf` by default; opt-in `KB_LLM_LOCAL_FALLBACK` for zero-config.
