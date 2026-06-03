# RAG Quick-Wins (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Apply the cheap, high-confidence retrieval/generation improvements from the spec's §8, each **gated by the eval harness** — keep a change only if its `compare` report shows its target metric improved.

**Architecture:** Builds on the PR1–PR3 harness (`app/eval/`, `scripts/eval_rag.py`). Each PR is a small, reversible change to the shared retriever/prompt layer, measured before/after with `eval_rag.py run` + `compare`. Nothing ships blind.

**Tech Stack:** Python 3.12 (Windows `py -3`, no venv), pytest, the eval harness, `app/retriever/{qdrant,faiss}.py`, `app/services/kb_rerank.py`, `app/api/kb_mvp.py`. Spec: `docs/superpowers/specs/2026-06-03-rag-answer-quality-eval-design.md` §8.

---

## Prerequisites (HARD GATE — do not start tasks without these)

Phase 2 is **measurement-driven**; without a real baseline every task is a guess. Before Task A:

1. **A real embedder.** Either Ollama (`KB_EMBEDDINGS_BACKEND=ollama` + `OLLAMA_EMBED_MODEL=...`, e.g. `bge-m3` / `nomic-embed-text`) or an API endpoint (`KB_EMBEDDINGS_BACKEND=api` + `EMBEDDINGS_API_BASE_URL` + `EMBEDDINGS_API_KEY`). The MVP default is the hashing fallback — `eval_rag.py run` will refuse it.
2. **An LLM provider** for `generate` (teacher) and `--judge`: a key (`DEEPSEEK_API_KEY` etc.) or local Ollama (`KB_LLM_PROVIDER=ollama` + a chat model).
3. **A baseline.** With the above set and the corpus (re)indexed under the real embedder:
   ```powershell
   py -3 -m scripts.kb_cli reindex --embedder <name>   # if the corpus was indexed under another embedder
   py -3 -m scripts.eval_rag generate --out var/data/eval/golden_auto.jsonl --limit 200
   py -3 -m scripts.eval_rag run --golden var/data/eval/golden_auto.jsonl --out var/data/eval/baseline.json --judge
   ```
   Keep `baseline.json` — every task below compares against it.

> Each task's "gate" is: re-run `eval_rag.py run ... --out var/data/eval/<task>.json` after the change, then `py -3 -m scripts.eval_rag compare var/data/eval/baseline.json var/data/eval/<task>.json`. **Keep the change only if the named metric improved and latency is acceptable; otherwise revert.** `log`/record what you dropped.

---

## Task A — PR4: Embedder reality guard (promote + document)

The `run` guard already refuses hashing (`scripts/eval_rag.py`). This task promotes the check to a reusable helper and documents real-embedder setup, so other entrypoints can reuse it.

**Files:**
- Modify: `scripts/eval_rag.py`
- Create: `app/eval/guards.py`
- Test: `tests/test_eval_guards.py`
- Modify: `README.md` (embedding-models section) — add the "configure a real embedder before eval" note.

- [ ] **Step 1: Failing test** — `tests/test_eval_guards.py`:
```python
import pytest
from app.eval.dataset import CorpusSignature
from app.eval.guards import ensure_real_embedder


def test_ensure_real_embedder_refuses_hash():
    sig = CorpusSignature(doc_count=1, max_chunk_id=1, embedder_name="hash", dim=256)
    with pytest.raises(SystemExit, match="hashing"):
        ensure_real_embedder(sig, allow_hashing=False)


def test_ensure_real_embedder_allows_real_or_flagged():
    real = CorpusSignature(doc_count=1, max_chunk_id=1, embedder_name="ollama", dim=1024)
    ensure_real_embedder(real, allow_hashing=False)  # no raise
    hashed = CorpusSignature(doc_count=1, max_chunk_id=1, embedder_name="hash", dim=256)
    ensure_real_embedder(hashed, allow_hashing=True)  # no raise
```
- [ ] **Step 2:** `py -3 -m pytest tests/test_eval_guards.py -v` → FAIL (no module).
- [ ] **Step 3:** Create `app/eval/guards.py`:
```python
"""Reusable eval guards (loud, never silent)."""
from __future__ import annotations

from app.eval.dataset import CorpusSignature


def ensure_real_embedder(sig: CorpusSignature, *, allow_hashing: bool) -> None:
    if sig.embedder_name == "hash" and not allow_hashing:
        raise SystemExit(
            "Refusing to produce a baseline on the hashing embedder (near-random "
            "results). Configure KB_EMBEDDINGS_BACKEND=ollama|api (+ model/base), or "
            "pass --allow-hashing for a throwaway smoke run."
        )
```
Then in `scripts/eval_rag.py` `cmd_run`, replace the inline hashing check with `guards.ensure_real_embedder(sig, allow_hashing=args.allow_hashing)` (import `from app.eval import guards`). Keep the existing CLI test green.
- [ ] **Step 4:** `py -3 -m pytest tests/test_eval_guards.py tests/test_eval_cli.py -v` → PASS.
- [ ] **Step 5:** Commit `feat(eval): reusable embedder-reality guard + docs`.

**Gate:** prerequisite (not metric-gated).

---

## Task B — PR5: e5 `query:` / `passage:` prefixes

`multilingual-e5` is trained to receive `"query: "` on queries and `"passage: "` on indexed text; omitting them lowers recall. Confirmed missing: `app/retriever/qdrant.py` and `app/retriever/faiss.py` call `_batched_encode` on raw text at both index and query time.

**Approach:** add a prefix helper applied at the two CALL SITES (not inside `_batched_encode`, which is shared and role-blind), behind a config flag, only for e5-family models. **Requires a reindex** (passages must be embedded with `passage:` for the `query:` side to match).

**Files:**
- Modify: `app/core/config.py` (add `VECTOR_E5_PREFIX: bool = False`)
- Create: `app/retriever/e5.py` (the helper)
- Modify: `app/retriever/qdrant.py` (apply at index + search call sites)
- Modify: `app/retriever/faiss.py` (same)
- Test: `tests/test_retriever_e5_prefix.py`
- Modify: `.env.example` (document `VECTOR_E5_PREFIX`)

- [ ] **Step 1: Failing test** — `tests/test_retriever_e5_prefix.py`:
```python
from app.retriever.e5 import e5_prefix


def test_e5_prefix_applies_for_e5_models_when_enabled():
    assert e5_prefix("кто платит налог", role="query",
                     model="intfloat/multilingual-e5-small", enabled=True) == "query: кто платит налог"
    assert e5_prefix("текст нормы", role="passage",
                     model="intfloat/multilingual-e5-small", enabled=True) == "passage: текст нормы"


def test_e5_prefix_noop_when_disabled_or_non_e5():
    assert e5_prefix("q", role="query", model="intfloat/multilingual-e5-small", enabled=False) == "q"
    assert e5_prefix("q", role="query", model="BAAI/bge-m3", enabled=True) == "q"
```
- [ ] **Step 2:** `py -3 -m pytest tests/test_retriever_e5_prefix.py -v` → FAIL.
- [ ] **Step 3:** Create `app/retriever/e5.py`:
```python
"""e5-family query/passage prefixing (https://huggingface.co/intfloat/multilingual-e5-small)."""
from __future__ import annotations


def _is_e5(model: str) -> bool:
    return "e5" in (model or "").lower()


def e5_prefix(text: str, *, role: str, model: str, enabled: bool) -> str:
    """Prepend 'query: ' / 'passage: ' for e5 models when enabled; else return text unchanged."""
    if not enabled or not _is_e5(model):
        return text
    if role not in ("query", "passage"):
        raise ValueError(f"role must be 'query' or 'passage', got {role!r}")
    return f"{role}: {text}"
```
- [ ] **Step 4:** Wire the call sites (verify exact method names first — grep `def search`, the index/upsert method, and the configured model name on the store). In `qdrant.py`: at the query site (`_batched_encode([query])`, ~`:273`) pass `e5_prefix(query, role="query", model=<model>, enabled=<flag>)`; at the index site (where chunk texts are embedded, ~`:217`) map each text through `e5_prefix(t, role="passage", ...)`. Mirror in `faiss.py` (~`:229` query, `:152` index). Read the flag from settings (`app/core/config.py`).
- [ ] **Step 5:** `py -3 -m pytest tests/test_retriever_e5_prefix.py tests/test_api_v1_search.py -q` → PASS (the stub `SentenceTransformer.encode` hashes text, so prefixed text simply produces different deterministic vectors — assert the prefix reaches `encode` by spying on the stub if practical).
- [ ] **Step 6:** Commit `feat(retriever): optional e5 query/passage prefixes (VECTOR_E5_PREFIX)`.

**Gate (REQUIRES REINDEX):** on a snapshot reindexed with `VECTOR_E5_PREFIX=true`, regenerate the golden set, `run`, and `compare` vs baseline. **Keep iff `recall@k`/`mrr@k` improve.** Document the reindex requirement in the PR.

---

## Task C — PR6: Russian reranker on both surfaces

Default cross-encoder `ms-marco-MiniLM-L-6-v2` is English; `.env.example:67` already recommends `BAAI/bge-reranker-v2-m3` for RU/multilingual. Query-time only — **no reindex.**

**Files:**
- Modify: `.env.example` (`KB_RERANK_MODEL` default → `BAAI/bge-reranker-v2-m3`; set `KB_RERANK_ENABLED=true`; align v1 `RERANK_ENABLED=true`)
- Modify: `app/services/kb_rerank.py` (`_DEFAULT_MODEL` → `BAAI/bge-reranker-v2-m3`)
- Test: `tests/test_reranking.py` (update the asserted default model name)

- [ ] **Step 1:** Update the default-model test in `tests/test_reranking.py` to expect `BAAI/bge-reranker-v2-m3`; run → FAIL.
- [ ] **Step 2:** Change `_DEFAULT_MODEL` in `app/services/kb_rerank.py:32` and the `.env.example` defaults.
- [ ] **Step 3:** `py -3 -m pytest tests/test_reranking.py tests/test_rerank.py -q` → PASS.
- [ ] **Step 4:** Commit `feat(rerank): default to multilingual bge-reranker-v2-m3, enable on both surfaces`.

**Gate:** with `KB_RERANK_ENABLED=true`, `run` + `compare` vs baseline. **Keep iff `mrr@k`/`hit@5` improve AND added latency is acceptable** (the cross-encoder is ~600 MB and adds per-query latency — record the elapsed_ms delta). If latency is unacceptable, keep enabled only for `KB_RERANK_CANDIDATES` ≤ a tuned cap.

---

## Task D — PR7: top_k / context-budget sweep (parameterized)

Data-driven: the winner depends on the baseline. No fixed values — sweep and pick the argmax of the gate metric.

**Files:**
- Create: `scripts/eval_sweep.py` (thin loop over `eval_rag.py run` for a set of `top_k` values, writing one report each, then `compare` to baseline)
- (After picking) Modify: MVP `top_k` default in `app/api/kb_mvp.py` (`ask`) and/or v1 `RETRIEVE_TOPK` / the 3000-token context budget in `chat_orchestrator.py`.

- [ ] **Step 1:** Implement `scripts/eval_sweep.py` accepting `--values 5,8,10,12`, running `cmd_run` per value (monkeypatching `top_k`, or via a new `--top-k` arg on `run`) and printing a combined table of `recall@k`, `completeness`, `faithfulness` per value.
- [ ] **Step 2:** Add a `--top-k` int arg to `eval_rag.py run` (defaults to `max(RETRIEVAL_KS)`), threaded into `evaluate`/`evaluate_generation`. Test it parses and changes retrieval depth.
- [ ] **Step 3:** Run the sweep against the real baseline corpus. Pick the value maximizing `completeness` **without** dropping `faithfulness` below baseline.
- [ ] **Step 4:** Set that value as the default; `compare` to confirm; commit `feat(retrieval): tune top_k/context budget (<chosen value>, evidence in PR)`.

**Gate:** `completeness` ↑ without `faithfulness` ↓. Paste the sweep table in the PR.

---

## Task E — PR8: Prompt tightening (grounding + citation discipline)

Sharpen the system prompt to enforce per-claim citations and an explicit "say I don't know" when context is insufficient.

**Files:**
- Modify: `app/api/kb_mvp.py:406` (`_RAG_SYSTEM_PROMPT`)
- Modify: `app/services/chat_orchestrator.py` (v1 system prompt, ~`:240`)
- Modify: `app/eval/generation_eval.py:RAG_SYSTEM_PROMPT` (re-pin to the new MVP text) + the drift test stays green.

- [ ] **Step 1:** Draft the tightened prompt (keep Russian, keep `[N]` format; add: "Каждое утверждение подкрепляй ссылкой [N]. Если в контексте нет ответа — прямо скажи об этом и не выдумывай."). Update `_RAG_SYSTEM_PROMPT` and the v1 prompt.
- [ ] **Step 2:** Update `generation_eval.RAG_SYSTEM_PROMPT` to the exact new MVP text; run `tests/test_eval_generation.py::test_system_prompt_matches_production` → PASS.
- [ ] **Step 3:** `run --judge` + `compare` vs baseline.
- [ ] **Step 4:** Commit `feat(rag): tighten grounding/citation prompt (evidence in PR)`.

**Gate:** `faithfulness` ↑ and `refusal_correct` ↑ (or steady) without `relevance` regressing. Revert if grounding tightening over-suppresses valid answers.

---

## Sequencing & PRs

Run in order A → B → C → D → E, **one PR each** (repo's ~≤400 LoC norm, PR-series style). After each, the `compare` table is the PR's evidence. B and C are the highest-confidence wins; D and E are tuning. Re-baseline after B (reindex changes the index) before measuring C–E, or measure each against the immediately-preceding state and note it.

## Self-review (completed)
- Spec §8 coverage: §8.1→A, §8.2→B, §8.3→C, §8.4→D, §8.5→E. All gates are `compare`-driven per §8/§9.
- No placeholders in code steps; the only parameterized value (top_k winner in D) is explicitly chosen at execution from the sweep, by design.
- Type/flag consistency: `VECTOR_E5_PREFIX` (config + e5.py + call sites), `KB_RERANK_MODEL`/`KB_RERANK_ENABLED` (kb_rerank + env), `RAG_SYSTEM_PROMPT` drift-pin (kb_mvp ↔ generation_eval) are referenced consistently.
