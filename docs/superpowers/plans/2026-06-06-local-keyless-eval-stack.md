# Local, keyless, in-process eval stack — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the RAG eval (`reindex` → `eval_rag run --judge` → `eval_sweep` → `compare`) run fully offline with no API keys and no background daemon, unblocking runbook gates B/C/D.

**Architecture:** Add one in-process embedder backend (`KB_EMBEDDINGS_BACKEND=st`, sentence-transformers, default `BAAI/bge-m3`) and one thin adapter (`KB_LLM_PROVIDER=gguf`) that wraps the **existing** in-process `app/llm/llama_cpp_provider.LlamaCppProvider` into the eval's provider interface. Reuse `scripts/download_model.py` for the one-time GGUF fetch. Zero new dependencies — `sentence-transformers`, `torch`, `llama-cpp-python`, `huggingface-hub` are already in `requirements.txt`.

**Tech Stack:** Python 3.13 (`py -3` launcher, no venv), pytest, sentence-transformers, llama-cpp-python, huggingface-hub. Design doc: `docs/superpowers/specs/2026-06-06-local-keyless-eval-stack-design.md`.

---

## File Structure

- **Modify** `app/services/kb_embeddings.py` — add `SentenceTransformerEmbedder` + an `st` branch in `_build_from_env`.
- **Modify** `app/services/kb_store.py` (line ~524) — use `embed_query()` for the search query when the embedder exposes it.
- **Create** `tests/test_kb_embeddings_st.py` — DI unit tests + one `@pytest.mark.integration` real-model test.
- **Modify** `app/services/kb_llm.py` — add `GgufEvalProvider` adapter + a `gguf` branch in `select_provider`.
- **Create** `tests/test_kb_llm_gguf.py` — DI unit tests + one `@pytest.mark.integration` real-model test.
- **Modify** `models/model_manifest.json` — add a `qwen2.5-3b-instruct` entry.
- **Modify** `.env.example` — add a "fully-local offline" block (3 new keys only).
- **Modify (conditional, gate C)** `app/services/kb_rerank.py` & `app/retriever/rerank.py` — swap `DEFAULT_MODEL_NAME` iff gate C wins.
- **Modify** `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md` & `README.md` — document the offline path.

**Conventions:** Windows, `py -3`. Run the focused test file after each change; do not run unrelated suites. Conventional Commits. End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 0: Platform setup (install the ML deps, then verify)

The heavy ML stack is **declared in `requirements.txt` but NOT installed** in this dev env (only `faiss-cpu` + `numpy` are present). The code in Tasks 1–4 and its unit tests run **without** these libs (DI + lazy imports), so do this task only when you are ready to run the real eval (Tasks 5–9). `llama-cpp-python` on Windows is the one real risk.

- [ ] **Step 1: Install the in-process stack (one-time, ~1.5 GB incl. CPU torch)**

Install in TWO commands — do **not** put `llama-cpp-python` in the same command as
the others. On a machine without a C/C++ compiler (no MSVC/CMake — the common
Windows case here) `pip install llama-cpp-python` tries to build from source and
fails, and the new pip resolver then **aborts the whole transaction** (nothing
installs). Verified working on this box (Python 3.13, Windows):

```powershell
# 1) Embedder stack — normal wheels, no compiler needed:
py -3 -m pip install sentence-transformers torch huggingface-hub
# 2) llama.cpp — PREBUILT CPU wheel (avoids the source build):
py -3 -m pip install --only-binary=:all: llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
Expected: both complete; the second pulls e.g. `llama_cpp_python-0.3.26-py3-none-win_amd64.whl`.
NOTE: pip installs versions *newer* than `requirements.txt` pins (torch 2.12 vs
2.5.1, transformers 5.x vs 4.57, hub 1.x vs 0.35) — fine for bge-m3/Qwen runs. Do
NOT `pip install -r requirements.txt` to "fix" the pins: it re-triggers the
llama-cpp source build and fails the same way.

- [ ] **Step 2: Verify the imports**

Run:
```powershell
py -3 -c "import sentence_transformers, torch; print('st ok')"
py -3 -c "import llama_cpp; print('llama_cpp ok')"
py -3 -c "import huggingface_hub; print('hub ok')"
```
Expected: `st ok` / `llama_cpp ok` / `hub ok`, no traceback. If `import llama_cpp` fails, STOP and resolve the wheel before continuing — the LLM half depends on it.

- [ ] **Step 3: Confirm the eval suite is green at baseline**

Run: `py -3 -m pytest tests/test_eval_metrics.py tests/test_eval_retrieval.py -q`
Expected: all pass (this is the untouched harness we build against).

---

## Task 1: `SentenceTransformerEmbedder` backend (`KB_EMBEDDINGS_BACKEND=st`)

**Files:**
- Create: `tests/test_kb_embeddings_st.py`
- Modify: `app/services/kb_embeddings.py`

- [ ] **Step 1: Write the failing unit tests (dependency-injected — no real model)**

Create `tests/test_kb_embeddings_st.py`:
```python
"""Unit tests for the in-process sentence-transformers embedder backend."""

from __future__ import annotations

import pytest

from app.services.kb_embeddings import SentenceTransformerEmbedder, _build_from_env


class _FakeST:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.last: str | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, text, **kwargs):
        import numpy as np

        self.last = text
        # Deterministic, text-dependent, fixed-length vector.
        seed = float(len(text) % 97 + 1)
        return np.full((self._dim,), seed, dtype=np.float32)


def test_st_embedder_name_dim_and_embed() -> None:
    emb = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", model=_FakeST(8))
    assert emb.name == "st"
    assert emb.model == "BAAI/bge-m3"
    assert emb.dimension == 8
    vec = emb.embed("привет мир")
    assert isinstance(vec, list) and len(vec) == 8
    assert all(isinstance(v, float) for v in vec)


def test_st_backend_is_selected_by_env_without_loading() -> None:
    # Building from env must NOT load a real model (no `model=` injected).
    emb = _build_from_env({"KB_EMBEDDINGS_BACKEND": "st", "ST_EMBED_MODEL": "BAAI/bge-m3"})
    assert emb.name == "st"
    assert getattr(emb, "model", None) == "BAAI/bge-m3"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_kb_embeddings_st.py -v`
Expected: FAIL — `ImportError: cannot import name 'SentenceTransformerEmbedder' from 'app.services.kb_embeddings'`.

- [ ] **Step 3: Implement the embedder class**

In `app/services/kb_embeddings.py`, add this class immediately after the `OllamaEmbedder` class (after its `embed` method, before `_extract_first_embedding`):
```python
class SentenceTransformerEmbedder:
    """In-process embedder backed by a local sentence-transformers model.

    Keyless: weights are fetched once from the HuggingFace hub into the local
    cache on first use, then reused offline. ``embed`` applies the e5
    ``passage: `` prefix and ``embed_query`` the ``query: `` prefix when the
    model is e5-family and prefixing is enabled (no-op otherwise — see
    ``app.retriever.e5``). The heavy import is lazy so importing this module
    stays cheap; pass ``model=`` to inject a fake in tests.
    """

    def __init__(
        self,
        *,
        model_name: str,
        e5_prefix_enabled: bool = False,
        model: object | None = None,
    ) -> None:
        self.name = "st"
        self.model = model_name
        self._e5_enabled = e5_prefix_enabled
        self._model = model
        self._dimension: Optional[int] = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy; lazy

            self._model = SentenceTransformer(self.model)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._ensure_model()
            getter = getattr(model, "get_sentence_embedding_dimension", None)
            dim = getter() if callable(getter) else None
            self._dimension = int(dim) if dim else len(self._encode("dim-probe"))
        return self._dimension

    def _encode(self, text: str) -> list[float]:
        model = self._ensure_model()
        vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        tolist = getattr(vec, "tolist", None)
        seq = tolist() if callable(tolist) else vec
        return [float(v) for v in seq]

    def embed(self, text: str) -> list[float]:
        prepared = e5_prefix(text, role="passage", model=self.model, enabled=self._e5_enabled)
        return self._encode(prepared)

    def embed_query(self, text: str) -> list[float]:
        prepared = e5_prefix(text, role="query", model=self.model, enabled=self._e5_enabled)
        return self._encode(prepared)
```

Add the e5 helper import near the top of the file, after the existing `from app.services.kb_store import EMBEDDING_DIM, embed as hashing_embed` line:
```python
from app.retriever.e5 import e5_prefix
```

- [ ] **Step 4: Wire the `st` backend into `_build_from_env`**

In `app/services/kb_embeddings.py`, change the unknown-backend guard set to include `"st"`:
```python
    elif explicit not in {"ollama", "api", "hash", "st"}:
        LOGGER.warning("Unknown KB_EMBEDDINGS_BACKEND=%r; falling back", explicit)
```
Then insert this branch immediately before the `if explicit == "ollama" ...` block:
```python
    if explicit == "st":
        st_model = _env("ST_EMBED_MODEL", env) or "BAAI/bge-m3"
        e5_enabled = (_env("VECTOR_E5_PREFIX", env) or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        record_embedder_backend("st")
        return SentenceTransformerEmbedder(model_name=st_model, e5_prefix_enabled=e5_enabled)
```
Finally add `"SentenceTransformerEmbedder",` to the `__all__` list.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_kb_embeddings_st.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Lint the touched file**

Run: `py -3 -m ruff check app/services/kb_embeddings.py tests/test_kb_embeddings_st.py; py -3 -m black --check app/services/kb_embeddings.py tests/test_kb_embeddings_st.py`
Expected: no errors (run `py -3 -m black app/services/kb_embeddings.py tests/test_kb_embeddings_st.py` first if black reports reformat).

- [ ] **Step 7: Commit**

```powershell
git add app/services/kb_embeddings.py tests/test_kb_embeddings_st.py
git commit -m "feat(embeddings): in-process sentence-transformers backend (KB_EMBEDDINGS_BACKEND=st)"
```

---

## Task 2: e5 query/passage prefixing + store wiring

The embedder already prefixes `passage:` in `embed()` (used by indexing/reindex) and `query:` in `embed_query()`. The MVP store must call `embed_query()` for the search query so the prefixes are asymmetric for e5 models. No-op for the `bge-m3` default.

**Files:**
- Modify: `tests/test_kb_embeddings_st.py` (add prefix tests)
- Modify: `app/services/kb_store.py` (line ~524)

- [ ] **Step 1: Write the failing prefix + store-wiring tests**

Append to `tests/test_kb_embeddings_st.py`:
```python
class _RecordingST:
    def __init__(self) -> None:
        self.last: str | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, text, **kwargs):
        import numpy as np

        self.last = text
        return np.zeros((4,), dtype=np.float32)


def test_e5_passage_and_query_prefixes_when_enabled() -> None:
    rec = _RecordingST()
    emb = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-base", e5_prefix_enabled=True, model=rec
    )
    emb.embed("текст документа")
    assert rec.last == "passage: текст документа"
    emb.embed_query("мой вопрос")
    assert rec.last == "query: мой вопрос"


def test_no_prefix_for_bge_even_when_enabled() -> None:
    rec = _RecordingST()
    emb = SentenceTransformerEmbedder(
        model_name="BAAI/bge-m3", e5_prefix_enabled=True, model=rec
    )
    emb.embed("текст")
    assert rec.last == "текст"


class _RecordingEmbedder:
    name = "st"
    dimension = 4

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.1, 0.2, 0.3, 0.4]


def test_store_uses_embed_query_for_search(tmp_path) -> None:
    from app.services.kb_store import KnowledgeBaseStore

    fake = _RecordingEmbedder()
    store = KnowledgeBaseStore(db_path=str(tmp_path / "kb.sqlite"), embedder=fake)
    store.add_document("doc", text="первый чанк текста. второй чанк текста.")
    store.search("поисковый запрос", top_k=3)
    assert fake.query_calls and fake.query_calls[-1] == "поисковый запрос"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_kb_embeddings_st.py -k "prefix or embed_query" -v`
Expected: `test_e5_passage_and_query_prefixes_when_enabled` and `test_no_prefix_for_bge_even_when_enabled` PASS (logic already in Task 1), but `test_store_uses_embed_query_for_search` FAILS (`query_calls` empty — the store still calls `embed`).

- [ ] **Step 3: Make the store prefer `embed_query`**

In `app/services/kb_store.py`, replace the single line at ~524:
```python
        q_vec = self._embedder.embed(cleaned)
```
with:
```python
        _embed_query = getattr(self._embedder, "embed_query", None)
        q_vec = _embed_query(cleaned) if callable(_embed_query) else self._embedder.embed(cleaned)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_kb_embeddings_st.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Run adjacent store/search tests for regressions**

Run: `py -3 -m pytest tests/test_kb_mvp_ask_retrieval.py tests/test_kb_store_retrieval_health.py -q`
Expected: PASS (the `getattr` fallback keeps hash/ollama/api embedders unchanged).

- [ ] **Step 6: Commit**

```powershell
git add app/services/kb_store.py tests/test_kb_embeddings_st.py
git commit -m "feat(retrieval): use embed_query for e5 query prefixing in MVP store"
```

---

## Task 3: `GgufEvalProvider` adapter + selection wiring (`KB_LLM_PROVIDER=gguf`)

Adapt the existing `app/llm/llama_cpp_provider.LlamaCppProvider` into the eval's `LLMProvider` interface (`name`/`model`/`generate -> LLMResponse`).

**Files:**
- Create: `tests/test_kb_llm_gguf.py`
- Modify: `app/services/kb_llm.py`

- [ ] **Step 1: Write the failing tests (DI — no real GGUF)**

Create `tests/test_kb_llm_gguf.py`:
```python
"""Unit tests for the in-process GGUF eval provider adapter."""

from __future__ import annotations

from app.services.kb_llm import GgufEvalProvider, select_provider


class _FakeInner:
    """Stand-in for app.llm.llama_cpp_provider.LlamaCppProvider."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, *, context=None) -> str:
        self.calls.append((prompt, dict(context or {})))
        return '{"faithfulness":5,"relevance":5,"completeness":5,"citation":5,"rationale":"ok"}'


def test_adapter_shapes_response_and_folds_system() -> None:
    inner = _FakeInner()
    prov = GgufEvalProvider(model_path="/models/qwen2.5-3b-instruct-q4_k_m.gguf", inner=inner)

    assert prov.name == "gguf"
    assert prov.model == "qwen2.5-3b-instruct-q4_k_m.gguf"

    resp = prov.generate("Вопрос?", system="Ты судья.", temperature=0.0, max_tokens=128)
    assert resp.provider == "gguf"
    assert resp.text.startswith("{") and "faithfulness" in resp.text

    sent_prompt, sent_ctx = inner.calls[0]
    assert "Ты судья." in sent_prompt and "Вопрос?" in sent_prompt
    assert sent_ctx["temperature"] == 0.0 and sent_ctx["max_tokens"] == 128


def test_select_provider_gguf_missing_model_returns_none(tmp_path) -> None:
    prov = select_provider(
        {"KB_LLM_PROVIDER": "gguf", "KB_LLM_GGUF_PATH": str(tmp_path / "absent.gguf")}
    )
    assert prov is None


def test_select_provider_gguf_present_does_not_load(tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF\x00\x00\x00")
    prov = select_provider({"KB_LLM_PROVIDER": "gguf", "KB_LLM_GGUF_PATH": str(model)})
    assert prov is not None and prov.name == "gguf"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_kb_llm_gguf.py -v`
Expected: FAIL — `ImportError: cannot import name 'GgufEvalProvider' from 'app.services.kb_llm'`.

- [ ] **Step 3: Implement the adapter + selection branch**

In `app/services/kb_llm.py`, add `from pathlib import Path` to the imports. Then add the adapter class after the `OpenAICompatibleProvider` class:
```python
class GgufEvalProvider:
    """Adapter exposing the in-process llama.cpp provider via the eval interface.

    Wraps :class:`app.llm.llama_cpp_provider.LlamaCppProvider` (constructed lazily
    so importing this module stays cheap). Folds ``system`` into the prompt and
    returns the shared :class:`LLMResponse`. Temperature defaults to 0 for stable
    judge verdicts. Pass ``inner=`` to inject a fake in tests.
    """

    def __init__(self, *, model_path: str, inner: object | None = None) -> None:
        self.name = "gguf"
        self.model = Path(model_path).name
        self._model_path = model_path
        self._inner = inner

    def _ensure_inner(self) -> object:
        if self._inner is None:
            from app.core.config import Settings
            from app.llm.llama_cpp_provider import LlamaCppProvider

            settings = Settings(llm_provider="llama-cpp", llm_model_path=self._model_path)
            self._inner = LlamaCppProvider(settings)
        return self._inner

    def is_available(self) -> bool:
        return Path(self._model_path).is_file()

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        inner = self._ensure_inner()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        context = {
            "temperature": 0.0 if temperature is None else float(temperature),
            "max_tokens": int(max_tokens) if max_tokens else 512,
        }
        started = time.perf_counter()
        text = inner.generate(full_prompt, context=context)  # type: ignore[attr-defined]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            text=(text or "").strip(),
            provider="gguf",
            model=self.model,
            elapsed_ms=round(elapsed_ms, 2),
        )


def _build_gguf_provider(env: Mapping[str, str] | None = None) -> Optional[GgufEvalProvider]:
    path = _env("KB_LLM_GGUF_PATH", env) or "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
    provider = GgufEvalProvider(model_path=path)
    if not provider.is_available():
        LOGGER.warning(
            "GGUF model not found at %s — run scripts/download_model.py "
            "--target qwen2.5-3b-instruct to fetch it.",
            path,
        )
        return None
    return provider
```

Wire it into `select_provider`. Replace the explicit-provider block at the top of `select_provider`:
```python
    explicit = _env("KB_LLM_PROVIDER", env)
    if explicit:
        try:
            return build_provider(explicit, env=env)
        except LLMUnavailable as exc:
            LOGGER.warning("Configured LLM provider unusable: %s", exc)
            return None
```
with:
```python
    explicit = _env("KB_LLM_PROVIDER", env)
    if explicit:
        if explicit.strip().lower() == "gguf":
            return _build_gguf_provider(env)
        try:
            return build_provider(explicit, env=env)
        except LLMUnavailable as exc:
            LOGGER.warning("Configured LLM provider unusable: %s", exc)
            return None
```
Then, just before the final `return None` of `select_provider`, add the opt-in local fallback:
```python
    if (_env("KB_LLM_LOCAL_FALLBACK", env) or "").strip().lower() in {"1", "true", "yes", "on"}:
        return _build_gguf_provider(env)

    return None
```
Finally add `"GgufEvalProvider",` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_kb_llm_gguf.py -v`
Expected: PASS (3 passed). The "present" test writes a fake GGUF file but never loads it (inner is lazy), so no llama.cpp call happens.

- [ ] **Step 5: Run the existing provider-factory tests for regressions**

Run: `py -3 -m pytest tests/test_llm_provider_factory.py tests/test_kb_mvp.py -q`
Expected: PASS (HTTP provider selection is unchanged; the `gguf` branch is additive).

- [ ] **Step 6: Lint + commit**

```powershell
py -3 -m ruff check app/services/kb_llm.py tests/test_kb_llm_gguf.py
py -3 -m black app/services/kb_llm.py tests/test_kb_llm_gguf.py
git add app/services/kb_llm.py tests/test_kb_llm_gguf.py
git commit -m "feat(llm): GGUF eval provider adapter over app/llm (KB_LLM_PROVIDER=gguf)"
```

---

## Task 4: GGUF manifest entry + `.env.example` offline block

**Files:**
- Modify: `models/model_manifest.json`
- Modify: `.env.example`

- [ ] **Step 1: Add the Qwen manifest entry**

In `models/model_manifest.json`, add a second entry (keep `default`):
```json
{
  "default": {
    "model_id": "bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "filename": "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
    "sha256": null,
    "license": "Apache-2.0",
    "url": "https://huggingface.co/bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
    "description": "TinyLlama 1.1B Chat v1.0 quantized to Q4_K_M (~625 MB) for llama.cpp-compatible inference."
  },
  "qwen2.5-3b-instruct": {
    "model_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
    "sha256": null,
    "license": "Apache-2.0",
    "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    "description": "Qwen2.5-3B-Instruct Q4_K_M (~2 GB) — RU-capable in-process judge/gen for the RAG eval."
  }
}
```
NOTE: confirm the exact `filename`/`url` casing against the repo file listing at <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/main> before the download step; adjust both fields if it differs.

- [ ] **Step 2: Add the offline block to `.env.example`**

Append after the embeddings block (after the `EMBEDDINGS_API_MODEL=...` line, ~line 49). Use ONLY new keys (existing `KB_EMBEDDINGS_BACKEND`, `KB_LLM_PROVIDER`, `VECTOR_E5_PREFIX` are referenced in comments, not redefined):
```bash
# --- Полностью локальный, оффлайн, без ключей стек для RAG-eval ------------
# Эмбеддер in-process через sentence-transformers. Включить: KB_EMBEDDINGS_BACKEND=st
# Модель качается один раз в кэш HuggingFace, дальше работает оффлайн.
ST_EMBED_MODEL=BAAI/bge-m3
# LLM-судья/генератор in-process через llama.cpp (GGUF). Включить: KB_LLM_PROVIDER=gguf
# Скачать модель один раз:
#   py -3 scripts/download_model.py --target qwen2.5-3b-instruct \
#     --output models/qwen2.5-3b-instruct-q4_k_m.gguf --allow-missing-hash
KB_LLM_GGUF_PATH=./models/qwen2.5-3b-instruct-q4_k_m.gguf
# Если облачный ключ не задан — авто-выбор локального GGUF (true/false).
KB_LLM_LOCAL_FALLBACK=false
```

- [ ] **Step 3: Verify `.env.example` invariants still hold**

Run: `py -3 -m pytest tests/test_env_example.py -v`
Expected: PASS — no duplicate keys; existing defaults unchanged; the 3 new keys are additive.

- [ ] **Step 4: Commit**

```powershell
git add models/model_manifest.json .env.example
git commit -m "chore(eval): Qwen2.5-3B GGUF manifest entry + .env offline block"
```

---

## Task 5: Reindex under `st` + offline baseline

Operational. Performs the one-time ~5 GB model downloads (bge-m3 + Qwen GGUF) and produces `baseline.json`.

- [ ] **Step 1: Download the GGUF once**

Run:
```powershell
py -3 scripts/download_model.py --target qwen2.5-3b-instruct `
  --output models/qwen2.5-3b-instruct-q4_k_m.gguf --allow-missing-hash
```
Expected: `INFO Model qwen2.5-3b-instruct (downloaded) -> models/...q4_k_m.gguf [~2.0e9 bytes]`. If it 404s, fix the `filename`/`url` in the manifest (Task 4 Step 1 NOTE) and retry.

- [ ] **Step 2: Select the local stack for this shell**

Run:
```powershell
$env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="BAAI/bge-m3"
$env:KB_LLM_PROVIDER="gguf"; $env:KB_LLM_GGUF_PATH="./models/qwen2.5-3b-instruct-q4_k_m.gguf"
```

- [ ] **Step 3: Reindex the MVP corpus under the real embedder (downloads bge-m3 once)**

Run: `py -3 -m scripts.kb_cli reindex --embedder st --force-yes`
Expected: `OK: processed N document(s), re-embedded M chunk(s)`. (First run downloads bge-m3 ~2.2 GB — a progress bar appears.) If it errors `--embedder 'st' but ... resolved to 'hash'`, the env from Step 2 is not set in this shell.

- [ ] **Step 4: Refresh the curated golden sidecar for the new signature**

Run: `py -3 -m scripts.build_curated_golden`
Expected: rewrites `data/eval/golden_curated.sig.json` with `embedder_name="st"`, `dim=1024` (chunk-ids unchanged).

- [ ] **Step 5: Produce the offline baseline (retrieval + judge)**

Run:
```powershell
py -3 -m scripts.eval_rag run --golden data/eval/golden_curated.jsonl `
  --out var/data/eval/baseline.json --judge
```
Expected: a markdown report prints; `var/data/eval/baseline.json` is written. The judge runs on the local Qwen GGUF (CPU — minutes for the small corpus), no network, no key.

- [ ] **Step 6: Commit the baseline artifact**

```powershell
git add var/data/eval/baseline.json data/eval/golden_curated.sig.json
git commit -m "chore(eval): offline baseline.json under local st + gguf stack"
```
(If `var/data/eval/` is gitignored, instead paste the printed report into the PR description and skip the `baseline.json` add — keep the sidecar commit.)

---

## Task 6: Gate C — Russian reranker (measure, then commit iff it wins)

Metric: `mrr@k` / `hit@5` up **and** latency acceptable. No code change to *measure*.

- [ ] **Step 1: Run with the RU reranker enabled (env only)**

Run:
```powershell
$env:KB_RERANK_ENABLED="true"; $env:KB_RERANK_MODEL="BAAI/bge-reranker-v2-m3"
py -3 -m scripts.eval_rag run --golden data/eval/golden_curated.jsonl `
  --out var/data/eval/gate_c_reranker.json --judge
```
Expected: report prints (first run downloads the reranker ~600 MB once).

- [ ] **Step 2: Compare against the baseline**

Run: `py -3 -m scripts.eval_rag compare var/data/eval/baseline.json var/data/eval/gate_c_reranker.json`
Expected: a delta table. Record it for the PR.

- [ ] **Step 3 (conditional): If `mrr@k`/`hit@5` improved, make the RU reranker the default**

Only if the gate passed. In `app/services/kb_rerank.py` change:
```python
DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```
to
```python
DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
```
Apply the identical change in `app/retriever/rerank.py` (its `DEFAULT_MODEL_NAME` at line ~18). Then update any test asserting the old name:

Run: `py -3 -m pytest tests/test_rerank.py tests/test_reranking.py -q` — if a test asserts `"ms-marco-MiniLM-L-6-v2"`, change that expected string to `"BAAI/bge-reranker-v2-m3"`. Re-run until green.

Also flip the `.env.example` defaults so the shipped config matches: set `KB_RERANK_ENABLED=true` and `KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3`, then re-run `py -3 -m pytest tests/test_env_example.py -q`.

- [ ] **Step 4: Commit (the table always; the default-swap only if it won)**

```powershell
git add app/services/kb_rerank.py app/retriever/rerank.py .env.example tests/test_rerank.py tests/test_reranking.py
git commit -m "feat(rerank): default to BAAI/bge-reranker-v2-m3 (gate C: mrr/hit +X.XX)"
```
If the gate did NOT win, skip the code changes and instead record the negative result in the PR/runbook (no commit, or a docs-only note).

---

## Task 7: Gate D — top_k sweep (measure, then set the winner)

Metric: argmax `completeness` without dropping `faithfulness`.

- [ ] **Step 1: Sweep top_k with the local judge**

Run:
```powershell
py -3 -m scripts.eval_sweep --golden data/eval/golden_curated.jsonl `
  --values 5,8,10,12 --judge
```
Expected: a markdown table with one row per top_k (`recall@5`, `recall@10`, `completeness`, `faithfulness`). Record it.

- [ ] **Step 2 (conditional): Set the winning top_k**

Pick the argmax `completeness` that does not reduce `faithfulness`. Locate the current MVP `ask` retrieval depth:

Run: `py -3 -m pytest -q -k nothing 2>$null; Select-String -Path app/api/kb_mvp.py -Pattern "top_k"` (or `grep -n "top_k" app/api/kb_mvp.py`)
Expected: the line(s) where the `ask` handler calls `store.search(..., top_k=...)`. If the winning value differs from what you find, change that literal (and/or the v1 `RETRIEVE_TOPK` default in `.env.example`). Add/adjust a focused test only if a named default constant changed.

- [ ] **Step 3: Commit**

```powershell
git add app/api/kb_mvp.py
git commit -m "perf(retrieval): set MVP ask top_k=N (gate D: completeness +X.XX, faithfulness flat)"
```
If the current default already wins, record the table in the PR and skip the code change.

---

## Task 8: Gate B — e5 embedder (measure, then commit iff it wins)

Metric: `recall@k` / `mrr@k` up. Uses the same `st` backend with an e5 model + prefixing.

- [ ] **Step 1: Reindex under e5 with prefixing, then run**

Run:
```powershell
$env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="intfloat/multilingual-e5-base"
$env:VECTOR_E5_PREFIX="true"
py -3 -m scripts.kb_cli reindex --embedder st --force-yes
py -3 -m scripts.build_curated_golden
py -3 -m scripts.eval_rag run --golden data/eval/golden_curated.jsonl `
  --out var/data/eval/gate_b_e5.json --judge
```
Expected: reindex re-embeds passages with the `passage:` prefix (downloads e5-base ~1.1 GB once); run produces the report. The signature now reads `dim=768`.

- [ ] **Step 2: Compare against the baseline**

NOTE: the baseline was built under bge-m3 (dim 1024) and a different sidecar. Compare retrieval metrics from the printed reports (recall/mrr) rather than via the signature-pinned `compare` if the sidecars differ. Record the delta.

- [ ] **Step 3 (conditional): If e5 wins, document it as the recommended embedder**

If `recall@k`/`mrr@k` improved, set `ST_EMBED_MODEL=intfloat/multilingual-e5-base` and `VECTOR_E5_PREFIX=true` as the recommended defaults in `.env.example` and the runbook (Task 9). Re-run `py -3 -m pytest tests/test_env_example.py -q`. Otherwise record the negative result and keep bge-m3.

- [ ] **Step 4: Restore the bge-m3 baseline index for day-to-day use (if e5 lost)**

If e5 did not win, reindex back:
```powershell
$env:ST_EMBED_MODEL="BAAI/bge-m3"; $env:VECTOR_E5_PREFIX="false"
py -3 -m scripts.kb_cli reindex --embedder st --force-yes; py -3 -m scripts.build_curated_golden
```

- [ ] **Step 5: Commit the decision**

```powershell
git add .env.example
git commit -m "docs(eval): record gate B (e5 vs bge-m3) result and recommended embedder"
```

---

## Task 9: Document the offline path

**Files:**
- Modify: `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md`
- Modify: `README.md`

- [ ] **Step 1: Add the fully-local option to the runbook**

In section "0. Stand up a real embedder + LLM (pick one)", add a third option above Ollama:
```markdown
**Fully local, in-process, keyless (no daemon, no key; ~5 GB one-time download):**

```powershell
py -3 scripts/download_model.py --target qwen2.5-3b-instruct `
  --output models/qwen2.5-3b-instruct-q4_k_m.gguf --allow-missing-hash
$env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="BAAI/bge-m3"
$env:KB_LLM_PROVIDER="gguf"; $env:KB_LLM_GGUF_PATH="./models/qwen2.5-3b-instruct-q4_k_m.gguf"
```
Then `reindex --embedder st` (step 1) and proceed. bge-m3 + Qwen2.5-3B run in-process on CPU; models are cached after the first fetch.
```

- [ ] **Step 2: Add a short "Offline eval" note to README**

In the embeddings/LLM configuration section of `README.md`, document `KB_EMBEDDINGS_BACKEND=st` (+ `ST_EMBED_MODEL`) and `KB_LLM_PROVIDER=gguf` (+ `KB_LLM_GGUF_PATH`) as the keyless, in-process option, pointing at `scripts/download_model.py` for the one-time fetch.

- [ ] **Step 3: Commit**

```powershell
git add docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md README.md
git commit -m "docs(eval): document fully-local keyless offline eval stack"
```

---

## Optional: real-model integration smoke tests

These download real models, so they are marked `@pytest.mark.integration` (registered in `pytest.ini`) and excluded from the default suite. Add them if you want CI (with cached models) to exercise the real path.

- [ ] **Embedder integration test** — append to `tests/test_kb_embeddings_st.py`:
```python
import pytest


@pytest.mark.integration
def test_st_embedder_real_bge_m3() -> None:
    emb = SentenceTransformerEmbedder(model_name="BAAI/bge-m3")
    assert emb.dimension == 1024
    vec = emb.embed("Договор аренды нежилого помещения")
    assert len(vec) == 1024
```
Run (opt-in): `py -3 -m pytest tests/test_kb_embeddings_st.py -m integration -v`

- [ ] **Provider integration test** — append to `tests/test_kb_llm_gguf.py`:
```python
import os

import pytest


@pytest.mark.integration
def test_gguf_real_generate() -> None:
    path = os.environ.get("KB_LLM_GGUF_PATH", "./models/qwen2.5-3b-instruct-q4_k_m.gguf")
    prov = GgufEvalProvider(model_path=path)
    if not prov.is_available():
        pytest.skip("GGUF model not downloaded")
    resp = prov.generate("Ответь одним словом: столица России?", system="Ты помощник.")
    assert isinstance(resp.text, str) and resp.text
```
Run (opt-in): `py -3 -m pytest tests/test_kb_llm_gguf.py -m integration -v`

- [ ] **Commit**
```powershell
git add tests/test_kb_embeddings_st.py tests/test_kb_llm_gguf.py
git commit -m "test(eval): opt-in integration smoke tests for st embedder + gguf provider"
```
