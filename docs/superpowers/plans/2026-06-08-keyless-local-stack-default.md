# Keyless Local Stack as Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh self-hosted install answer with real RAG out of the box — a bundled GGUF LLM + a real (e5) embedder as the keyless default — with external API / GPU / air-gap as optional compose profiles.

**Architecture:** Add the already-existing local GGUF provider as the final *keyless* link in the MVP LLM auto-selection chain, switch the embedder default from `hashing` to a bundled e5 model, and guard against silent embedder/index dimension mismatch. Hardware detection is a separate advisory module (detect ≠ decide). Provisioning ships the default model in the image; lighter `api`/`gpu` profiles override via env only. Same engine reuses into a future cloud tier C unchanged.

**Tech Stack:** Python 3.13 (`py -3.13`), FastAPI, llama.cpp (`llama_cpp`), sentence-transformers, SQLite (MVP store), pytest with `tests/stubs/`, Docker Compose.

**Spec:** [docs/superpowers/specs/2026-06-08-keyless-local-stack-default-design.md](../specs/2026-06-08-keyless-local-stack-default-design.md)

---

## ✅ PRECONDITION — satisfied as of 2026-06-08

The local keyless stack this plan extends (`GgufEvalProvider`, `_build_gguf_provider`, the
`SentenceTransformerEmbedder`, and tests `tests/test_kb_llm_gguf.py` /
`tests/test_kb_embeddings_st.py`) is **already merged into `main`** via PRs #577 (`cea8514`)
and #579 (`0bca7ee`). Implementation happens on top of `main`. **Not blocked.**

Sanity-check before starting (all must succeed on a fresh `main`):

```bash
git checkout main && git pull
git grep -l "GgufEvalProvider" -- app/ tests/        # prints app/services/kb_llm.py
git ls-files | grep -E "test_kb_llm_gguf|test_kb_embeddings_st"   # both exist
```

Then branch the work:
```bash
git checkout -b feat/keyless-local-stack-default main
```

**Key fact that shapes Tasks 1–2:** the keyless LLM + ST embedder exist but are **opt-in**,
not the out-of-box default:
- GGUF fallback in `select_provider` fires **only** when `KB_LLM_LOCAL_FALLBACK` is truthy
  (`app/services/kb_llm.py`, the block before `return None`). Task 1 flips it to default-on.
- The ST embedder is used **only** when `KB_EMBEDDINGS_BACKEND=st` (`_build_from_env`,
  `app/services/kb_embeddings.py`); the implicit default is still `HashingEmbedder`. Task 2
  adds an implicit ST default before the hashing fallback.

---

## File structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `app/services/kb_llm.py` | add GGUF as keyless fallback in `select_provider` | Modify |
| `app/services/kb_embeddings.py` | e5/ST embedder as default in `_build_from_env` | Modify |
| `app/services/embedder_signature.py` | persist + verify embedder signature vs index | **Create** |
| `app/services/hardware_probe.py` | advisory RAM/CPU/CUDA probe (no decisions) | **Create** |
| `app/services/startup_preflight.py` | one-shot "what is active" startup log | **Create** |
| `scripts/dev_server_mvp.py` | call preflight on startup | Modify |
| `models/model_manifest.json` | add embedder + gpu entries; pin licenses/sha256 | Modify |
| `Dockerfile` | `BUNDLE_MODEL` build-arg → bake default model+embedder | Modify |
| `compose.yml` / `compose.api.yml` / `compose.gpu.yml` | profiles (env only) | Modify/Create |
| `.env.example` | document new env knobs | Modify |
| `tests/test_select_provider_keyless.py` | LLM fallback-chain tests | **Create** |
| `tests/test_embedder_default.py` | embedder default + e5 prefix tests | **Create** |
| `tests/test_embedder_signature.py` | signature guard tests | **Create** |
| `tests/test_hardware_probe.py` | probe advisory tests | **Create** |

---

## Task 0: Re-anchor to merged `main` (verification only, no code)

**Files:** none (reading only)

- [ ] **Step 1: Confirm precondition + branch**

Run the PRECONDITION block above. Confirm both greps succeed and you are on a fresh
`feat/keyless-local-stack-default` branch off `main`.

- [ ] **Step 2: Re-read the four integration seams and record current symbols**

Read and note the *current* (post-merge) shape — line numbers will differ from the spec:
```bash
py -3.13 -c "import app.services.kb_llm as m; print([n for n in dir(m) if 'gguf' in n.lower() or n=='select_provider'])"
py -3.13 -c "import app.services.kb_embeddings as m; print([n for n in dir(m)])"
py -3.13 -c "import app.retriever.e5 as m; print(m.e5_prefix)"
```
Expected: `select_provider`, a `_build_gguf_provider` (or equivalent) and `GgufEvalProvider`
exist in `kb_llm`; a sentence-transformers embedder class exists in `kb_embeddings`;
`e5_prefix(text, *, role, model, enabled)` exists in `app/retriever/e5.py`.

- [ ] **Step 3: Record the default GGUF path + manifest default**

```bash
py -3.13 -c "import app.services.kb_llm as m,inspect;print(inspect.getsource(m._build_gguf_provider))"
type models\model_manifest.json
```
Note the exact default model path string and the manifest `default` entry — Tasks 1 and 6
must use these exact values. If they disagree (e.g. path says qwen but manifest says
TinyLlama), fix the manifest in Task 6 to match the path string used by `_build_gguf_provider`.

- [ ] **Step 4: Note the embedder class name + constructor**

From Step 2 output, record the exact sentence-transformers embedder class name and its
`__init__` signature (e.g. `SentenceTransformerEmbedder(model_name=..., enable_e5_prefix=...)`).
Task 2 must reference this exact name. If the class is missing, **stop** — precondition unmet.

---

## Task 1: Make the keyless GGUF fallback default-on

**Files:**
- Modify: `app/services/kb_llm.py` (function `select_provider`)
- Test: `tests/test_select_provider_keyless.py` (create)

**Context:** The GGUF keyless link already exists in `select_provider`, but it is gated
behind an opt-in flag — currently:
```python
    if (_env("KB_LLM_LOCAL_FALLBACK", env) or "").strip().lower() in {"1", "true", "yes", "on"}:
        return _build_gguf_provider(env)
    return None
```
So out of the box (no env at all) GGUF is never tried → `None` → extractive. For the
self-hosted product we flip the default: try the local GGUF **unless explicitly disabled**
via `KB_LLM_LOCAL_FALLBACK=0`/`false`/`off`. The explicit `KB_LLM_PROVIDER=gguf` path and the
cloud-key priority stay unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_select_provider_keyless.py`:
```python
"""Keyless auto-selection: local GGUF is the DEFAULT fallback (opt-out, not opt-in)."""
from __future__ import annotations

import app.services.kb_llm as kb_llm


def test_gguf_used_by_default_when_no_keys(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: sentinel)
    assert kb_llm.select_provider(env={}) is sentinel  # no env at all → GGUF


def test_gguf_disabled_explicitly(monkeypatch):
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: object())
    assert kb_llm.select_provider(env={"KB_LLM_LOCAL_FALLBACK": "0"}) is None


def test_none_when_no_keys_and_no_model(monkeypatch):
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: None)
    assert kb_llm.select_provider(env={}) is None


def test_external_key_wins_over_gguf(monkeypatch):
    called = {"gguf": False}

    def _spy(env=None):
        called["gguf"] = True
        return object()

    monkeypatch.setattr(kb_llm, "_build_gguf_provider", _spy)
    provider = kb_llm.select_provider(env={"DEEPSEEK_API_KEY": "sk-test"})
    assert provider is not None and provider.name == "deepseek"
    assert called["gguf"] is False


def test_explicit_gguf_still_works(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: sentinel)
    assert kb_llm.select_provider(env={"KB_LLM_PROVIDER": "gguf"}) is sentinel
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/test_select_provider_keyless.py -v`
Expected: `test_gguf_used_by_default_when_no_keys` FAILS (current gate is opt-in, so an
empty env returns `None`). `test_gguf_disabled_explicitly` should pass already.

- [ ] **Step 3: Flip the gate to opt-out**

In `app/services/kb_llm.py`, in `select_provider`, replace the existing opt-in block:
```python
    if (_env("KB_LLM_LOCAL_FALLBACK", env) or "").strip().lower() in {"1", "true", "yes", "on"}:
        return _build_gguf_provider(env)

    return None
```
with default-on (opt-out) logic:
```python
    # Keyless default: try the bundled local GGUF so the product works out of the
    # box with no API key. Disable explicitly with KB_LLM_LOCAL_FALLBACK=0/false/off.
    if (_env("KB_LLM_LOCAL_FALLBACK", env) or "on").strip().lower() not in {"0", "false", "no", "off"}:
        gguf = _build_gguf_provider(env)
        if gguf is not None:
            return gguf

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_select_provider_keyless.py -v`
Expected: 5 passed.

- [ ] **Step 5: Guard against regressions**

Run: `py -3.13 -m pytest tests/test_kb_llm_gguf.py tests/test_llm_provider_factory.py -q`
Confirm with `echo $LASTEXITCODE` → `0` (piping hides the summary line).

- [ ] **Step 6: Commit**

```bash
git add app/services/kb_llm.py tests/test_select_provider_keyless.py
git commit -m "feat(llm): make bundled GGUF the keyless default (opt-out via KB_LLM_LOCAL_FALLBACK=0)"
```

---

## Task 2: e5/ST embedder as the default

**Files:**
- Modify: `app/services/kb_embeddings.py` (function `_build_from_env`)
- Test: `tests/test_embedder_default.py` (create)

**Context:** `_build_from_env` already supports `SentenceTransformerEmbedder` but **only** via
explicit `KB_EMBEDDINGS_BACKEND=st` (default model there is `BAAI/bge-m3`, e5 prefixing gated
on `VECTOR_E5_PREFIX`). When no backend is chosen it still falls through to `HashingEmbedder`.
We add an **implicit** ST default: when nothing is explicitly chosen and ST is available,
return a light `intfloat/multilingual-e5-small` embedder with e5 prefixing **on**. Hashing
stays as last resort; the explicit `KB_EMBEDDINGS_BACKEND=hash` path is preserved. Verified
ctor (Task 0): `SentenceTransformerEmbedder(model_name=..., e5_prefix_enabled=...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embedder_default.py`:
```python
"""Default embedder is the real ST/e5 model, not hashing, when available."""
from __future__ import annotations

import app.services.kb_embeddings as emb
from app.retriever.e5 import e5_prefix


def setup_function(_):
    emb.reset_embedder()


def teardown_function(_):
    emb.reset_embedder()


def test_default_is_st_when_available(monkeypatch):
    sentinel = emb.HashingEmbedder()  # any Embedder; identity is what we assert
    monkeypatch.setattr(emb, "_try_build_st_embedder", lambda env: sentinel, raising=False)
    chosen = emb._build_from_env(env={})
    assert chosen is sentinel


def test_falls_back_to_hash_when_st_unavailable(monkeypatch):
    monkeypatch.setattr(emb, "_try_build_st_embedder", lambda env: None, raising=False)
    chosen = emb._build_from_env(env={})
    assert isinstance(chosen, emb.HashingEmbedder)


def test_explicit_hash_skips_st(monkeypatch):
    called = {"st": False}

    def _spy(env):
        called["st"] = True
        return object()

    monkeypatch.setattr(emb, "_try_build_st_embedder", _spy, raising=False)
    chosen = emb._build_from_env(env={"KB_EMBEDDINGS_BACKEND": "hash"})
    assert isinstance(chosen, emb.HashingEmbedder)
    assert called["st"] is False


def test_e5_prefix_query_vs_passage():
    assert e5_prefix("foo", role="query", model="multilingual-e5-small", enabled=True) == "query: foo"
    assert e5_prefix("foo", role="passage", model="multilingual-e5-small", enabled=True) == "passage: foo"
    assert e5_prefix("foo", role="query", model="not-e5", enabled=True) == "foo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/test_embedder_default.py -v`
Expected: `test_default_is_st_when_available` FAILS (default is currently hashing). The
`e5_prefix` test passes (module already exists); the explicit-hash test may already pass.

- [ ] **Step 3: Add the ST default builder + wire it in**

`SentenceTransformerEmbedder` is already defined in this module (no new import needed).
Add a helper above `_build_from_env`:
```python
def _try_build_st_embedder(env: Mapping[str, str] | None) -> Optional[Embedder]:
    """Return the implicit-default ST e5 embedder, or None if unavailable.

    Unavailable = optional dependency missing OR weights not on disk. Never raises —
    absence simply means 'fall through to hash'. Light e5-small with prefixing on.
    """
    model_name = _env("ST_EMBED_MODEL", env) or "intfloat/multilingual-e5-small"
    try:
        candidate = SentenceTransformerEmbedder(model_name=model_name, e5_prefix_enabled=True)
    except Exception as exc:  # dependency or weights missing — advisory, not fatal
        LOGGER.info("ST embedder unavailable (%s); using fallback embedder", exc)
        return None
    record_embedder_backend("st")
    return candidate
```
Then in `_build_from_env`, **replace the final hashing fallback block** (the
`record_embedder_backend("hash"); return HashingEmbedder()` at the end) with:
```python
    if not explicit:
        st = _try_build_st_embedder(env)
        if st is not None:
            return st

    record_embedder_backend("hash")
    return HashingEmbedder()
```
Keep the existing `KB_API_KEY`-set warning above this block intact. Note: this makes the
default embedder e5-small (384 dim) — see Task 3, the signature guard catches any existing
hashing-built index so the switch can never silently corrupt search.

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_embedder_default.py -v`
Expected: all passed.

- [ ] **Step 5: Regression check the existing embedder tests**

Run: `py -3.13 -m pytest tests/test_kb_embeddings_st.py -q`
Expected: exit code 0.

- [ ] **Step 6: Commit**

```bash
git add app/services/kb_embeddings.py tests/test_embedder_default.py
git commit -m "feat(embeddings): default to bundled e5 embedder instead of hashing"
```

---

## Task 3: Embedder signature guard (migration safety)

**Files:**
- Create: `app/services/embedder_signature.py`
- Test: `tests/test_embedder_signature.py` (create)

**Context:** Switching the default hashing→e5 changes vector dimension. On an existing
index this silently corrupts search. We persist a signature (`name:dim`) and, on startup,
hard-stop on mismatch rather than auto-reindexing (a large corpus reindex could burn hours).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embedder_signature.py`:
```python
"""Embedder signature guard: hard-stop on index/embedder mismatch."""
from __future__ import annotations

import pytest

from app.services.embedder_signature import (
    EmbedderMismatchError,
    signature_for,
    verify_or_store,
)


class _FakeEmbedder:
    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dimension = dim


def test_signature_format():
    assert signature_for(_FakeEmbedder("e5-small", 384)) == "e5-small:384"


def test_fresh_index_stores_and_passes():
    store: dict[str, str] = {}
    verify_or_store(_FakeEmbedder("e5-small", 384), load=store.get,
                    save=lambda s: store.__setitem__("sig", s))
    assert store["sig"] == "e5-small:384"


def test_matching_signature_passes():
    store = {"sig": "e5-small:384"}
    verify_or_store(_FakeEmbedder("e5-small", 384), load=store.get,
                    save=lambda s: store.__setitem__("sig", s))  # no raise


def test_mismatch_raises_with_instructions():
    store = {"sig": "hash:256"}
    with pytest.raises(EmbedderMismatchError) as exc:
        verify_or_store(_FakeEmbedder("e5-small", 384), load=store.get,
                        save=lambda s: None)
    msg = str(exc.value)
    assert "reindex" in msg.lower()
    assert "hash:256" in msg and "e5-small:384" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/test_embedder_signature.py -v`
Expected: FAIL — `app.services.embedder_signature` does not exist.

- [ ] **Step 3: Implement the guard**

Create `app/services/embedder_signature.py`:
```python
"""Persist + verify the embedder signature against the existing index.

Mismatch is a hard, loud failure (never silent, never auto-reindex). The storage
hooks (load/save) are injected so this stays pure and testable; the MVP store wires
them to its SQLite meta table.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol


class _HasSignature(Protocol):
    name: str
    dimension: int


class EmbedderMismatchError(RuntimeError):
    """Raised when the active embedder disagrees with the indexed vectors."""


def signature_for(embedder: _HasSignature) -> str:
    return f"{embedder.name}:{int(embedder.dimension)}"


def verify_or_store(
    embedder: _HasSignature,
    *,
    load: Callable[[str], Optional[str]],
    save: Callable[[str], None],
) -> None:
    """Store the signature on a fresh index; raise on mismatch otherwise."""
    current = signature_for(embedder)
    stored = load("sig")
    if stored is None:
        save(current)
        return
    if stored != current:
        raise EmbedderMismatchError(
            f"Embedder/index mismatch: index was built with {stored!r} but the active "
            f"embedder is {current!r}. The vectors are not comparable. Either run "
            f"`kb-cli reindex --embedder {embedder.name}` to rebuild the index, or set "
            f"KB_EMBEDDINGS_BACKEND to the original backend to keep the existing index."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_embedder_signature.py -v`
Expected: all passed.

- [ ] **Step 5: Wire into MVP store startup (read kb_store first)**

Read the MVP store to find the schema-init/meta pattern:
```bash
py -3.13 -c "import app.services.kb_store as m; print(m.__file__)"
```
In the store's init (next to `_init_schema` / `_COLUMN_MIGRATIONS`), ensure a key/value
meta table exists (e.g. `kv_meta(key TEXT PRIMARY KEY, value TEXT)`), then call
`verify_or_store(get_embedder(), load=<select value where key=?>, save=<upsert>)`
after the embedder is resolved during app startup. Keep the SQL consistent with the
store's existing migration style (per project memory: store self-migrates, no Alembic for MVP).

- [ ] **Step 6: Commit**

```bash
git add app/services/embedder_signature.py tests/test_embedder_signature.py app/services/kb_store.py
git commit -m "feat(embeddings): hard-stop on embedder/index signature mismatch"
```

---

## Task 4: Hardware probe (advisory only)

**Files:**
- Create: `app/services/hardware_probe.py`
- Test: `tests/test_hardware_probe.py` (create)

**Context:** On unknown hardware (the agreed profile-C reality), warn loudly when RAM is too
small for the default model, but never block startup and never make the provider decision.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hardware_probe.py`:
```python
"""Hardware probe is advisory: it warns but never decides or raises."""
from __future__ import annotations

from app.services.hardware_probe import probe, ProbeResult


def test_enough_ram_no_warning():
    r = probe(total_ram_gb=16.0, cores=8, has_cuda=False, model_needs_gb=4.0)
    assert isinstance(r, ProbeResult)
    assert r.ram_warning is False
    assert r.advice == ""


def test_low_ram_warns_with_advice():
    r = probe(total_ram_gb=2.0, cores=2, has_cuda=False, model_needs_gb=4.0)
    assert r.ram_warning is True
    assert "api" in r.advice.lower()  # suggests the lighter api profile


def test_probe_never_raises_on_unknown():
    r = probe(total_ram_gb=None, cores=None, has_cuda=False, model_needs_gb=4.0)
    assert r.ram_warning is False  # unknown → no false alarm, no crash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/test_hardware_probe.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the probe**

Create `app/services/hardware_probe.py`:
```python
"""Advisory hardware probe. Detects, never decides. Pure core + thin OS wrapper."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    total_ram_gb: Optional[float]
    cores: Optional[int]
    has_cuda: bool
    ram_warning: bool
    advice: str


def probe(
    *,
    total_ram_gb: Optional[float],
    cores: Optional[int],
    has_cuda: bool,
    model_needs_gb: float,
) -> ProbeResult:
    """Pure decision core (injected facts) — easy to test deterministically."""
    ram_warning = total_ram_gb is not None and total_ram_gb < model_needs_gb
    advice = ""
    if ram_warning:
        advice = (
            f"Available RAM ~{total_ram_gb:.1f} GB is below the ~{model_needs_gb:.1f} GB "
            f"the bundled model needs. Use the lighter 'api' profile with an external key, "
            f"or a smaller model. Startup continues but inference may be slow or fail."
        )
        LOGGER.warning(advice)
    return ProbeResult(total_ram_gb, cores, has_cuda, ram_warning, advice)


def probe_system(model_needs_gb: float = 4.0) -> ProbeResult:
    """Thin OS wrapper: gather real facts, then call the pure core."""
    ram_gb: Optional[float] = None
    try:
        import psutil  # optional

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        ram_gb = None
    cores = os.cpu_count()
    has_cuda = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return probe(total_ram_gb=ram_gb, cores=cores, has_cuda=has_cuda,
                 model_needs_gb=model_needs_gb)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_hardware_probe.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/services/hardware_probe.py tests/test_hardware_probe.py
git commit -m "feat(ops): advisory hardware probe for low-RAM warning"
```

---

## Task 5: Startup preflight log

**Files:**
- Create: `app/services/startup_preflight.py`
- Modify: `scripts/dev_server_mvp.py` (call preflight in startup/lifespan)
- Test: extend `tests/test_hardware_probe.py` is wrong scope — create `tests/test_startup_preflight.py`

**Context:** One readable line at startup so the operator knows what actually loaded
(LLM provider, embedder, mode). Prevents "the AI is dumb" misdiagnosis.

- [ ] **Step 1: Write the failing test**

Create `tests/test_startup_preflight.py`:
```python
"""Preflight produces one readable status line and never raises."""
from __future__ import annotations

from app.services.startup_preflight import format_preflight


def test_format_includes_llm_embedder_mode():
    line = format_preflight(llm_name="gguf", llm_model="qwen2.5-3b",
                            embedder_name="e5-small", mode="bundled")
    assert "gguf" in line and "qwen2.5-3b" in line
    assert "e5-small" in line
    assert "bundled" in line.lower()


def test_format_handles_missing_llm():
    line = format_preflight(llm_name=None, llm_model=None,
                            embedder_name="hash", mode="degraded")
    assert "extractive" in line.lower() or "none" in line.lower()
    assert "hash" in line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest tests/test_startup_preflight.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement preflight**

Create `app/services/startup_preflight.py`:
```python
"""One-shot startup status line: what LLM/embedder/mode is actually active."""
from __future__ import annotations

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def format_preflight(
    *,
    llm_name: Optional[str],
    llm_model: Optional[str],
    embedder_name: str,
    mode: str,
) -> str:
    llm = f"{llm_name}({llm_model})" if llm_name else "none → extractive fallback"
    return f"KB.AI ready · LLM={llm} · Embedder={embedder_name} · Mode={mode}"


def log_preflight(*, llm_name, llm_model, embedder_name, mode) -> None:
    LOGGER.info(format_preflight(llm_name=llm_name, llm_model=llm_model,
                                 embedder_name=embedder_name, mode=mode))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest tests/test_startup_preflight.py -v`
Expected: all passed.

- [ ] **Step 5: Call preflight at MVP startup**

In `scripts/dev_server_mvp.py` startup (after provider + embedder are resolved), call
`probe_system()` (Task 4) then `log_preflight(...)` with the resolved provider name/model,
`get_embedder().name`, and a `mode` string derived from env
(`"bundled"`/`"api"`/`"gpu"`/`"airgap"`/`"degraded"`). Wrap in try/except that logs but
never blocks startup.

- [ ] **Step 6: Commit**

```bash
git add app/services/startup_preflight.py scripts/dev_server_mvp.py tests/test_startup_preflight.py
git commit -m "feat(ops): startup preflight line for active LLM/embedder/mode"
```

---

## Task 6: Model manifest + Dockerfile bundling

**Files:**
- Modify: `models/model_manifest.json`
- Modify: `Dockerfile`

**Context:** The default profile bakes the default GGUF + the e5 embedder into the image so
air-gapped/compliance installs work offline. Lighter `api` profile skips the bake.

- [ ] **Step 1: Pin the qwen target + add a gpu entry in the manifest**

The manifest already has `default` (TinyLlama, ~625 MB) and `qwen2.5-3b-instruct`
(filename `qwen2.5-3b-instruct-q4_k_m.gguf`, which is exactly the path
`_build_gguf_provider` expects). The product default LLM is **qwen2.5-3b-instruct** (RU-capable),
so the bundle must fetch that target — not `default`. In `models/model_manifest.json`:
fill the real `sha256` for `qwen2.5-3b-instruct` (currently `null` — required for commercial
distribution and for Task 3's integrity check) and add a `gpu` (7B) entry:
```json
  "gpu": {
    "model_id": "<7B GGUF id, e.g. Qwen/Qwen2.5-7B-Instruct-GGUF>",
    "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
    "sha256": "<fill real hash>",
    "license": "Apache-2.0",
    "url": "<resolve URL>",
    "description": "Qwen2.5-7B-Instruct Q4_K_M for the gpu profile."
  }
```
(The e5-small embedder is fetched by `sentence-transformers` at first use; no GGUF manifest
entry is needed for it. If you want it baked too, add it in the Dockerfile step below.)

- [ ] **Step 2: Add the BUNDLE_MODEL build-arg to the Dockerfile**

In `Dockerfile`, add near the top:
```dockerfile
ARG BUNDLE_MODEL=true
```
Add a conditional layer (after deps install) that, when `BUNDLE_MODEL=true`, fetches the
qwen default LLM and warms the e5-small embedder into the image:
```dockerfile
RUN if [ "$BUNDLE_MODEL" = "true" ]; then \
      py -3 -m scripts.download_model --target qwen2.5-3b-instruct --output ./models/qwen2.5-3b-instruct-q4_k_m.gguf && \
      py -3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" ; \
    fi
```
(Confirm the exact `download_model.py` flags from Task 0 — it exposes `--target` and `--output`.)

- [ ] **Step 3: Verify the manifest parses + download script accepts the target**

Run:
```bash
py -3.13 -c "import json;print(list(json.load(open('models/model_manifest.json')).keys()))"
py -3.13 -m pytest tests/scripts/test_download_model.py -q
```
Expected: keys include `default`, `qwen2.5-3b-instruct`, `gpu`; download_model tests exit 0.

- [ ] **Step 4: Commit**

```bash
git add models/model_manifest.json Dockerfile
git commit -m "feat(provisioning): bundle default GGUF + e5 embedder via BUNDLE_MODEL arg"
```

---

## Task 7: Compose profiles + env docs

**Files:**
- Modify: `compose.yml`
- Create: `compose.api.yml`, `compose.gpu.yml`
- Modify: `.env.example`
- Test: extend `tests/test_env_example.py`

**Context:** Profiles are env composition only — zero code. Default = bundled. `api` = light
image + external key. `gpu` = 7B + CUDA. `airgap` = explicit alias of default for CISO docs.

- [ ] **Step 1: Write the failing profile test**

Extend `tests/test_env_example.py` (or create `tests/test_compose_profiles.py`):
```python
"""Each compose profile yields a coherent, non-contradictory env set."""
from __future__ import annotations

import pathlib

import yaml


def _services_env(path: str) -> dict:
    data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    svc = next(iter(data["services"].values()))
    env = svc.get("environment", {})
    if isinstance(env, list):
        env = dict(e.split("=", 1) for e in env)
    return env


def test_api_profile_disables_local_gguf():
    env = _services_env("compose.api.yml")
    assert str(env.get("KB_LLM_LOCAL_FALLBACK", "")).lower() in {"0", "false", "no", "off"}


def test_gpu_profile_requests_gpu_layers():
    env = _services_env("compose.gpu.yml")
    assert int(env.get("LLM_GPU_LAYERS", "0")) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest tests/test_compose_profiles.py -v`
Expected: FAIL — override files do not exist.

- [ ] **Step 3: Create the override files**

The serving service in `compose.yml` is **`kb_api`** (confirmed Task 0; other services are
`qdrant`, `kb_web`, `web`). Override files target `kb_api`.

`compose.api.yml`:
```yaml
services:
  kb_api:
    environment:
      KB_LLM_LOCAL_FALLBACK: "0"   # disable local GGUF; use the external key below
      DEEPSEEK_API_KEY: "${DEEPSEEK_API_KEY:?set a provider key for the api profile}"
      KB_EMBEDDINGS_BACKEND: "api"
```
`compose.gpu.yml`:
```yaml
services:
  kb_api:
    environment:
      KB_LLM_PROVIDER: "gguf"
      KB_LLM_GGUF_PATH: "./models/qwen2.5-7b-instruct-q4_k_m.gguf"
      LLM_GPU_LAYERS: "35"
```

- [ ] **Step 4: Document env knobs in `.env.example`**

Add documented entries: `ST_EMBED_MODEL`, `VECTOR_E5_PREFIX`, `KB_LLM_GGUF_PATH`,
`KB_LLM_LOCAL_FALLBACK`, `LLM_GPU_LAYERS`, and a comment block describing the four profiles
(default/api/gpu/airgap).

- [ ] **Step 5: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/test_compose_profiles.py tests/test_env_example.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add compose.yml compose.api.yml compose.gpu.yml .env.example tests/test_compose_profiles.py
git commit -m "feat(deploy): api/gpu/airgap compose profiles over the bundled default"
```

---

## Final verification (whole-feature)

- [ ] **Step 1: Full suite, CI-mirrored**

Run: `py -3.13 -m pytest -q --ignore=backend`
Confirm with `echo $LASTEXITCODE` → `0` (piping drops the summary line — rely on exit code).

- [ ] **Step 2: Lint + format + types on touched files**

```bash
py -3.13 -m ruff check app/services/ tests/
py -3.13 -m black --check app/services/ tests/
py -3.13 -m mypy app/services/kb_llm.py app/services/kb_embeddings.py app/services/embedder_signature.py app/services/hardware_probe.py app/services/startup_preflight.py
```
Expected: ruff/black clean; mypy adds **no new** errors on touched lines (baseline is
pre-existing; judge by new errors only).

- [ ] **Step 3: Manual smoke — keyless default actually answers**

With no provider keys and the default model present, start the MVP server
(`uvicorn scripts.dev_server_mvp:app --port 8001`), confirm the preflight log shows
`LLM=gguf(...) · Embedder=...e5... · Mode=bundled`, ingest one document, ask one question,
and confirm a cited answer comes back (not the extractive fallback).

---

## Self-review notes (author)

- **Spec §3 (architecture):** Tasks 1+2 (keyless LLM + e5 default). ✓
- **Spec §4 (6 components):** Task 1 (LLM), Task 2 (embedder), Task 6 (provisioning), Task 7
  (profiles), Task 4 (hardware_probe), Task 5 (preflight). ✓
- **Spec §6 (errors/migration):** Task 3 (signature guard, hard-stop), Task 4 (low-RAM warn),
  Task 6 (sha256 validation via manifest), Task 7 (api profile fails loud without key). ✓
- **Spec §7 (testing + app.state cache leak):** every task is TDD; `reset_embedder()` used in
  `setup/teardown` (Task 2). Note for executor: also reset any cached LLM provider in
  fixtures that select providers, per memory `repo_v1_testclient_override_bypass`.
- **Soft spots — resolved against `main` @ 54f2b79 (2026-06-08):**
  ST class = `SentenceTransformerEmbedder(model_name=, e5_prefix_enabled=)`;
  default GGUF path = `./models/qwen2.5-3b-instruct-q4_k_m.gguf` (manifest target
  `qwen2.5-3b-instruct`); `download_model.py` flags = `--target` / `--output`;
  compose serving service = `kb_api`; env knobs = `KB_LLM_LOCAL_FALLBACK` (opt-out),
  `ST_EMBED_MODEL`, `VECTOR_E5_PREFIX`. Task 0 re-checks these in case `main` moves before
  execution.
