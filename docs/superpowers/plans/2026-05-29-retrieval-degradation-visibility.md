# Retrieval Degradation Visibility — Implementation Plan (PR1 "ops-loud")

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make silent retrieval degradations visible to operators by introducing a unified `RetrievalReport` contract that both retrieval paths populate, surfaced in both health endpoints and a Prometheus gauge — with zero client-facing change.

**Architecture:** A new dependency-light module `app/observability/retrieval_health.py` owns the contract (`RetrievalReason`, `RetrievalSeverity`, `RetrievalReport`), a thread-safe registry (reason → timestamp), a per-request `ContextVar`, and a self-contained Prometheus gauge. `kb_store.search()` (MVP/SQLite) and `vectorstore.search()` (mature/Qdrant) call `report(...)` on every query. `/api/kb/health` and `/ops/health/dependencies` read `snapshot()`. **Deviation from spec §6:** the gauge lives in `retrieval_health.py` (defensive `prometheus_client` import), NOT in `metrics.py`, so importing it from the light MVP path does not drag `metrics.py`'s `sqlalchemy` dependency in (honours spec decision D1).

**Tech Stack:** Python 3.12, `prometheus_client`, FastAPI, SQLite (stdlib `sqlite3`), pytest. Windows: run via `py -3`.

**Scope:** PR1 only (spec §10). PR2 ("customer-loud": `/ask` field, `/ask/stream` `meta`, v1 chat, UI banner, i18n) is a **separate plan** authored after PR1 merges, because it depends on this contract being final and touches `app/api/v1/chat.py`, `data/www/index.html`, `data/www/i18n/ru.json`.

**Reference spec:** `docs/superpowers/specs/2026-05-29-retrieval-degradation-visibility-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `app/observability/retrieval_health.py` | Contract + registry + ContextVar + gauge + `report`/`current_report`/`snapshot`/`reset` | Create |
| `app/services/vectorstore.py` | Report `VECTOR_BACKEND_DOWN` on grep fallback, healthy on success | Modify |
| `app/services/kb_store.py` | Detect hashing / dim-mismatch / truncation in `search()` | Modify |
| `app/api/kb_mvp.py` | Add `retrieval` block + `degraded` to `/api/kb/health` | Modify |
| `app/api/v1/ops.py` | Merge `snapshot()` into `/ops/health/dependencies` | Modify |
| `docs/observability_slo.md` | Document the `kb_retrieval_degraded` alert rule | Modify |
| `tests/test_retrieval_health.py` | Unit tests for the module | Create |
| `tests/test_services_vectorstore.py` | Extend with fallback-reporting tests | Modify |
| `tests/test_kb_store_retrieval_health.py` | kb_store detection tests | Create |
| `tests/test_kb_mvp_health_retrieval.py` | `/api/kb/health` surfacing test | Create |
| `tests/test_api_v1_ops_retrieval.py` | `/ops/health/dependencies` surfacing test | Create |

---

## Task 1: `retrieval_health` contract, registry, and gauge

**Files:**
- Create: `app/observability/retrieval_health.py`
- Test: `tests/test_retrieval_health.py`

- [ ] **Step 1: Write the failing test for the contract types**

Create `tests/test_retrieval_health.py`:

```python
"""Tests for the unified retrieval-degradation contract."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

import app.observability.retrieval_health as rh
from app.observability.retrieval_health import (
    RetrievalReason,
    RetrievalReport,
    RetrievalSeverity,
)


def test_report_severity_critical_dominates_warning():
    rep = RetrievalReport(
        source="sqlite",
        reasons=(RetrievalReason.SEARCH_TRUNCATED, RetrievalReason.HASHING_EMBEDDER),
    )
    assert rep.degraded is True
    assert rep.severity is RetrievalSeverity.CRITICAL


def test_clean_report_is_not_degraded():
    rep = RetrievalReport(source="vector")
    assert rep.degraded is False
    assert rep.severity is RetrievalSeverity.OK
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.observability.retrieval_health'`

- [ ] **Step 3: Write the contract types**

Create `app/observability/retrieval_health.py`:

```python
"""Unified retrieval-degradation contract shared by both retrieval paths.

Dependency-light by design (only an optional ``prometheus_client`` import):
both the heavy ``/api/v1`` vector path and the light MVP ``/api/kb`` SQLite
path import this without coupling either to the other's dependencies.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum


class RetrievalSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"      # results may be incomplete
    CRITICAL = "critical"    # results likely useless or empty


class RetrievalReason(str, Enum):
    HEALTHY = "healthy"
    VECTOR_BACKEND_DOWN = "vector_backend_down"        # Qdrant/FAISS failed -> grep fallback
    HASHING_EMBEDDER = "hashing_embedder"              # near-random matches
    EMBEDDING_DIM_MISMATCH = "embedding_dim_mismatch"  # index incoherent with active embedder
    SEARCH_TRUNCATED = "search_truncated"              # hard-limit hit


_SEVERITY: dict[RetrievalReason, RetrievalSeverity] = {
    RetrievalReason.VECTOR_BACKEND_DOWN: RetrievalSeverity.CRITICAL,
    RetrievalReason.HASHING_EMBEDDER: RetrievalSeverity.CRITICAL,
    RetrievalReason.EMBEDDING_DIM_MISMATCH: RetrievalSeverity.CRITICAL,
    RetrievalReason.SEARCH_TRUNCATED: RetrievalSeverity.WARNING,
}


def severity_of(reasons) -> RetrievalSeverity:
    sev = [_SEVERITY[r] for r in reasons if r in _SEVERITY]
    if RetrievalSeverity.CRITICAL in sev:
        return RetrievalSeverity.CRITICAL
    if RetrievalSeverity.WARNING in sev:
        return RetrievalSeverity.WARNING
    return RetrievalSeverity.OK


@dataclass(frozen=True)
class RetrievalReport:
    source: str                                   # "sqlite" | "vector" | "fallback"
    reasons: tuple[RetrievalReason, ...] = ()
    detail: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.reasons)

    @property
    def severity(self) -> RetrievalSeverity:
        return severity_of(self.reasons)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/observability/retrieval_health.py tests/test_retrieval_health.py
git commit -m "feat(observability): add RetrievalReport degradation contract"
```

- [ ] **Step 6: Write the failing test for registry + ContextVar + snapshot**

Append to `tests/test_retrieval_health.py`:

```python
def test_report_then_snapshot_lists_active_reason():
    rh.reset()
    rh.report(
        RetrievalReport(
            source="fallback",
            reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,),
            detail="boom",
        )
    )
    snap = rh.snapshot()
    assert snap["degraded"] is True
    assert snap["severity"] == "critical"
    assert snap["reasons"][0]["reason"] == "vector_backend_down"
    assert snap["reasons"][0]["detail"] == "boom"


def test_clean_run_clears_that_sources_reasons():
    rh.reset()
    rh.report(
        RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,))
    )
    rh.report(RetrievalReport(source="vector"))  # clean run
    assert rh.snapshot()["degraded"] is False


def test_current_report_returns_last_report_for_this_context():
    rh.reset()
    rep = RetrievalReport(source="sqlite", reasons=(RetrievalReason.HASHING_EMBEDDER,))
    rh.report(rep)
    assert rh.current_report() is rep


def test_ttl_backstop_drops_stale_reason(monkeypatch):
    rh.reset()
    clock = {"t": 1000.0}
    monkeypatch.setattr(rh.time, "monotonic", lambda: clock["t"])
    rh.report(
        RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,))
    )
    assert rh.snapshot(ttl_seconds=300.0)["degraded"] is True
    clock["t"] = 1000.0 + 301.0
    assert rh.snapshot(ttl_seconds=300.0)["degraded"] is False


def test_snapshot_includes_extra_active_probes():
    rh.reset()
    snap = rh.snapshot(extra=((RetrievalReason.HASHING_EMBEDDER, "embedder=hash"),))
    assert snap["degraded"] is True
    assert snap["reasons"][0]["reason"] == "hashing_embedder"
```

- [ ] **Step 7: Run the new tests to verify they fail**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: FAIL — `AttributeError: module 'app.observability.retrieval_health' has no attribute 'reset'`

- [ ] **Step 8: Implement the registry, ContextVar, and snapshot**

Append to `app/observability/retrieval_health.py`:

```python
# Which reasons each search source is authoritative for, so a clean run
# from that source clears them immediately (TTL is only a backstop).
_SOURCE_REASONS: dict[str, set[RetrievalReason]] = {
    "sqlite": {
        RetrievalReason.HASHING_EMBEDDER,
        RetrievalReason.EMBEDDING_DIM_MISMATCH,
        RetrievalReason.SEARCH_TRUNCATED,
    },
    "vector": {RetrievalReason.VECTOR_BACKEND_DOWN, RetrievalReason.HASHING_EMBEDDER},
    "fallback": {RetrievalReason.VECTOR_BACKEND_DOWN, RetrievalReason.HASHING_EMBEDDER},
}

_DEFAULT_TTL = 300.0
_LOCK = threading.Lock()
_REGISTRY: dict[RetrievalReason, tuple[float, str]] = {}  # reason -> (monotonic_ts, detail)
_CURRENT: ContextVar[RetrievalReport | None] = ContextVar("retrieval_report", default=None)


def report(rep: RetrievalReport) -> None:
    """Record *rep* for health/metrics and expose it to this request context."""

    _CURRENT.set(rep)
    active = set(rep.reasons)
    governed = _SOURCE_REASONS.get(rep.source, set())
    now = time.monotonic()
    with _LOCK:
        for reason in active:
            _REGISTRY[reason] = (now, rep.detail)
        for reason in governed - active:
            _REGISTRY.pop(reason, None)


def current_report() -> RetrievalReport | None:
    """Return the report recorded earlier in this request context, if any."""

    return _CURRENT.get()


def snapshot(ttl_seconds: float = _DEFAULT_TTL, extra: tuple = ()) -> dict:
    """Current degradations within *ttl_seconds*, merged with active probes.

    *extra* is an iterable of ``(RetrievalReason, detail)`` from cheap
    config-level probes a health endpoint runs (e.g. embedder == hash).
    """

    now = time.monotonic()
    reasons: list[dict] = []
    seen: set[RetrievalReason] = set()
    with _LOCK:
        items = list(_REGISTRY.items())
    for reason, (ts, detail) in items:
        if now - ts <= ttl_seconds:
            reasons.append(
                {
                    "reason": reason.value,
                    "severity": _SEVERITY[reason].value,
                    "detail": detail,
                    "age_s": round(now - ts, 1),
                }
            )
            seen.add(reason)
    for reason, detail in extra:
        if reason not in seen:
            reasons.append(
                {
                    "reason": reason.value,
                    "severity": _SEVERITY[reason].value,
                    "detail": detail,
                    "age_s": 0.0,
                }
            )
            seen.add(reason)
    return {"degraded": bool(reasons), "severity": severity_of(seen).value, "reasons": reasons}


def reset() -> None:
    """Clear all recorded state (test helper)."""

    with _LOCK:
        _REGISTRY.clear()
    _CURRENT.set(None)
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: PASS (7 passed)

- [ ] **Step 10: Commit**

```bash
git add app/observability/retrieval_health.py tests/test_retrieval_health.py
git commit -m "feat(observability): add retrieval-health registry and snapshot"
```

- [ ] **Step 11: Write the failing test for the gauge**

Append to `tests/test_retrieval_health.py`:

```python
def test_gauge_set_on_report_and_cleared_on_clean_run():
    rh.reset()
    rh.report(
        RetrievalReport(source="vector", reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,))
    )
    assert (
        REGISTRY.get_sample_value(
            "kb_retrieval_degraded",
            {"reason": "vector_backend_down", "severity": "critical"},
        )
        == 1.0
    )
    rh.report(RetrievalReport(source="vector"))  # clean run clears governed reason
    assert (
        REGISTRY.get_sample_value(
            "kb_retrieval_degraded",
            {"reason": "vector_backend_down", "severity": "critical"},
        )
        == 0.0
    )
```

- [ ] **Step 12: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_retrieval_health.py::test_gauge_set_on_report_and_cleared_on_clean_run -q`
Expected: FAIL — `assert None == 1.0` (gauge does not exist yet)

- [ ] **Step 13: Add the gauge and wire it into `report()`**

In `app/observability/retrieval_health.py`, add the gauge near the top (after the imports, before `RetrievalSeverity`):

```python
try:  # pragma: no cover - prometheus_client is optional in minimal MVP envs
    from prometheus_client import Gauge

    _RETRIEVAL_DEGRADED = Gauge(
        "kb_retrieval_degraded",
        "Whether retrieval is currently degraded (1) per reason.",
        labelnames=("reason", "severity"),
    )
except Exception:  # pragma: no cover - gauge becomes a no-op when unavailable
    _RETRIEVAL_DEGRADED = None


def _set_gauge(reason: "RetrievalReason", active: bool) -> None:
    if _RETRIEVAL_DEGRADED is None:
        return
    severity = _SEVERITY.get(reason, RetrievalSeverity.WARNING).value
    _RETRIEVAL_DEGRADED.labels(reason=reason.value, severity=severity).set(
        1.0 if active else 0.0
    )
```

Then, inside `report()`, after the `with _LOCK:` block, add:

```python
    for reason in active:
        _set_gauge(reason, True)
    for reason in governed - active:
        _set_gauge(reason, False)
```

(`_set_gauge` references `_SEVERITY` and `RetrievalSeverity`, which are defined below it at module load time — both are resolved when `_set_gauge` is *called*, not defined, so ordering is fine.)

- [ ] **Step 14: Run the full module test to verify it passes**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: PASS (8 passed)

- [ ] **Step 15: Commit**

```bash
git add app/observability/retrieval_health.py tests/test_retrieval_health.py
git commit -m "feat(observability): expose kb_retrieval_degraded gauge"
```

**Final `app/observability/retrieval_health.py` for reference** (the assembled module after Steps 3, 8, 13):

```python
"""Unified retrieval-degradation contract shared by both retrieval paths."""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum

try:  # pragma: no cover - prometheus_client is optional in minimal MVP envs
    from prometheus_client import Gauge

    _RETRIEVAL_DEGRADED = Gauge(
        "kb_retrieval_degraded",
        "Whether retrieval is currently degraded (1) per reason.",
        labelnames=("reason", "severity"),
    )
except Exception:  # pragma: no cover
    _RETRIEVAL_DEGRADED = None


class RetrievalSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class RetrievalReason(str, Enum):
    HEALTHY = "healthy"
    VECTOR_BACKEND_DOWN = "vector_backend_down"
    HASHING_EMBEDDER = "hashing_embedder"
    EMBEDDING_DIM_MISMATCH = "embedding_dim_mismatch"
    SEARCH_TRUNCATED = "search_truncated"


_SEVERITY: dict[RetrievalReason, RetrievalSeverity] = {
    RetrievalReason.VECTOR_BACKEND_DOWN: RetrievalSeverity.CRITICAL,
    RetrievalReason.HASHING_EMBEDDER: RetrievalSeverity.CRITICAL,
    RetrievalReason.EMBEDDING_DIM_MISMATCH: RetrievalSeverity.CRITICAL,
    RetrievalReason.SEARCH_TRUNCATED: RetrievalSeverity.WARNING,
}

_SOURCE_REASONS: dict[str, set[RetrievalReason]] = {
    "sqlite": {
        RetrievalReason.HASHING_EMBEDDER,
        RetrievalReason.EMBEDDING_DIM_MISMATCH,
        RetrievalReason.SEARCH_TRUNCATED,
    },
    "vector": {RetrievalReason.VECTOR_BACKEND_DOWN, RetrievalReason.HASHING_EMBEDDER},
    "fallback": {RetrievalReason.VECTOR_BACKEND_DOWN, RetrievalReason.HASHING_EMBEDDER},
}

_DEFAULT_TTL = 300.0
_LOCK = threading.Lock()
_REGISTRY: dict[RetrievalReason, tuple[float, str]] = {}
_CURRENT: ContextVar[RetrievalReport | None] = ContextVar("retrieval_report", default=None)


def severity_of(reasons) -> RetrievalSeverity:
    sev = [_SEVERITY[r] for r in reasons if r in _SEVERITY]
    if RetrievalSeverity.CRITICAL in sev:
        return RetrievalSeverity.CRITICAL
    if RetrievalSeverity.WARNING in sev:
        return RetrievalSeverity.WARNING
    return RetrievalSeverity.OK


def _set_gauge(reason: "RetrievalReason", active: bool) -> None:
    if _RETRIEVAL_DEGRADED is None:
        return
    severity = _SEVERITY.get(reason, RetrievalSeverity.WARNING).value
    _RETRIEVAL_DEGRADED.labels(reason=reason.value, severity=severity).set(
        1.0 if active else 0.0
    )


@dataclass(frozen=True)
class RetrievalReport:
    source: str
    reasons: tuple[RetrievalReason, ...] = ()
    detail: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.reasons)

    @property
    def severity(self) -> RetrievalSeverity:
        return severity_of(self.reasons)


def report(rep: RetrievalReport) -> None:
    _CURRENT.set(rep)
    active = set(rep.reasons)
    governed = _SOURCE_REASONS.get(rep.source, set())
    now = time.monotonic()
    with _LOCK:
        for reason in active:
            _REGISTRY[reason] = (now, rep.detail)
        for reason in governed - active:
            _REGISTRY.pop(reason, None)
    for reason in active:
        _set_gauge(reason, True)
    for reason in governed - active:
        _set_gauge(reason, False)


def current_report() -> RetrievalReport | None:
    return _CURRENT.get()


def snapshot(ttl_seconds: float = _DEFAULT_TTL, extra: tuple = ()) -> dict:
    now = time.monotonic()
    reasons: list[dict] = []
    seen: set[RetrievalReason] = set()
    with _LOCK:
        items = list(_REGISTRY.items())
    for reason, (ts, detail) in items:
        if now - ts <= ttl_seconds:
            reasons.append(
                {
                    "reason": reason.value,
                    "severity": _SEVERITY[reason].value,
                    "detail": detail,
                    "age_s": round(now - ts, 1),
                }
            )
            seen.add(reason)
    for reason, detail in extra:
        if reason not in seen:
            reasons.append(
                {
                    "reason": reason.value,
                    "severity": _SEVERITY[reason].value,
                    "detail": detail,
                    "age_s": 0.0,
                }
            )
            seen.add(reason)
    return {"degraded": bool(reasons), "severity": severity_of(seen).value, "reasons": reasons}


def reset() -> None:
    with _LOCK:
        _REGISTRY.clear()
    _CURRENT.set(None)
```

---

## Task 2: Report `VECTOR_BACKEND_DOWN` from the mature path

**Files:**
- Modify: `app/services/vectorstore.py` (imports; `search()` success path ~line 244; fallback `except` ~line 230)
- Test: `tests/test_services_vectorstore.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_services_vectorstore.py` — a new import at the top, an autouse reset fixture, and two tests:

```python
import app.observability.retrieval_health as retrieval_health
```

```python
@pytest.fixture(autouse=True)
def _reset_retrieval_health():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def test_search_reports_vector_backend_down_on_fallback(monkeypatch):
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", ExplodingVectorStore())
    vectorstore.index_chunks([{"id": 1, "text": "beta", "tenant_id": "t1"}])

    vectorstore.search("beta", top_k=1, tenant_id="t1")

    rep = retrieval_health.current_report()
    assert rep is not None
    assert retrieval_health.RetrievalReason.VECTOR_BACKEND_DOWN in rep.reasons
    assert retrieval_health.snapshot()["severity"] == "critical"


def test_search_reports_healthy_on_vector_success(monkeypatch):
    dummy = DummyVectorStore()
    dummy.results = [{"text": "hit"}]
    monkeypatch.setattr(vectorstore, "_VECTOR_STORE", dummy)

    vectorstore.search("anything", top_k=5, tenant_id="t1")

    rep = retrieval_health.current_report()
    assert rep is not None
    assert rep.source == "vector"
    assert rep.degraded is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_services_vectorstore.py -q`
Expected: FAIL — `assert None is not None` (no report recorded yet)

- [ ] **Step 3: Add the import**

In `app/services/vectorstore.py`, after the existing `from app.observability.metrics import ...` line (line 23), add:

```python
from app.observability import retrieval_health
```

- [ ] **Step 4: Report on the success path**

In `search()`, replace the success tail (currently):

```python
    record_search_operation("vector", "success", time.perf_counter() - start, len(hits))
    return hits
```

with:

```python
    retrieval_health.report(retrieval_health.RetrievalReport(source="vector"))
    record_search_operation("vector", "success", time.perf_counter() - start, len(hits))
    return hits
```

- [ ] **Step 5: Report on the fallback path**

In `search()`, inside the `except _VECTOR_ERRORS as exc:` block, immediately before `fallback_hits = _search_fallback(...)`, add:

```python
        retrieval_health.report(
            retrieval_health.RetrievalReport(
                source="fallback",
                reasons=(retrieval_health.RetrievalReason.VECTOR_BACKEND_DOWN,),
                detail=str(exc),
            )
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_services_vectorstore.py -q`
Expected: PASS (all tests, including the pre-existing ones)

- [ ] **Step 7: Commit**

```bash
git add app/services/vectorstore.py tests/test_services_vectorstore.py
git commit -m "feat(retriever): report vector-backend-down on grep fallback"
```

---

## Task 3: Detect hashing / dim-mismatch / truncation in `kb_store.search()`

**Files:**
- Modify: `app/services/kb_store.py` (import; `search()` at line 472)
- Test: `tests/test_kb_store_retrieval_health.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kb_store_retrieval_health.py`:

```python
"""kb_store.search() raises the right retrieval-degradation reasons."""

from __future__ import annotations

import pytest

import app.observability.retrieval_health as retrieval_health
import app.services.kb_store as kb_store
from app.services.kb_store import KnowledgeBaseStore


class _StubEmbedder:
    def __init__(self, dim: int, name: str = "stub") -> None:
        self.name = name
        self.dimension = dim
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.1] * self._dim


@pytest.fixture(autouse=True)
def _reset():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def _store(tmp_path, embedder):
    return KnowledgeBaseStore(tmp_path / "kb.sqlite3", embedder=embedder)


def test_hashing_embedder_is_reported(tmp_path):
    store = _store(tmp_path, _StubEmbedder(dim=8, name="hash"))
    store.add_document("Doc", "alpha beta gamma", source="text")

    store.search("beta", top_k=3)

    rep = retrieval_health.current_report()
    assert retrieval_health.RetrievalReason.HASHING_EMBEDDER in rep.reasons


def test_dim_mismatch_is_reported(tmp_path):
    store = _store(tmp_path, _StubEmbedder(dim=8, name="real"))
    store.add_document("Doc", "alpha beta gamma", source="text")
    store._embedder = _StubEmbedder(dim=16, name="real")  # swap without reindex

    hits = store.search("beta", top_k=3)

    assert hits == []
    rep = retrieval_health.current_report()
    assert retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH in rep.reasons


def test_truncation_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_store, "_search_hard_limit", lambda: 2)
    store = _store(tmp_path, _StubEmbedder(dim=8, name="real"))
    store.add_document("D1", "alpha", source="text")
    store.add_document("D2", "beta", source="text")
    store.add_document("D3", "gamma", source="text")

    store.search("alpha", top_k=2)

    rep = retrieval_health.current_report()
    assert retrieval_health.RetrievalReason.SEARCH_TRUNCATED in rep.reasons


def test_clean_search_is_not_degraded(tmp_path):
    store = _store(tmp_path, _StubEmbedder(dim=8, name="real"))
    store.add_document("Doc", "alpha beta gamma", source="text")

    store.search("beta", top_k=3)

    rep = retrieval_health.current_report()
    assert rep is not None
    assert rep.source == "sqlite"
    assert rep.degraded is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_kb_store_retrieval_health.py -q`
Expected: FAIL — `AttributeError: 'NoneType' object has no attribute 'reasons'` (no report recorded yet)

- [ ] **Step 3: Add the import**

In `app/services/kb_store.py`, add to the imports block near the top:

```python
from app.observability import retrieval_health
```

- [ ] **Step 4: Add detection inside `search()`**

In `app/services/kb_store.py`, replace the block from `hard_limit = _search_hard_limit()` through the existing hard-limit `LOGGER.warning(...)` call (lines 485–501) with:

```python
        hard_limit = _search_hard_limit()
        reasons: list[retrieval_health.RetrievalReason] = []
        detail = ""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.document_id, d.title, c.chunk_index, c.text, c.embedding, c.dim,
                       d.source, d.filename, c.page_number, d.has_original_file
                FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id
                WHERE c.dim = ?
                LIMIT ?
                """,
                (q_dim, hard_limit),
            ).fetchall()
            if not rows:
                has_chunks = conn.execute(
                    "SELECT EXISTS(SELECT 1 FROM kb_chunks)"
                ).fetchone()[0]
                if has_chunks:
                    dims = [
                        str(r[0])
                        for r in conn.execute(
                            "SELECT DISTINCT dim FROM kb_chunks LIMIT 3"
                        ).fetchall()
                    ]
                    reasons.append(retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH)
                    detail = f"query dim {q_dim} not in stored dims {{{', '.join(dims)}}}"
        if len(rows) >= hard_limit:
            LOGGER.warning(
                "kb_store.search hit hard limit (%d chunks). Consider Qdrant for large corpora.",
                hard_limit,
            )
            reasons.append(retrieval_health.RetrievalReason.SEARCH_TRUNCATED)
            detail = detail or f"scan capped at {hard_limit} chunks"
        if getattr(self._embedder, "name", None) == HASHING_EMBEDDER_NAME:
            reasons.append(retrieval_health.RetrievalReason.HASHING_EMBEDDER)
            detail = detail or "embedder=hash (near-random semantic matches)"
        retrieval_health.report(
            retrieval_health.RetrievalReport(
                source="sqlite", reasons=tuple(reasons), detail=detail
            )
        )
```

The scoring loop that follows (`scored: List[...]` onward) is unchanged — it still reads `rows`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_kb_store_retrieval_health.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Run the existing kb_store tests to confirm no regression**

Run: `py -3 -m pytest tests/test_kb_store_pages.py tests/test_kb_mvp_search_response.py tests/test_npa_retrieval_golden.py -q`
Expected: PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add app/services/kb_store.py tests/test_kb_store_retrieval_health.py
git commit -m "feat(kb-store): detect hashing, dim-mismatch and truncation degradations"
```

---

## Task 4: Surface degradation in `/api/kb/health`

**Files:**
- Modify: `app/api/kb_mvp.py` (import; `health()` at line 598)
- Test: `tests/test_kb_mvp_health_retrieval.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_mvp_health_retrieval.py`:

```python
"""/api/kb/health surfaces retrieval degradation without breaking liveness."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.observability.retrieval_health as retrieval_health
import app.services.kb_embeddings as kb_embeddings
from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_with_hashing_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("KB_EMBEDDINGS_BACKEND", raising=False)
    kb_embeddings.reset_embedder()  # force re-resolution to the hashing default
    retrieval_health.reset()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    store.add_document("doc1", text="alpha " * 50)
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app


def test_health_reports_hashing_embedder_as_critical(app_with_hashing_store):
    client = TestClient(app_with_hashing_store)

    data = client.get("/api/kb/health").json()

    assert data["status"] == "ok"  # liveness probes must keep working
    assert data["degraded"] is True
    assert data["retrieval"]["severity"] == "critical"
    reasons = [r["reason"] for r in data["retrieval"]["reasons"]]
    assert "hashing_embedder" in reasons
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_kb_mvp_health_retrieval.py -q`
Expected: FAIL — `KeyError: 'degraded'` (field not in response yet)

- [ ] **Step 3: Add the import**

In `app/api/kb_mvp.py`, add to the imports block:

```python
from app.observability import retrieval_health
```

- [ ] **Step 4: Read distinct embedding-dim count alongside kb_stats**

In `health()`, inside the existing `try:` block that reads `kb_documents`/`kb_chunks` counts (after the `chunks_count` query, ~line 621), add another query:

```python
                row = conn.execute("SELECT COUNT(DISTINCT dim) FROM kb_chunks").fetchone()
                distinct_dims = int(row[0]) if row else 0
```

And initialise `distinct_dims = 0` next to the other counters near the top of `health()` (with `documents_count = 0` etc.) so it is always defined.

- [ ] **Step 5: Build the retrieval snapshot before the return**

In `health()`, immediately before the final `return {` statement, add:

```python
    extra: list[tuple] = []
    try:
        if kb_embeddings.embedder_status().get("name") == "hash":
            extra.append(
                (retrieval_health.RetrievalReason.HASHING_EMBEDDER, "embedder=hash")
            )
    except Exception:  # pragma: no cover - never let a probe break health
        pass
    if distinct_dims > 1:
        extra.append(
            (
                retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH,
                f"{distinct_dims} distinct embedding dims present",
            )
        )
    retrieval = retrieval_health.snapshot(extra=tuple(extra))
```

- [ ] **Step 6: Add the fields to the returned dict**

In the `return {` dict of `health()`, add two keys (e.g. after `"status": "ok",`):

```python
        "degraded": retrieval["degraded"],
        "retrieval": retrieval,
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `py -3 -m pytest tests/test_kb_mvp_health_retrieval.py -q`
Expected: PASS (1 passed)

- [ ] **Step 8: Run the existing health tests to confirm no regression**

Run: `py -3 -m pytest tests/test_kb_compliance_mode_health.py -q`
Expected: PASS (3 passed)

- [ ] **Step 9: Commit**

```bash
git add app/api/kb_mvp.py tests/test_kb_mvp_health_retrieval.py
git commit -m "feat(api): surface retrieval degradation in /api/kb/health"
```

---

## Task 5: Surface degradation in `/ops/health/dependencies`

**Files:**
- Modify: `app/api/v1/ops.py` (`dependencies()` at line 36)
- Test: `tests/test_api_v1_ops_retrieval.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_v1_ops_retrieval.py`:

```python
"""/ops/health/dependencies merges retrieval degradation state."""

from __future__ import annotations

import app.api.v1.ops as ops
import app.observability.retrieval_health as retrieval_health


class _OkStore:
    def ensure_ready(self) -> None:
        return None


def test_dependencies_reports_retrieval_degradation(monkeypatch):
    retrieval_health.reset()
    monkeypatch.setattr(ops, "get_vector_store", lambda: _OkStore())
    retrieval_health.report(
        retrieval_health.RetrievalReport(
            source="fallback",
            reasons=(retrieval_health.RetrievalReason.VECTOR_BACKEND_DOWN,),
        )
    )

    result = ops.dependencies()

    assert result["checks"]["vector_store"] == "ok"
    assert result["checks"]["retrieval"] == "critical"
    assert result["status"] == "degraded"


def test_dependencies_ok_when_no_degradation(monkeypatch):
    retrieval_health.reset()
    monkeypatch.setattr(ops, "get_vector_store", lambda: _OkStore())

    result = ops.dependencies()

    assert result["checks"]["retrieval"] == "ok"
    assert result["status"] == "ok"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_api_v1_ops_retrieval.py -q`
Expected: FAIL — `KeyError: 'retrieval'`

- [ ] **Step 3: Implement the merge**

In `app/api/v1/ops.py`, add the import near the top:

```python
from app.observability import retrieval_health
```

Then replace the body of `dependencies()` with:

```python
def dependencies() -> dict[str, object]:
    """Dependency checks with degradation semantics."""
    checks: dict[str, str] = {"vector_store": "ok"}
    status = "ok"
    try:
        get_vector_store().ensure_ready()
    except Exception:
        checks["vector_store"] = "degraded"
        status = "degraded"
    snap = retrieval_health.snapshot()
    if snap["degraded"]:
        checks["retrieval"] = snap["severity"]
        status = "degraded"
    else:
        checks["retrieval"] = "ok"
    return {"status": status, "checks": checks, "version": get_version_info()}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `py -3 -m pytest tests/test_api_v1_ops_retrieval.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/api/v1/ops.py tests/test_api_v1_ops_retrieval.py
git commit -m "feat(api): surface retrieval degradation in /ops/health/dependencies"
```

---

## Task 6: Document the alert rule

**Files:**
- Modify: `docs/observability_slo.md`

- [ ] **Step 1: Append the alert-rule section**

Add this section to the end of `docs/observability_slo.md`:

```markdown
## Alert: retrieval degraded

`kb_retrieval_degraded{reason,severity}` is `1` while a retrieval path is
running in a quality-compromised mode and `0` otherwise. Reasons:
`vector_backend_down` (Qdrant/FAISS down → grep fallback), `hashing_embedder`
(no real embedder configured), `embedding_dim_mismatch` (index incoherent with
the active embedder — reindex needed), `search_truncated` (corpus exceeds the
scan cap).

Recommended Prometheus rule — page when any critical degradation persists for
5 minutes:

```yaml
- alert: RetrievalDegradedCritical
  expr: max_over_time(kb_retrieval_degraded{severity="critical"}[5m]) == 1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Retrieval degraded ({{ $labels.reason }})"
    description: "Answers may be irrelevant or empty. Check embedder config, reindex state, and vector backend health."
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/observability_slo.md
git commit -m "docs(observability): document kb_retrieval_degraded alert rule"
```

---

## Task 7: Full verification and PR1 ship checkpoint

**Files:** none (verification only)

- [ ] **Step 1: Run all retrieval-related tests**

Run: `py -3 -m pytest tests/test_retrieval_health.py tests/test_services_vectorstore.py tests/test_kb_store_retrieval_health.py tests/test_kb_mvp_health_retrieval.py tests/test_api_v1_ops_retrieval.py -q`
Expected: PASS (all)

- [ ] **Step 2: Lint and format the touched files**

Run: `py -3 -m ruff check app/observability/retrieval_health.py app/services/vectorstore.py app/services/kb_store.py app/api/kb_mvp.py app/api/v1/ops.py`
Then: `py -3 -m black --check app/observability/retrieval_health.py app/services/vectorstore.py app/services/kb_store.py app/api/kb_mvp.py app/api/v1/ops.py tests/test_retrieval_health.py tests/test_kb_store_retrieval_health.py tests/test_kb_mvp_health_retrieval.py tests/test_api_v1_ops_retrieval.py`
Expected: no errors. If black reports changes, run without `--check`, then `git add -u && git commit -m "style: black-format retrieval-degradation files"`.

- [ ] **Step 3: Type-check**

Run: `py -3 -m mypy app/observability/retrieval_health.py`
Expected: no errors (mypy excludes `backend/`; this targets the new module).

- [ ] **Step 4: Run the broader suite, skipping infra-marked tests**

Run: `py -3 -m pytest -m "not requires_postgres" -q`
Expected: PASS. If the in-process search default (hashing) now records degradation during unrelated tests, confirm no test asserts on a *clean* `retrieval_health` global without resetting it; the autouse fixtures added in Tasks 2–3 isolate the new tests. Investigate any failure rather than skipping it.

- [ ] **Step 5: PR1 is ready**

PR1 ("ops-loud") is now complete and independently shippable. Open a PR from `feat/retrieval-degradation-visibility` to `main`. PR2 ("customer-loud") will be planned separately.

---

## Self-Review (completed during authoring)

**Spec coverage (PR1 scope of §10):**
- §3 contract (`RetrievalReason`/`RetrievalSeverity`/`RetrievalReport`, registry, ContextVar, snapshot, TTL, `_SOURCE_REASONS` clear) → Task 1.
- §4 detection: `VECTOR_BACKEND_DOWN` → Task 2; `HASHING_EMBEDDER` / `EMBEDDING_DIM_MISMATCH` / `SEARCH_TRUNCATED` → Task 3.
- §5.1 health surfacing (`/api/kb/health` + active probes; `/ops/health/dependencies`) → Tasks 4–5. `status:"ok"` preserved (D4) — asserted in Task 4 Step 1.
- §6 gauge + alert → Task 1 (gauge) + Task 6 (alert doc).
- §9 testing incl. stub-drift guard → tests reuse `ExplodingVectorStore`/`_VECTOR_ERRORS`, never real qdrant classes.
- §11 D2 (hashing = CRITICAL) → `_SEVERITY`. D3 (ContextVar, no signature change) → Tasks 2–3. D5 (TTL backstop) → Task 1 Step 6 test.
- **Out of PR1 (deferred to PR2 plan):** §5.2 `/ask` field, `/ask/stream` `meta`, v1 chat, §5.3 UI banner + i18n.

**Placeholder scan:** none — every code step shows complete code; every run step shows the exact `py -3` command and expected outcome.

**Type/name consistency:** `report` / `current_report` / `snapshot` / `reset` / `severity_of` / `RetrievalReport(source, reasons, detail)` / `RetrievalReason.*` are used identically across Tasks 1–5. Gauge name `kb_retrieval_degraded` with labels `{reason, severity}` matches between Task 1 (definition + test) and Task 6 (alert rule).
