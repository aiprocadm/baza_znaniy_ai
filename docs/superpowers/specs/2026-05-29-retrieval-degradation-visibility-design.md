# Retrieval Degradation Visibility — Design

**Date:** 2026-05-29
**Author scope:** technical design for trust-hardening the RAG answer path
**Status:** Design document. Subordinate to `2026-05-22-project-vision-design.md`.
**Decision context:** During pilots the product can silently degrade retrieval quality and keep answering as if nothing is wrong. This spec makes every such degradation **loud** — surfaced in health, in the `/ask` response, in the UI, and as a Prometheus metric — through a single unified contract shared by both HTTP surfaces.

> Этот документ описывает **что и как сделать**, не привязываясь к срокам. Реализация стадируется на 2 PR (см. §10).

---

## 1. Context and problem

KB.AI answers questions over a customer corpus. When retrieval silently degrades, the LLM still produces a confident-looking answer — but built on near-random, empty, or truncated context. The user cannot tell a degraded answer from a good one. For a pilot, this is the single worst failure mode: uptime dashboards stay green while the customer concludes "the AI is dumb." This destroys trust precisely when we are trying to earn it.

The repo intentionally ships **two parallel retrieval paths** (`docs/architecture.md`), and each degrades differently:

### 1.1 Customer-facing MVP path — `/api/kb/*`

`POST /api/kb/ask` → `_retrieve_with_rerank()` (`app/api/kb_mvp.py:335`) → `KnowledgeBaseStore.search()` (`app/services/kb_store.py:472`), a SQLite cosine scan. **It does not use Qdrant or the grep-fallback.** Its silent degradations are:

| # | Degradation | Mechanism | Pilot sees |
|---|---|---|---|
| 1 | **Hashing embedder active** | `KB_EMBEDDINGS_BACKEND` unset → hashing fallback (`HASHING_EMBEDDER_NAME = "hash"`, `kb_store.py:38`); near-random semantic matches | Confident but irrelevant answers |
| 2 | **Embedding dim mismatch** | `search()` filters `WHERE c.dim = ?` (`kb_store.py:492`). After an embedder swap **without reindex** (CLAUDE.md gotcha) all stored chunks carry the old dimension → **0 rows** | "В базе нет данных" — indistinguishable from an empty KB |
| 3 | **Hard-limit truncation** | `_search_hard_limit()` caps the scan (`kb_store.py:41`); on a large corpus only the first N chunks (arbitrary SQL order) are scored. Currently a `LOGGER.warning` only (`kb_store.py:497`) | "Не найдено", though the answer exists in an unscanned chunk |

### 1.2 Mature path — `/api/v1/*`

Uses `app/services/vectorstore.py` (Qdrant/FAISS via the `VectorStore` Protocol in `app/retriever/vector_store.py`). On backend failure `search()` catches `_VECTOR_ERRORS` and **silently falls back to a substring grep** over an in-memory index (`vectorstore.py:230`, the `_search_fallback` substring scan). CLAUDE.md flags this explicitly: the fallback is "a substring scan, **not a real vector search**."

| # | Degradation | Mechanism | Operator sees |
|---|---|---|---|
| 4 | **Vector backend down** | Qdrant/FAISS raises → grep-fallback path | Plausible-but-wrong substring matches |

### 1.3 What already exists (do not rebuild)

- Gauge precedent for "active state" — `EMBEDDER_BACKEND_ACTIVE` (`metrics.py:140`), `SQLMODEL_METADATA_HEALTH` (`metrics.py:147`). The set-one-zero-others idiom is the model for our gauge.
- Search counters `kb_search_queries_total{source,status}` already distinguish `vector` vs `fallback` (`metrics.py:74`, `record_search_operation` at `:216`). The signal exists in Prometheus — but nobody watches Prometheus during a pilot.
- `/api/kb/health` (`kb_mvp.py:598`) already returns `embedder`/`llm`/`reranker`/`kb_stats`.
- `/ops/health/dependencies` (`app/api/v1/ops.py:36`) already probes `get_vector_store().ensure_ready()` and reports `vector_store: degraded`.

**The gap is not detection-from-scratch; it is a persistent, human-visible signal.** The degradations are logged and/or counted but never surfaced where a pilot or operator actually looks.

## 2. Goal and non-goals

**Goal.** A single unified contract that both retrieval paths populate, surfaced loudly at three points: health endpoints (persistent, queryable out-of-band), the `/ask` response (honest about the specific query), and the UI (a banner the pilot cannot miss) — plus a Prometheus gauge for alerting.

**Non-goals.**
- Not *fixing* the degradations (e.g. auto-reindex on dim mismatch, hybrid search for large corpora) — only making them visible.
- Not unifying the two HTTP paths (anti-pattern per `docs/architecture.md`).
- No new heavy dependencies; the contract must not pull multi-tenant deps into the light MVP install.

## 3. The unified contract — `app/observability/retrieval_health.py` (new)

A dependency-free module (only `prometheus_client`, already a runtime dep). Lives in `app/observability/` because both the light MVP path and the heavy v1 path already import it, so it couples neither to the other.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class RetrievalSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"      # results may be incomplete
    CRITICAL = "critical"    # results likely useless or empty

class RetrievalReason(str, Enum):
    HEALTHY                = "healthy"
    VECTOR_BACKEND_DOWN    = "vector_backend_down"      # Qdrant/FAISS failed → grep fallback (mature path)
    HASHING_EMBEDDER       = "hashing_embedder"         # near-random matches (both paths)
    EMBEDDING_DIM_MISMATCH = "embedding_dim_mismatch"   # index incoherent with active embedder
    SEARCH_TRUNCATED       = "search_truncated"         # hard-limit hit (MVP path)

_SEVERITY: dict[RetrievalReason, RetrievalSeverity] = {
    RetrievalReason.VECTOR_BACKEND_DOWN:    RetrievalSeverity.CRITICAL,
    RetrievalReason.HASHING_EMBEDDER:       RetrievalSeverity.CRITICAL,  # see §11 decision D2
    RetrievalReason.EMBEDDING_DIM_MISMATCH: RetrievalSeverity.CRITICAL,
    RetrievalReason.SEARCH_TRUNCATED:       RetrievalSeverity.WARNING,
}

@dataclass(frozen=True)
class RetrievalReport:
    source: str                                  # "vector" | "fallback" | "sqlite"
    reasons: tuple[RetrievalReason, ...] = ()    # empty when clean
    detail: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.reasons)

    @property
    def severity(self) -> RetrievalSeverity:
        sev = [_SEVERITY[r] for r in self.reasons]
        if RetrievalSeverity.CRITICAL in sev:
            return RetrievalSeverity.CRITICAL
        if RetrievalSeverity.WARNING in sev:
            return RetrievalSeverity.WARNING
        return RetrievalSeverity.OK
```

**Module API:**

- `report(report: RetrievalReport) -> None` — upserts each *active* reason into a thread-safe registry with a fresh `time.monotonic()` stamp, and **clears** the reasons governed by `report.source` that are absent from `report.reasons` (a small `_SOURCE_REASONS` map says which reasons each source can raise — e.g. `sqlite` governs hashing/dim-mismatch/truncated, `vector` governs backend-down). So a clean run drops that source's reasons immediately, not only after the TTL. Drives the gauge (§6).
- `current_report() -> RetrievalReport | None` — reads the per-request report from a `contextvars.ContextVar` (set by `search()`; see §11 decision D3). Used by endpoints to attach the honest per-query signal.
- `snapshot(ttl_seconds: float = 300.0) -> dict` — active degradations whose timestamp is within `ttl`, merged with cheap active probes (§5.1). Shape:
  ```json
  {"degraded": true, "severity": "critical",
   "reasons": [{"reason": "embedding_dim_mismatch", "severity": "critical",
                "detail": "query dim 384 ≠ stored dims {768}", "age_s": 12.4}]}
  ```
- A degradation **self-heals**: after `ttl` of clean operation (or an explicit clean `report`), `snapshot()` returns `degraded: false`.

## 4. Detection points (two — exactly where swallowing happens today)

| Path / file | Reason | Trigger to add | Severity |
|---|---|---|---|
| `kb_store.search()` (`kb_store.py:472`) | `HASHING_EMBEDDER` | active embedder kind == `HASHING_EMBEDDER_NAME` | CRITICAL |
| `kb_store.search()` | `EMBEDDING_DIM_MISMATCH` | `WHERE c.dim=?` returned 0 rows **and** `kb_chunks` non-empty (cheap `SELECT EXISTS(...)` + `SELECT DISTINCT dim … LIMIT 2` for `detail`) | CRITICAL |
| `kb_store.search()` | `SEARCH_TRUNCATED` | `len(rows) >= hard_limit` (the condition already at `kb_store.py:497`) | WARNING |
| `vectorstore.search()` (`vectorstore.py:198`) | `VECTOR_BACKEND_DOWN` | inside the existing `except _VECTOR_ERRORS` branch (`vectorstore.py:230`) | CRITICAL |
| both | `HEALTHY` | clean run → `report(RetrievalReport(source=…))` clears prior state | OK |

Detection reuses the existing `_VECTOR_ERRORS` tuple (`vectorstore.py:34`, already `None`-filtered for absent qdrant classes) — so it never references real qdrant exception types the test stub lacks (CLAUDE.md stub-drift guard).

`HASHING_EMBEDDER` may also be reported from the mature path's embedder resolution; both paths share the same check helper.

## 5. Surfacing

### 5.1 Health (persistent: state + active probes)

- **`/api/kb/health`** (`kb_mvp.py:598`): add a `"retrieval"` block = `snapshot()` and a top-level `"degraded": bool`. **`"status"` stays `"ok"`** — `/health` is reachable without an API key for liveness probes (`kb_auth.py:14`); changing `status` would break external healthchecks. Active probes added here (true regardless of recent queries): embedder kind (reuse `kb_embeddings.embedder_status()`), and dim coherence via a cheap `SELECT DISTINCT dim FROM kb_chunks LIMIT 2` compared to the active embedder dimension.
- **`/ops/health/dependencies`** (`ops.py:36`): merge `snapshot()` into the existing `checks` dict, so a *recent* grep-fallback or hashing state is reported even when a now-recovered `ensure_ready()` probe passes (flapping backend). Existing `vector_store` probe and `status` semantics preserved.

### 5.2 `/ask` response (honest about this query)

- `AskResponse` (`kb_mvp.py:180`) gains an **optional** field `retrieval: RetrievalReportOut | None = None` (backward-compatible: `None` default → existing clients unaffected). `RetrievalReportOut` mirrors the snapshot (`degraded`, `severity`, and `reasons[]` each with `reason`/`severity`/`detail`). The server sends **machine-readable reasons only** — human copy is built client-side from `data/www/i18n/ru.json` keyed by `reason`, keeping localization in one place (CLAUDE.md). Populated in `ask()` from `retrieval_health.current_report()` immediately after `_retrieve_with_rerank` (`kb_mvp.py:921`).
- **`/ask/stream`** (`kb_mvp.py:985`): extend the **existing** `meta` event payload (`{conversation_id, sources, rerank}` → `+ retrieval`, `kb_mvp.py:991`). `meta` is emitted before tokens, so the banner appears before the answer streams. No new event type.
- The v1 chat response model gains the same optional `retrieval` field, fed from `current_report()`.

### 5.3 UI

- **End-user** `data/www/index.html`: render a dismissible banner above the answer when `retrieval.degraded`. Colour by severity — amber (WARNING) / red (CRITICAL) — with a reason-specific, instructive hint. All strings go in `data/www/i18n/ru.json` (CLAUDE.md: no hardcoded RU strings). Example copy:
  - `hashing_embedder` (CRITICAL): «Поиск работает в режиме hashing-эмбеддера — ответы будут нерелевантны. Задайте `KB_EMBEDDINGS_BACKEND` и выполните `kb-cli reindex`.»
  - `embedding_dim_mismatch` (CRITICAL): «Индекс рассогласован с активным эмбеддером — поиск ничего не находит. Выполните `kb-cli reindex`.»
  - `search_truncated` (WARNING): «Корпус больше лимита сканирования — ответ мог пропустить релевантные фрагменты.»
  - `vector_backend_down` (CRITICAL): «Векторный бэкенд недоступен — поиск временно работает по подстроке, качество снижено.»
- **Admin (PR2, optional)**: a status pill in the Operations Console (`data/www/admin.html`) polling `/api/kb/health`, so degradation is visible before any question is asked.

## 6. Metric & alert

- New gauge `kb_retrieval_degraded{reason, severity}` set 1/0 per reason by `retrieval_health.report()`. **Implemented self-contained in `app/observability/retrieval_health.py`** (defensive `prometheus_client` import) rather than in `metrics.py` — importing `metrics.py` would drag its `sqlalchemy` dependency into the light MVP path, violating decision D1. The gauge follows the same set-active-zero-others idiom as `record_embedder_backend`.
- `docs/observability_slo.md`: document an alert rule — fire when `max_over_time(kb_retrieval_degraded{severity="critical"}[5m]) == 1`.

## 7. Data flow

```
kb_store.search()  (MVP, SQLite)   ─┐
vectorstore.search() (mature, Qdrant) ─┤→ RetrievalReport ──┬─→ retrieval_health.report() ─→ registry + Gauge
                                       │   (unified contract)│        │
                                       │                     │        └─→ snapshot() ─→ /api/kb/health, /ops/health/dependencies ─→ alert
                                       │                     └─→ ContextVar ─→ current_report()
                                       │                              │
                                       │                              └─→ /ask, /ask/stream(meta), v1 chat ─→ UI banner (data/www)
```

## 8. Files to create or modify

**New:**
- `app/observability/retrieval_health.py` — contract (enums, `RetrievalReport`), registry, `report`/`current_report`/`snapshot`, ContextVar.
- `tests/test_retrieval_health.py` — unit tests for the module.
- `tests/test_retrieval_degradation_api.py` — health/`/ask`/stream surfacing tests.

**Modified:**
- `app/observability/metrics.py` — `kb_retrieval_degraded` gauge + `record_retrieval_degraded` helper.
- `app/services/kb_store.py` — detection for reasons 1–3 + `report(...)` on clean/degraded.
- `app/services/vectorstore.py` — detection for reason 4 inside the existing `except _VECTOR_ERRORS` branch + `report(...)`.
- `app/api/kb_mvp.py` — `RetrievalReportOut` model; `retrieval` field on `AskResponse`; populate in `ask()`; `meta` event in `ask_stream()`; `retrieval` block + `degraded` in `health()`.
- `app/api/v1/ops.py` — merge `snapshot()` into `/health/dependencies`.
- v1 chat response model + endpoint — optional `retrieval` field.
- `data/www/index.html` + `data/www/i18n/ru.json` — banner + strings.
- `data/www/admin.html` (PR2, optional) — status pill.
- `docs/observability_slo.md` — alert rule.
- `tests/stubs/*` — only if a stub lacks a method the new code calls (check before assuming a code bug — CLAUDE.md).

## 9. Testing (TDD)

- **Unit `retrieval_health`:** `report`/clear/TTL/`snapshot`; `severity` aggregation (CRITICAL dominates WARNING); gauge set-active-zero-others; ContextVar isolation between two sequential queries.
- **`kb_store.search`:** stub embedder kind `hash` → `HASHING_EMBEDDER`; stored dim ≠ query dim with non-empty table → `EMBEDDING_DIM_MISMATCH` + 0 hits; low `KB_SEARCH_HARD_LIMIT` over a larger corpus → `SEARCH_TRUNCATED`; clean run → `HEALTHY` clears state.
- **`vectorstore.search`:** force a member of `_VECTOR_ERRORS` → `VECTOR_BACKEND_DOWN` reported, grep hits still returned (extend existing fallback tests).
- **API:** `/api/kb/health` exposes the `retrieval` block + `degraded`; `/api/kb/ask` carries `retrieval`; `/ask/stream` emits `retrieval` in `meta`.
- Run path-scoped per CLAUDE.md: `py -3 -m pytest -k "retrieval"`.

## 10. Staging (2 PRs, ~400 LoC each per CONTRIBUTING)

- **PR1 — "ops-loud":** `retrieval_health` + gauge + detection on both paths + both health endpoints + alert doc + unit/health tests. No client-facing surface change. Ships operator/alert visibility immediately.
- **PR2 — "customer-loud":** `retrieval` field on `/ask` + `/ask/stream` + v1 chat + UI banner + i18n + optional admin pill + API tests. Ships the pilot-facing banner.

PR1 is independently valuable and carries no UX risk; PR2 builds on its contract.

## 11. Design decisions

- **D1 — Home in `app/observability/`.** Keeps the contract dependency-free so the light MVP install does not inherit multi-tenant deps; both paths couple only to a narrow interface.
- **D2 — Hashing embedder = CRITICAL (not WARNING).** It is the out-of-box default, so a CRITICAL banner fires on a fresh, unconfigured install. Accepted deliberately: for a pilot, silent hashing is the #1 trust-killer (CLAUDE.md). The copy is instructive (tells the operator how to fix it), so CRITICAL reads as "set this up", not "broken". *Revisit if first-run noise proves worse than the silent-garbage risk.*
- **D3 — ContextVar over signature change.** `kb_store.search()` returns `List[SearchHit]` and `vectorstore.search()` returns `List[dict]`, both called from many sites. `search()` pushes the report to the registry (for health/gauge, zero signature change) and sets a `contextvars.ContextVar`; the endpoint reads it via `current_report()` in the same request context (correct for both sync MVP threadpool endpoints and async v1).
- **D4 — Keep `/health → status:"ok"`.** Degradation rides in a new sub-block; external liveness probes that key on `status` keep working.
- **D5 — TTL self-heal (300 s) as backstop.** Explicit clears (§3, `_SOURCE_REASONS`) handle the common case — a clean query drops that source's reasons at once. The TTL is the backstop: it ages out a stale reason if a worker stops serving queries after a transient blip, so `snapshot()` never pins a degradation forever without manual reset.

## 12. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Per-process state — degradation in worker A invisible in worker B's health | Accepted: each worker reports its own; the gauge aggregates across workers in Prometheus; health is inherently per-process |
| Dim-coherence probe cost on every `/health` | Single `SELECT DISTINCT dim … LIMIT 2` (indexed, tiny); health is low-QPS |
| ContextVar leakage across requests | Explicit clean `report` at the start of `search()`; isolation test in §9 |
| Stub drift in qdrant exception handling | Reuse the existing `_VECTOR_ERRORS` tuple; do not reference real qdrant classes |
| Banner fatigue (CRITICAL on every dev install) | Instructive copy; dismissible; D2 marked for revisit |
| `/ask/stream` `meta` consumers ignoring the new key | Additive field; documented event contract already lists `meta` payload keys |

## 13. Out of scope

- Auto-remediation (auto-reindex on mismatch, hybrid/sparse search for large corpora) — deferred to the existing vectorstore refactor track.
- Indexing-time degradation (`index_chunks` fallback, `vectorstore.py:81`) — search-path focused here; can extend the same contract later.
- Unifying the two HTTP surfaces (anti-pattern, `docs/architecture.md`).

## 14. Alignment with vision

Subordinate to `2026-05-22-project-vision-design.md`. The vision gates progress on pilots succeeding (kill-criterion, month 6) and demands defensible quality. Silent retrieval degradation directly threatens a pilot's trust; making it loud protects the asset the pilot is evaluating. Touches only the MVP path (`/api/kb/*`, "the basis for every customer installation") plus a strictly additive read of the mature path — never modifying `/api/v1` retrieval behaviour.

## 15. Open questions

- **Q1:** D2 — keep hashing at CRITICAL, or downgrade to WARNING to reduce fresh-install noise? (Default: CRITICAL with instructive copy.)
- **Q2:** Is the admin status pill (PR2) wanted in v1, or deferred until a pilot asks?
