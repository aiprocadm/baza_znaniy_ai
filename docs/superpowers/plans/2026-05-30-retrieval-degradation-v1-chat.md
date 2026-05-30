# Retrieval Degradation Visibility — Implementation Plan (PR2b "v1 chat", multi-tenant path)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry the per-query `RetrievalReport` (built in PR1, #562) onto the **v1 multi-tenant** `ChatResponse`, so the operations console / API clients get the same honest, optional `retrieval` signal that the MVP `/api/kb/ask` already returns after PR2 (#563).

**Architecture:** PR1 already records a per-request `RetrievalReport` in a `ContextVar` from inside `vectorstore.search()` (the mature/Qdrant path). The v1 chat orchestrator (`chat_orchestrator.handle_chat`) calls that `search()` in its **legacy** retrieval mode, so `retrieval_health.current_report()` is populated mid-request. This PR adds an **optional, backward-compatible** `retrieval` field to `ChatResponse` and populates it **inside `handle_chat`** (right after `search()`), reusing the pure `retrieval_health.report_payload` serializer (DRY home, shared with the MVP path). **Zero change to retrieval behaviour or existing response fields** — the field defaults to `None` for healthy queries.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pytest (Windows: `py -3`).

**Scope (this PR):** the **v1 multi-tenant chat path only** — `app/models/__init__.py` (`ChatResponse`) and `app/services/chat_orchestrator.py` (`handle_chat`). Per the two-path architecture (`docs/architecture.md`, CLAUDE.md "do NOT merge"), the v1 path keeps its **own** output models — a deliberate, tiny duplicate of the MVP `RetrievalReportOut` in `kb_mvp.py`; only the `report_payload` serializer is shared. The end-user UI banner + i18n shipped in PR2 for the MVP UI; the v1 **admin status-pill** (spec §5.3, open question Q2) remains deferred until a pilot asks.

**Reference spec:** `docs/superpowers/specs/2026-05-29-retrieval-degradation-visibility-design.md` (§5.2 line 139: *"The v1 chat response model gains the same optional `retrieval` field, fed from `current_report()`"*; §8 line 180). The PR1 contract this builds on lives in `app/observability/retrieval_health.py` (merged #562). MVP precedent is `app/api/kb_mvp.py` (PR2, #563).

---

## Known limitation (documented, by design)

`handle_chat` has two retrieval modes:
- **Legacy mode** (`langchain_enabled=False`, the default) calls `app.services.vectorstore.search()` → `retrieval_health.report(...)` runs → `current_report()` is populated → `retrieval` is meaningful.
- **LangChain mode** retrieves via its own chain, **not** our `search()`, so `current_report()` is unset for that request → `retrieval` is `None`.

This is acceptable: the field is optional and best-effort. Populating it for the LangChain retriever is out of scope (it would require instrumenting `app/langchain/factory.py`, a separate concern). The plan only touches the legacy path; LangChain responses keep `retrieval=None` via the field default — no special-casing needed.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `app/models/__init__.py` | `RetrievalReasonOut`/`RetrievalReportOut` (v1's own copy); optional `retrieval` field on `ChatResponse` | Modify |
| `app/services/chat_orchestrator.py` | Read `current_report()` after `search()`; pass `retrieval` to the legacy `ChatResponse` | Modify |
| `tests/test_api_v1_chat_retrieval.py` | Model-level coercion test + end-to-end `handle_chat` wiring tests (degraded + clean) | Create |

---

## Task 1: v1 output models + optional `ChatResponse.retrieval` field

**Files:**
- Modify: `app/models/__init__.py` (add two models before `class ChatResponse` @114; add a field on `ChatResponse`)
- Test: `tests/test_api_v1_chat_retrieval.py` (create)

- [ ] **Step 1: Write the failing model tests**

Create `tests/test_api_v1_chat_retrieval.py`:

```python
"""PR2b: the v1 ChatResponse carries an optional per-query retrieval report."""

from __future__ import annotations

from app.models import ChatResponse


def _base_kwargs() -> dict:
    return {
        "answer": "ok",
        "citations": [],
        "conversation_id": "c1",
        "citations_insufficient": False,
        "latency_ms": 1.0,
    }


def test_chat_response_retrieval_defaults_to_none():
    resp = ChatResponse(**_base_kwargs())
    assert resp.retrieval is None


def test_chat_response_coerces_retrieval_dict():
    resp = ChatResponse(
        **_base_kwargs(),
        retrieval={
            "degraded": True,
            "severity": "critical",
            "reasons": [
                {"reason": "vector_backend_down", "severity": "critical", "detail": "boom"}
            ],
        },
    )
    assert resp.retrieval is not None
    assert resp.retrieval.degraded is True
    assert resp.retrieval.severity == "critical"
    assert resp.retrieval.reasons[0].reason == "vector_backend_down"
    assert resp.retrieval.reasons[0].detail == "boom"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_api_v1_chat_retrieval.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'retrieval'` / validation error (field does not exist yet).

- [ ] **Step 3: Add the output models**

In `app/models/__init__.py`, immediately **before** `class ChatResponse(BaseModel):` (line 114), add:

```python
class RetrievalReasonOut(BaseModel):
    """A single retrieval-degradation reason surfaced to the v1 client.

    Mirrors one entry of ``retrieval_health.report_payload(...)`` /
    ``snapshot()``. The v1 path keeps its own copy of this model (separate
    from the MVP ``kb_mvp.RetrievalReportOut``) per the two-path architecture.
    """

    reason: str
    severity: str
    detail: str = ""


class RetrievalReportOut(BaseModel):
    """Per-query retrieval-degradation summary attached to ``ChatResponse``."""

    degraded: bool
    severity: str
    reasons: List[RetrievalReasonOut] = Field(default_factory=list)


```

(`BaseModel`, `Field`, `List`, `Optional` are already imported at the top of this module — they are used by the surrounding models.)

- [ ] **Step 4: Add the optional field to `ChatResponse`**

In `class ChatResponse`, after the `max_generation_tokens: Optional[int] = Field(...)` block (ends line 127), add:

```python
    retrieval: Optional[RetrievalReportOut] = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_api_v1_chat_retrieval.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Confirm no regression in models / existing chat tests**

Run: `py -3 -m pytest tests/test_reranking.py tests/test_chat_formatting.py -q`
Expected: PASS — the new field is additive (`None` default); existing assertions ignore it.

- [ ] **Step 7: Commit**

```bash
git add app/models/__init__.py tests/test_api_v1_chat_retrieval.py
git commit -m "feat(models): add optional retrieval field to v1 ChatResponse"
```

---

## Task 2: Populate `retrieval` inside `handle_chat` (legacy path)

**Files:**
- Modify: `app/services/chat_orchestrator.py` (import @16/@18; compute after `search()` @225; field on the legacy `ChatResponse` @303)
- Test: `tests/test_api_v1_chat_retrieval.py` (extend)

- [ ] **Step 1: Write the failing wiring tests**

Append to `tests/test_api_v1_chat_retrieval.py`:

```python
import types
from typing import Any, List

import pytest

import app.observability.retrieval_health as retrieval_health
from app.api.v1 import chat as chat_module
from app.core.config import Settings
from app.models import ChatRequest
from app.observability.retrieval_health import RetrievalReason, RetrievalReport
from app.services import chat_orchestrator


class _StubChatStore:
    def ensure_conversation(self, user_id: str, conversation_id: str | None) -> str:
        return "conversation-id"

    def get_summary(self, conversation_id: str) -> str:
        return ""

    def get_recent_messages(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        return []

    def record_exchange(self, conversation_id: str, message: str, answer: str) -> None:
        return None

    def messages_since_summary(self, conversation_id: str) -> int:
        return 0


class _StubSummarizer:
    def summarize(self, conversation_id: str) -> None:
        return None


class _StubLLM:
    def ensure_model(self) -> None:
        return None

    def generate(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        return "Ответ"


def _build_request(settings: Settings) -> Any:
    min_citations, max_citations = settings.citations_bounds
    state = types.SimpleNamespace(
        settings=settings,
        chat_store=_StubChatStore(),
        llm_provider=_StubLLM(),
        llm_client=None,
        vector_store=None,
        summarizer=_StubSummarizer(),
        memory_store=None,
        fallback_index=[],
        reranker=None,
        chat_history_limit=settings.chat_history_limit,
        retrieve_topk=settings.retrieve_topk,
        rerank_topk=settings.rerank_limit,
        min_citations=min_citations,
        max_citations=max_citations,
        rerank_enabled=False,
        chat_summary_trigger=settings.chat_summary_trigger,
    )
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state))


_SAMPLE_HITS: List[dict[str, Any]] = [
    {"file": "doc1.pdf", "page": 1, "text": "alpha", "score": 0.5},
    {"file": "doc2.pdf", "page": 2, "text": "beta", "score": 0.4},
]


@pytest.fixture(autouse=True)
def _reset_retrieval_health():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def _run_chat(monkeypatch, stub_search) -> Any:
    settings = Settings().model_copy(
        update={
            "langchain_enabled": False,
            "retrieve_topk": 2,
            "chat_min_citations": 1,
            "chat_max_citations": 2,
        }
    )
    request = _build_request(settings)
    payload = ChatRequest(user_id="u", message="вопрос", conversation_id=None)
    monkeypatch.setattr(chat_orchestrator, "search", stub_search)
    return chat_module.chat(payload, request=request)


def test_chat_carries_retrieval_when_degraded(monkeypatch):
    def stub_search(query: str, top_k: int = 10) -> List[dict[str, Any]]:
        # Faithfully simulate the real vectorstore.search() grep-fallback report.
        retrieval_health.report(
            RetrievalReport(
                source="fallback",
                reasons=(RetrievalReason.VECTOR_BACKEND_DOWN,),
                detail="boom",
            )
        )
        return list(_SAMPLE_HITS)

    response = _run_chat(monkeypatch, stub_search)

    assert response.retrieval is not None
    assert response.retrieval.degraded is True
    assert response.retrieval.severity == "critical"
    assert any(r.reason == "vector_backend_down" for r in response.retrieval.reasons)


def test_chat_omits_retrieval_when_clean(monkeypatch):
    def stub_search(query: str, top_k: int = 10) -> List[dict[str, Any]]:
        retrieval_health.report(RetrievalReport(source="vector"))  # clean run
        return list(_SAMPLE_HITS)

    response = _run_chat(monkeypatch, stub_search)

    assert response.retrieval is None
```

- [ ] **Step 2: Run the wiring tests to verify they fail**

Run: `py -3 -m pytest tests/test_api_v1_chat_retrieval.py -k "carries or omits" -q`
Expected: FAIL — `assert None is not None` (handle_chat does not populate `retrieval` yet).

- [ ] **Step 3: Import the contract into the orchestrator**

In `app/services/chat_orchestrator.py`, add to the imports. After line 16 (`from app.models import ChatRequest, ChatResponse, Citation`), change it to also import the new model:

```python
from app.models import ChatRequest, ChatResponse, Citation, RetrievalReportOut
```

And after line 18 (`from app.observability.metrics import record_chat_completion`), add:

```python
from app.observability import retrieval_health
```

- [ ] **Step 4: Compute the report payload right after `search()`**

In `handle_chat`, the legacy retrieval call is line 225:

```python
        hits = list(search(payload.message, top_k=runtime.retrieve_topk))
```

Immediately after it, add:

```python
        retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())
```

- [ ] **Step 5: Pass the field to the legacy `ChatResponse`**

In the legacy-path `return ChatResponse(` block (line 303), add the `retrieval` field after `max_generation_tokens=runtime.llm_max_tokens,`:

```python
        return ChatResponse(
            answer=formatted_answer,
            citations=citations,
            conversation_id=conversation_id,
            citations_insufficient=not has_minimum,
            latency_ms=latency_ms,
            max_context_tokens=runtime.llm_ctx,
            max_generation_tokens=runtime.llm_max_tokens,
            retrieval=RetrievalReportOut(**retrieval_out) if retrieval_out else None,
        )
```

(The LangChain return at line 217 is intentionally left unchanged — it does not call `search()`, so `retrieval` stays `None` via the field default. See "Known limitation".)

- [ ] **Step 6: Run the wiring tests to verify they pass**

Run: `py -3 -m pytest tests/test_api_v1_chat_retrieval.py -q`
Expected: PASS (4 passed — 2 model + 2 wiring).

- [ ] **Step 7: Confirm no regression across the v1 chat path**

Run: `py -3 -m pytest tests/test_reranking.py tests/test_chat_llm_integration.py -q`
Expected: PASS — `retrieval` is additive; the legacy/langchain flows are otherwise unchanged.

- [ ] **Step 8: Commit**

```bash
git add app/services/chat_orchestrator.py tests/test_api_v1_chat_retrieval.py
git commit -m "feat(chat): surface retrieval degradation on v1 ChatResponse"
```

---

## Task 3: Full verification and PR2b ship checkpoint

**Files:** none (verification only)

- [ ] **Step 1: Run all retrieval + chat tests**

Run: `py -3 -m pytest tests/test_api_v1_chat_retrieval.py tests/test_retrieval_health.py tests/test_reranking.py tests/test_chat_formatting.py tests/test_chat_llm_integration.py -q`
Expected: PASS (all).

- [ ] **Step 2: Lint, format, type-check the touched Python**

Run: `py -3 -m ruff check app/models/__init__.py app/services/chat_orchestrator.py tests/test_api_v1_chat_retrieval.py`
Then: `py -3 -m black --check app/models/__init__.py app/services/chat_orchestrator.py tests/test_api_v1_chat_retrieval.py`
Then: `py -3 -m mypy app/services/chat_orchestrator.py app/models/__init__.py`
Expected: ruff/black clean. For mypy, confirm **no new** errors are introduced by this change (the repo carries a pre-existing mypy baseline; compare against `git stash` if unsure). If black rewrites, run without `--check` and commit `style: black-format PR2b files`.

- [ ] **Step 3: Broader suite (mirror CI: skip postgres-marked + ignore legacy backend/)**

Run: `py -3 -m pytest -m "not requires_postgres" -q --ignore=backend`
Expected: PASS. Investigate any failure rather than skipping it. The autouse `_reset_retrieval_health` fixture isolates the new tests from the global contract registry.

- [ ] **Step 4: PR2b is ready**

Open a PR from `feat/retrieval-degradation-v1-chat` to `main`. Title: `feat: surface retrieval degradation on v1 ChatResponse (PR2b, multi-tenant path)`. Body notes: completes the spec §5.2 v1 chat field deferred by PR2 (#563); LangChain-mode retrieval is a documented `None` (out of scope); admin status-pill (Q2) still deferred.

---

## Self-Review (completed during authoring)

**Spec coverage (PR2b scope):**
- §5.2 line 139 — "v1 chat response model gains the same optional `retrieval` field, fed from `current_report()`" → Task 1 (field) + Task 2 (populate from `current_report()` after `search()`).
- §8 line 180 — "v1 chat response model + endpoint — optional `retrieval` field" → Tasks 1–2.
- §5.2 "machine-readable reasons only" → `RetrievalReportOut`/`RetrievalReasonOut` carry `reason`/`severity`/`detail`; no human copy server-side (parity with MVP).
- **Deferred (called out in Scope):** §5.3 v1 admin status-pill (open question Q2); LangChain-retriever instrumentation (Known limitation).

**Decisions:**
- D-A — `retrieval` is **omitted (`None`) for healthy queries** (and for LangChain mode), keeping healthy responses byte-identical to today. `report_payload` returns `None` unless degraded.
- D-B — populate **inside `handle_chat`** (not the endpoint), so both the sync `POST /api/v1/chat` and the `WebSocket /ws/chat` entry points (the latter runs `handle_chat` in a separate threadpool/ContextVar context) get the field from the same context that `search()` set it in.
- D-C — v1 keeps its **own** `RetrievalReportOut` (duplicate of `kb_mvp.RetrievalReportOut`) per the two-path "do NOT merge" rule; only `report_payload` is shared (DRY where it counts — the serializer — not where the architecture wants separation).

**Placeholder scan:** none — every code step shows complete code; every run step shows the exact `py -3` command and expected outcome.

**Type/name consistency:** `report_payload` returns `{degraded, severity, reasons:[{reason, severity, detail}]}`; `RetrievalReportOut(degraded, severity, reasons)` / `RetrievalReasonOut(reason, severity, detail)` mirror it; `ChatResponse.retrieval: Optional[RetrievalReportOut]`; `handle_chat` builds it via `RetrievalReportOut(**retrieval_out) if retrieval_out else None`. Names align across Tasks 1–2 and match the merged MVP precedent in `kb_mvp.py`.
