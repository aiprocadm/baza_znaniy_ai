# Retrieval Degradation Visibility — Implementation Plan (PR2 "customer-loud", MVP path)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retrieval degradation visible to the **end user** of the MVP chat UI by carrying the per-query `RetrievalReport` (built in PR1) into the `/api/kb/ask` response and the `/api/kb/ask/stream` `meta` event, and rendering a dismissible, severity-coloured banner above the answer — copy localized via the existing i18n system.

**Architecture:** PR1 already records a per-request `RetrievalReport` in a `ContextVar` from inside `kb_store.search()`. This PR adds one pure serializer (`retrieval_health.report_payload`), surfaces it as an **optional, backward-compatible** field on the MVP response/stream, and consumes it in `data/www/index.html` with a banner whose text comes from `data/www/i18n/ru.json` via the global `window.t()` helper. **Zero change to retrieval behaviour or to existing response fields** — the new field defaults to `None`/absent for healthy queries.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pytest (Windows: `py -3`); vanilla HTML/JS for `data/www/`.

**Scope (this PR):** the **MVP customer path only** — `/api/kb/ask`, `/api/kb/ask/stream`, the end-user UI banner, and i18n. Per the spec's two-path architecture (`docs/architecture.md`, CLAUDE.md: "do NOT merge"), the **v1 multi-tenant chat** field (`ChatResponse.retrieval` in `app/api/v1/chat.py` + `app/models/__init__.py`) is a **deliberately separate follow-up plan** ("PR2b"), authored after this ships, so each PR stays within the ~400 LoC CONTRIBUTING budget and respects the path separation. The optional admin status-pill (spec §5.3, open question Q2) remains deferred until a pilot asks.

**Reference spec:** `docs/superpowers/specs/2026-05-29-retrieval-degradation-visibility-design.md` (§5.2 `/ask` + stream, §5.3 UI). The PR1 contract this builds on lives in `app/observability/retrieval_health.py` (merged in #562).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `app/observability/retrieval_health.py` | Add `report_payload(rep)` — pure `RetrievalReport → JSON-safe dict | None` serializer (DRY home for MVP now, v1 later) | Modify |
| `app/api/kb_mvp.py` | `RetrievalReasonOut`/`RetrievalReportOut` models; optional `retrieval` field on `AskResponse`; populate in `ask()`; add `retrieval` to `ask_stream()` `meta` | Modify |
| `data/www/i18n/ru.json` | Banner title + per-reason copy (spec §5.3) | Modify |
| `data/www/index.html` | `#ask-degraded` banner element, severity CSS, `renderDegradation()`, wire into `askSync`/`askStream`/form-reset | Modify |
| `tests/test_retrieval_health.py` | Unit tests for `report_payload` | Modify |
| `tests/test_kb_mvp_ask_retrieval.py` | `/ask` + `/ask/stream` carry `retrieval` (degraded) and omit it (clean) | Create |
| `tests/test_www_i18n_retrieval_keys.py` | Guard: ru.json keys present + index.html wires the banner | Create |

---

## Task 1: `report_payload` serializer in the contract module

**Files:**
- Modify: `app/observability/retrieval_health.py`
- Test: `tests/test_retrieval_health.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_health.py`:

```python
def test_report_payload_none_when_missing_or_clean():
    assert rh.report_payload(None) is None
    assert rh.report_payload(RetrievalReport(source="sqlite")) is None


def test_report_payload_mirrors_degraded_report():
    rep = RetrievalReport(
        source="sqlite",
        reasons=(RetrievalReason.HASHING_EMBEDDER, RetrievalReason.SEARCH_TRUNCATED),
        detail="embedder=hash",
    )
    payload = rh.report_payload(rep)
    assert payload["degraded"] is True
    assert payload["severity"] == "critical"  # CRITICAL dominates WARNING
    by_reason = {r["reason"]: r for r in payload["reasons"]}
    assert by_reason["hashing_embedder"]["severity"] == "critical"
    assert by_reason["search_truncated"]["severity"] == "warning"
    assert by_reason["hashing_embedder"]["detail"] == "embedder=hash"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: FAIL — `AttributeError: module 'app.observability.retrieval_health' has no attribute 'report_payload'`

- [ ] **Step 3: Implement `report_payload`**

Append to `app/observability/retrieval_health.py` (after `snapshot`, before `reset` is fine — order is irrelevant for module-level functions):

```python
def report_payload(rep: "RetrievalReport | None") -> dict | None:
    """Render a per-query report as a JSON-safe dict, or ``None`` when clean.

    Returns ``None`` when no report was recorded for this request context or
    the report is clean, so endpoints keep ``retrieval`` absent for healthy
    queries (backward compatible). When degraded, mirrors the ``snapshot()``
    reason entries (minus ``age_s``) so a client renders either source
    identically. ``detail`` is the report-level detail, shared across reasons.
    """

    if rep is None or not rep.degraded:
        return None
    return {
        "degraded": True,
        "severity": rep.severity.value,
        "reasons": [
            {
                "reason": reason.value,
                "severity": _SEVERITY.get(reason, RetrievalSeverity.WARNING).value,
                "detail": rep.detail,
            }
            for reason in rep.reasons
        ],
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_retrieval_health.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add app/observability/retrieval_health.py tests/test_retrieval_health.py
git commit -m "feat(observability): add report_payload serializer for per-query reports"
```

---

## Task 2: Carry `retrieval` on the `/api/kb/ask` response

**Files:**
- Modify: `app/api/kb_mvp.py` (models before `class AskResponse` @181; field on `AskResponse`; populate in `ask()` @942 + @962)
- Test: `tests/test_kb_mvp_ask_retrieval.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kb_mvp_ask_retrieval.py`:

```python
"""/api/kb/ask and /ask/stream carry the per-query retrieval degradation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.observability.retrieval_health as retrieval_health
from app.api.kb_mvp import router
from app.services import kb_llm
from app.services.kb_store import KnowledgeBaseStore


class _StubEmbedder:
    """A non-hashing embedder (name != 'hash') with a fixed dimension."""

    def __init__(self, dim: int = 8, name: str = "real") -> None:
        self.name = name
        self.dimension = dim
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        return [0.1] * self._dim


@pytest.fixture(autouse=True)
def _reset_retrieval_health():
    retrieval_health.reset()
    yield
    retrieval_health.reset()


def _client(store: KnowledgeBaseStore, monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/kb")
    app.state.kb_mvp_store = store
    monkeypatch.setattr(kb_llm, "select_provider", lambda: None)  # force extractive
    return TestClient(app)


def test_ask_reports_hashing_embedder_as_critical(tmp_path: Path, monkeypatch):
    # Default store -> hashing embedder (no KB_EMBEDDINGS_BACKEND) -> CRITICAL
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    data = client.post("/api/kb/ask", json={"question": "alpha"}).json()

    assert data["retrieval"] is not None
    assert data["retrieval"]["degraded"] is True
    assert data["retrieval"]["severity"] == "critical"
    reasons = [r["reason"] for r in data["retrieval"]["reasons"]]
    assert "hashing_embedder" in reasons


def test_ask_omits_retrieval_when_clean(tmp_path: Path, monkeypatch):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=_StubEmbedder())
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    data = client.post("/api/kb/ask", json={"question": "alpha"}).json()

    assert data["retrieval"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_kb_mvp_ask_retrieval.py -q`
Expected: FAIL — `KeyError: 'retrieval'` (field not on the response yet).

- [ ] **Step 3: Add the output models**

In `app/api/kb_mvp.py`, immediately **before** `class AskResponse(BaseModel):` (line 181), add:

```python
class RetrievalReasonOut(BaseModel):
    reason: str
    severity: str
    detail: str = ""


class RetrievalReportOut(BaseModel):
    degraded: bool
    severity: str
    reasons: List[RetrievalReasonOut] = Field(default_factory=list)


```

- [ ] **Step 4: Add the optional field to `AskResponse`**

In `class AskResponse`, after the `rerank: Optional[RerankInfo] = None` line (line 188), add:

```python
    retrieval: Optional[RetrievalReportOut] = None
```

- [ ] **Step 5: Populate it in `ask()`**

In `ask()`, the retrieval call at line 942 is:

```python
    hits, rerank_info = _retrieve_with_rerank(store, payload.question, payload.top_k)
```

Add a line immediately after it:

```python
    retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())
```

Then in the `return AskResponse(` block (line 962), add the field after `rerank=rerank_info,`:

```python
        rerank=rerank_info,
        retrieval=RetrievalReportOut(**retrieval_out) if retrieval_out else None,
        conversation_id=conversation.id,
```

(`retrieval_health` is already imported at `app/api/kb_mvp.py:25`; `BaseModel`/`Field`/`Optional`/`List` are already imported.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_kb_mvp_ask_retrieval.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Confirm no regression in the existing ask/search response tests**

Run: `py -3 -m pytest tests/test_kb_mvp_search_response.py -q`
Expected: PASS (4 passed) — the new field is additive; existing assertions ignore it.

- [ ] **Step 8: Commit**

```bash
git add app/api/kb_mvp.py tests/test_kb_mvp_ask_retrieval.py
git commit -m "feat(api): carry per-query retrieval degradation on /api/kb/ask"
```

---

## Task 3: Carry `retrieval` in the `/api/kb/ask/stream` `meta` event

**Files:**
- Modify: `app/api/kb_mvp.py` (`ask_stream()` @1038 + the `meta` dict @1044; docstring @1012)
- Test: `tests/test_kb_mvp_ask_retrieval.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kb_mvp_ask_retrieval.py`:

```python
def _read_meta_event(client: TestClient, question: str) -> dict:
    with client.stream("POST", "/api/kb/ask/stream", json={"question": question}) as resp:
        assert resp.status_code == 200
        text = "".join(chunk.decode("utf-8") for chunk in resp.iter_bytes())
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "event: meta" and i + 1 < len(lines):
            data_line = lines[i + 1]
            if data_line.startswith("data: "):
                return json.loads(data_line[len("data: ") :])
    raise AssertionError(f"meta event not found:\n{text}")


def test_ask_stream_meta_carries_retrieval_when_degraded(tmp_path: Path, monkeypatch):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")  # hashing default
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    meta = _read_meta_event(client, "alpha")

    assert meta["retrieval"] is not None
    assert meta["retrieval"]["severity"] == "critical"
    assert any(r["reason"] == "hashing_embedder" for r in meta["retrieval"]["reasons"])


def test_ask_stream_meta_retrieval_none_when_clean(tmp_path: Path, monkeypatch):
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=_StubEmbedder())
    store.add_document("doc1", text="alpha beta gamma " * 20)
    client = _client(store, monkeypatch)

    meta = _read_meta_event(client, "alpha")

    assert meta["retrieval"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_kb_mvp_ask_retrieval.py -k stream -q`
Expected: FAIL — `KeyError: 'retrieval'` (not in the meta payload yet).

- [ ] **Step 3: Compute the payload before the generator**

In `ask_stream()`, after line 1038–1039:

```python
    hits, rerank_info = _retrieve_with_rerank(store, payload.question, payload.top_k)
    source_payload = [_hit_to_out(hit).model_dump() for hit in hits]
```

add:

```python
    retrieval_out = retrieval_health.report_payload(retrieval_health.current_report())
```

- [ ] **Step 4: Add it to the `meta` dict**

In the `event_generator()` body, the `meta` dict (lines 1044–1048) becomes:

```python
        meta = {
            "conversation_id": conversation.id,
            "sources": source_payload,
            "rerank": rerank_info.model_dump() if rerank_info else None,
            "retrieval": retrieval_out,
        }
```

- [ ] **Step 5: Update the docstring event contract**

In the `ask_stream()` docstring, change the `meta` line (line 1012) from:

```python
    * ``event: meta``  — ``{conversation_id, sources, rerank}``
```

to:

```python
    * ``event: meta``  — ``{conversation_id, sources, rerank, retrieval}``
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_kb_mvp_ask_retrieval.py -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Confirm no regression in the existing stream test**

Run: `py -3 -m pytest tests/test_kb_mvp_search_response.py -k stream -q`
Expected: PASS (1 passed).

- [ ] **Step 8: Commit**

```bash
git add app/api/kb_mvp.py tests/test_kb_mvp_ask_retrieval.py
git commit -m "feat(api): emit retrieval degradation in /ask/stream meta event"
```

---

## Task 4: i18n copy for the banner

**Files:**
- Modify: `data/www/i18n/ru.json`
- Test: `tests/test_www_i18n_retrieval_keys.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_www_i18n_retrieval_keys.py`:

```python
"""The end-user banner copy + wiring exist (guards against typos / missing keys)."""

from __future__ import annotations

import json
from pathlib import Path

WWW = Path(__file__).resolve().parents[1] / "data" / "www"

REQUIRED_KEYS = [
    "retrieval.banner.title",
    "retrieval.reason.hashing_embedder",
    "retrieval.reason.embedding_dim_mismatch",
    "retrieval.reason.search_truncated",
    "retrieval.reason.vector_backend_down",
]


def test_ru_json_has_retrieval_banner_keys():
    dictionary = json.loads((WWW / "i18n" / "ru.json").read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in dictionary]
    assert not missing, f"missing i18n keys: {missing}"


def test_index_html_wires_degradation_banner():
    html = (WWW / "index.html").read_text(encoding="utf-8")
    assert 'id="ask-degraded"' in html
    assert "renderDegradation" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `py -3 -m pytest tests/test_www_i18n_retrieval_keys.py -q`
Expected: FAIL — `assert not [...]` (keys missing) and the index.html assertions fail.

- [ ] **Step 3: Add the keys to `ru.json`**

In `data/www/i18n/ru.json`, change the last real entry so the new keys are appended (the file currently ends with `"admin.section.docs": "Управление документами"` followed by `}`). Replace:

```json
  "admin.section.docs": "Управление документами"
}
```

with:

```json
  "admin.section.docs": "Управление документами",
  "retrieval.banner.title": "Качество поиска снижено",
  "retrieval.reason.hashing_embedder": "Поиск работает в режиме hashing-эмбеддера — ответы будут нерелевантны. Задайте KB_EMBEDDINGS_BACKEND и выполните kb-cli reindex.",
  "retrieval.reason.embedding_dim_mismatch": "Индекс рассогласован с активным эмбеддером — поиск ничего не находит. Выполните kb-cli reindex.",
  "retrieval.reason.search_truncated": "Корпус больше лимита сканирования — ответ мог пропустить релевантные фрагменты.",
  "retrieval.reason.vector_backend_down": "Векторный бэкенд недоступен — поиск временно работает по подстроке, качество снижено."
}
```

(The `tests/test_www_i18n_retrieval_keys.py::test_index_html_wires_degradation_banner` part stays red until Task 5 — that is expected and called out in Task 5 Step 6.)

- [ ] **Step 4: Run the JSON-keys test to verify that half passes**

Run: `py -3 -m pytest tests/test_www_i18n_retrieval_keys.py::test_ru_json_has_retrieval_banner_keys -q`
Expected: PASS (1 passed). The `test_index_html_wires_degradation_banner` test remains FAIL until Task 5.

- [ ] **Step 5: Commit**

```bash
git add data/www/i18n/ru.json tests/test_www_i18n_retrieval_keys.py
git commit -m "feat(www): add retrieval-degradation banner copy (ru i18n)"
```

---

## Task 5: End-user degradation banner in `data/www/index.html`

**Files:**
- Modify: `data/www/index.html` (HTML element @476; `<style>` block ~@176; JS `renderDegradation`, `askSync` @913, `askStream` meta @984, form reset @1010)

- [ ] **Step 1: Add the banner element above the answer**

In `data/www/index.html`, the ask panel currently has (lines 476–477):

```html
        <div id="ask-answer" class="pre" hidden></div>
        <div id="ask-sources" class="sources" style="margin-top: 0.75rem;"></div>
```

Insert the banner **before** `#ask-answer`:

```html
        <div id="ask-degraded" class="degraded-banner" role="alert" hidden></div>
        <div id="ask-answer" class="pre" hidden></div>
        <div id="ask-sources" class="sources" style="margin-top: 0.75rem;"></div>
```

- [ ] **Step 2: Add severity CSS**

In the `<style>` block, after the `.sources { ... }` rule (line 176), add:

```css
    .degraded-banner { border-radius: 8px; padding: 0.7rem 0.9rem; margin: 0 0 0.75rem; font-size: 0.9rem; line-height: 1.35; }
    .degraded-banner strong { display: block; margin-bottom: 0.2rem; }
    .degraded-banner ul { margin: 0.3rem 0 0; padding-left: 1.1rem; }
    .degraded-banner.warning { background: #41360f; border: 1px solid #b8860b; color: #ffe7a3; }
    .degraded-banner.critical { background: #441616; border: 1px solid #c0392b; color: #ffcfca; }
```

- [ ] **Step 3: Add the `renderDegradation` helper**

In the main `<script>` (after `escapeHtml`, defined at line 577, is a natural home), add:

```javascript
    function renderDegradation(retrieval) {
      const el = $("ask-degraded");
      if (!el) return;
      const reasons = (retrieval && retrieval.reasons) || [];
      if (!retrieval || !retrieval.degraded || !reasons.length) {
        el.hidden = true;
        el.innerHTML = "";
        return;
      }
      el.className =
        "degraded-banner " + (retrieval.severity === "critical" ? "critical" : "warning");
      const title = t("retrieval.banner.title", "Качество поиска снижено");
      const items = reasons
        .map((r) => `<li>${escapeHtml(t("retrieval.reason." + r.reason, r.detail || r.reason))}</li>`)
        .join("");
      el.innerHTML = `<strong>${escapeHtml(title)}</strong><ul>${items}</ul>`;
      el.hidden = false;
    }
```

- [ ] **Step 4: Call it from `askSync`**

In `askSync`, after `renderSources("ask-sources", data.sources);` (line 913), add:

```javascript
      renderDegradation(data.retrieval);
```

- [ ] **Step 5: Call it from `askStream` (meta event) and reset on a new ask**

In `askStream`, the `meta` branch (lines 980–984) ends with `renderSources("ask-sources", data.sources || []);`. After it, add:

```javascript
            renderDegradation(data.retrieval);
```

In the `ask-form` submit handler, the line `$("ask-answer").hidden = true;` (line 1010) hides the previous answer before a new request. Immediately after it, add:

```javascript
      $("ask-degraded").hidden = true;
```

- [ ] **Step 6: Run the i18n/wiring guard test to verify it now fully passes**

Run: `py -3 -m pytest tests/test_www_i18n_retrieval_keys.py -q`
Expected: PASS (2 passed) — both the keys test and the wiring test are green now.

- [ ] **Step 7: Commit**

```bash
git add data/www/index.html
git commit -m "feat(www): show severity-coloured retrieval-degradation banner in chat UI"
```

---

## Task 6: Full verification, browser proof, and PR2 ship checkpoint

**Files:** none (verification only)

- [ ] **Step 1: Run all PR2-related Python tests**

Run: `py -3 -m pytest tests/test_retrieval_health.py tests/test_kb_mvp_ask_retrieval.py tests/test_www_i18n_retrieval_keys.py tests/test_kb_mvp_search_response.py -q`
Expected: PASS (all).

- [ ] **Step 2: Lint, format, type-check the touched Python**

Run: `py -3 -m ruff check app/observability/retrieval_health.py app/api/kb_mvp.py`
Then: `py -3 -m black --check app/observability/retrieval_health.py app/api/kb_mvp.py tests/test_retrieval_health.py tests/test_kb_mvp_ask_retrieval.py tests/test_www_i18n_retrieval_keys.py`
Then: `py -3 -m mypy app/observability/retrieval_health.py`
Expected: no errors. If black rewrites, run without `--check` and commit `style: black-format PR2 files`.

- [ ] **Step 3: Validate the JSON parses (catch trailing-comma typos)**

Run: `py -3 -c "import json,pathlib; json.loads(pathlib.Path('data/www/i18n/ru.json').read_text(encoding='utf-8')); print('ru.json OK')"`
Expected: `ru.json OK`.

- [ ] **Step 4: Broader suite (mirror CI: skip postgres-marked + ignore legacy backend/)**

Run: `py -3 -m pytest -m "not requires_postgres" -q --ignore=backend`
Expected: PASS. (The `--ignore=backend` matches `.github/workflows/ci.yml`: legacy `backend/tests` fail at collection by design and are not part of the active tree.)

- [ ] **Step 5: Browser proof of the banner (MVP dev server + preview tools)**

Start the light MVP dev server (no Qdrant, hashing embedder default → a CRITICAL banner is expected on a fresh install, which is exactly what we want to see):

Run (background): `py -3 -m uvicorn scripts.dev_server_mvp:app --port 8001`

Then with the preview tools: open `http://localhost:8001/`, go to the «Вопрос-ответ» tab, add a document (or use an existing one), ask a question, and confirm the red CRITICAL banner appears above the answer with the hashing-embedder copy. Capture a `preview_screenshot` as proof. Stop the server when done.

- [ ] **Step 6: PR2 (MVP) is ready**

Open a PR from `feat/retrieval-degradation-customer-loud` to `main`. Title: `feat: surface retrieval degradation to end users (PR2 customer-loud, MVP path)`. Note in the body that the v1 chat field is a separate follow-up plan.

---

## Self-Review (completed during authoring)

**Spec coverage (PR2 scope, MVP path):**
- §5.2 `/ask` optional `retrieval` field (backward-compatible `None` default) → Task 2. Populated from `current_report()` immediately after `_retrieve_with_rerank` (matches spec) → Task 2 Step 5.
- §5.2 `/ask/stream` — extend the **existing** `meta` event (no new event type) → Task 3.
- §5.2 "machine-readable reasons only; human copy built client-side from i18n keyed by `reason`" → server sends `reason`/`severity`/`detail` (Task 1 `report_payload`); copy in ru.json keyed by reason (Task 4); banner maps reason→copy via `t()` (Task 5 Step 3).
- §5.3 UI banner above the answer, severity-coloured (amber WARNING / red CRITICAL), reason-specific instructive copy → Task 5. Copy strings are the spec's verbatim examples → Task 4 Step 3.
- §5.3 "all strings in `data/www/i18n/ru.json`, no hardcoded RU" → the banner pulls every string via `t()`; only the i18n JSON holds copy.
- **Deferred (own plan, called out in Scope):** §5.2 v1 chat field (`app/api/v1/chat.py` + `app/models`); §5.3 optional admin status-pill (open question Q2).

**Decisions:**
- D-A — `retrieval` is **omitted (`None`) for healthy queries** (not `{degraded:false}`), keeping healthy responses byte-identical to today and the client check trivial (`if (retrieval && retrieval.degraded)`). `report_payload` returns `None` unless degraded.
- D-B — serializer lives in `retrieval_health.report_payload` (pure, additive — no contract/behaviour change), so the deferred v1 plan reuses it (DRY).
- D-C — per-reason `detail` is the report-level `detail` (the `RetrievalReport` carries one). Acceptable: `detail` is supplementary; the banner copy comes from i18n by `reason`.

**Placeholder scan:** none — every code step shows complete code; every run step shows the exact `py -3` command and expected outcome.

**Type/name consistency:** `report_payload` returns `{degraded, severity, reasons:[{reason, severity, detail}]}`; `RetrievalReportOut`/`RetrievalReasonOut` mirror that shape; `meta.retrieval` carries the same dict; `renderDegradation` reads `retrieval.degraded`/`.severity`/`.reasons[].reason`. Names align across Tasks 1–5.
