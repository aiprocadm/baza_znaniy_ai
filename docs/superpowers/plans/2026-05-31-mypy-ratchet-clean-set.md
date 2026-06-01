# mypy Ratchet Clean-Set + CI Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> ⚠️ **Session note (2026-05-31):** This plan was authored in a session that experienced intermittent **tool-output corruption** (fabricated/injected lines in some Read/Bash results). The plan content below is built only from *cross-verified* facts (stable across repeated runs + independent tools). **Every code edit MUST be gated by its `mypy <file>` → 0 verification step, run in a clean session.** If a verification step does not show the expected result, STOP and re-read the file — do not force the edit. Treat `app/api/v1/search.py:96` (Task 3) as the one fix not experimentally pre-verified; its verification step is the gate.

**Goal:** Drive five files to **0 mypy errors** with zero behavioral change, and add a non-advisory CI ratchet gate (`scripts/check_mypy_ratchet.py`) that fails the build if any documented clean file regresses.

**Architecture:** Five independent, type-only edits (one commit each), each verified by (a) the touched file reporting 0 errors under the whole-package `mypy app` check and (b) the module's existing behavioral tests still passing. One new characterization test pins `create_user`'s response (the only caller-visible refactor). A sixth commit adds the ratchet script + CI wiring, seeded with the seven now-clean files.

**Tech Stack:** Python 3.12+, mypy (config `[tool.mypy]` in `pyproject.toml`), pytest, SQLModel/SQLAlchemy, Pydantic v2, FastAPI TestClient. Windows: invoke via `py -3` (no venv).

**Spec:** `docs/superpowers/specs/2026-05-31-mypy-ratchet-clean-set-design.md`
**Branch:** `chore/mypy-ratchet-clean-set` (exists; 1 commit ahead of `main` with the spec).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `app/llm/lora_runtime.py` | LoRA registry/manifest parsing | Modify: JSON manifest typed `dict[str, Any]` (2 → 0) |
| `app/services/kb_store.py` | SQLite KB store | Modify: loop-var rename + `cast` on `lastrowid` (4 → 0) |
| `app/api/v1/search.py` | v1 search endpoint | Modify: narrow `dict[str, object]` hit fields + explicit `sources` list (5 → 0) |
| `app/services/vectorstore.py` | Vector search + in-memory fallback | Modify: hoist `isinstance` narrowing of `meta` (8 → 0) |
| `app/api/v1/users.py` | v1 user CRUD | Modify: `UserRole(...)` coercion + `cast` (5 → 0) |
| `tests/test_v1_users_endpoint.py` | Characterization test for `create_user` | Create |
| `scripts/check_mypy_ratchet.py` | CI ratchet gate | Create |
| `.github/workflows/ci.yml` | CI pipeline | Modify: add non-advisory ratchet step after the advisory MyPy step (line 204–206) |

No `backend/**` changes (so no `legacy-path-approved` label needed). Deferred (NOT in this plan, with reasons in the spec): `app/models/__init__.py` (optional-import fallback family), `app/api/kb_mvp.py` (async-generator typing).

**Verified baseline (cross-checked, stable):** `py -3 -m mypy app` → **225 errors in 45 files**. The 24 clean-set errors and their codes:

```
app/services/vectorstore.py:162,167,172,177,180,182,191,193  [attr-defined] "object" has no attribute "get"   (8)
app/services/kb_store.py:407,428                             [assignment] int | None vs int                    (2)
app/services/kb_store.py:427,775                             [arg-type] int(int | None)                        (2)
app/llm/lora_runtime.py:128                                  [call-overload] int(object)                        (1)
app/llm/lora_runtime.py:133                                  [arg-type] float(object)                           (1)
app/api/v1/users.py:27,80                                    [arg-type] email str | None vs str                (2)
app/api/v1/users.py:29,82                                    [arg-type] role str vs UserRole                    (2)
app/api/v1/users.py:76                                       [arg-type] user_id int | None vs str | None        (1)
app/api/v1/search.py:68                                      [arg-type] file object vs str | None               (1)
app/api/v1/search.py:69                                      [arg-type] page object vs int | None               (1)
app/api/v1/search.py:70                                      [arg-type] float(object)                           (1)
app/api/v1/search.py:71                                      [arg-type] text object vs str                      (1)
app/api/v1/search.py:96                                      [misc] List[dict] vs List[SearchHit]               (1)
```

---

## Task 0: Sanity baseline

**Files:** none.

- [ ] **Step 1: Confirm the branch and clean tree**

```powershell
git rev-parse --abbrev-ref HEAD   # expect: chore/mypy-ratchet-clean-set
git status --short                 # expect: empty (the spec is already committed)
```

- [ ] **Step 2: Capture the red mypy baseline for the clean-set**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'lora_runtime\.py|kb_store\.py|v1\\search\.py|services\\vectorstore\.py|v1\\users\.py'
```
Expected: the 24 error lines listed in the File Structure baseline above. Record the global total (`Found 225 errors in 45 files`).

- [ ] **Step 3: Confirm the affected behavioral tests pass before any change**

```powershell
py -3 -m pytest tests/test_services_vectorstore.py tests/test_services_files.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`. (Confirm pass via the exit code — piping pytest drops the final "N passed" line.)

---

## Task 1: `app/llm/lora_runtime.py` → 0 (JSON manifest is `Any`)

**Files:**
- Modify: `app/llm/lora_runtime.py` (import line 11; `_load_manifest` return annotation line 94)
- Verify against: `tests/test_api_lora.py` (and any `test_lora*`)

**Background — the 2 errors:** `_load_manifest` is annotated `-> dict[str, object]` (line 94), so `manifest.get("seq_len", 0)` is `object` and `int(object)` (L128) / `float(object)` (L133) fail. JSON values are genuinely dynamic; `json.load()` returns `Any`. Re-annotating the manifest as `dict[str, Any]` makes `.get()` return `Any`, which `int()`/`float()` accept. **Zero runtime change** (annotation-only); the only consumer is `_iter_manifests`, whose other accesses use `str(...)`, which accepts `Any`. Behavior on a malformed value is unchanged: `int("abc")` still raises `ValueError`, still caught at L137 → `RegistryError`.

- [ ] **Step 1: Add `Any` to the typing import**

Change line 11:
```python
from typing import Any, Iterable
```
(was `from typing import Iterable`)

- [ ] **Step 2: Re-annotate `_load_manifest`**

Change line 94:
```python
def _load_manifest(path: Path) -> dict[str, Any]:
```
(was `def _load_manifest(path: Path) -> dict[str, object]:`)

- [ ] **Step 3: Verify lora_runtime.py reports 0 mypy errors**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'lora_runtime\.py'
```
Expected: **no output**. If any line appears, STOP and re-read the file before proceeding.

- [ ] **Step 4: Verify behavior unchanged**

```powershell
py -3 -m pytest tests/test_api_lora.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`. (If `tests/test_api_lora.py` does not exist in this checkout, substitute the lora test module found via `Get-ChildItem tests -Filter '*lora*'`.)

- [ ] **Step 5: Commit**

```powershell
git add app/llm/lora_runtime.py
git commit -F - <<'MSG'
chore(mypy): type LoRA JSON manifest as dict[str, Any]

_load_manifest returns json.load() (Any); annotating it dict[str, object]
forced int(object)/float(object) errors at lora_runtime.py:128/133. JSON
values are genuinely dynamic, so dict[str, Any] is the honest type and
int()/float() accept Any. Annotation-only; runtime and RegistryError-on-bad-
value behavior unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
```
(On Windows PowerShell, use a here-string file or `git commit -m "..."` — do NOT paste `@'...'@` into a bash heredoc.)

---

## Task 2: `app/services/kb_store.py` → 0 (loop-var reuse + `lastrowid`)

**Files:**
- Modify: `app/services/kb_store.py` (typing import line 11; lines 387–390, 427, 775)
- Verify against: `tests/test_services_kb_store.py` / `tests/test_kb_store*.py`

**Background — the 4 errors:**
- **L407, L428 `[assignment]`:** the name `page_no` is first bound at L387 `for page_no, page_text in pages:` where `pages: Optional[Sequence[tuple[int, str]]]` → mypy pins `page_no` as `int`. Later loops at L407 (`for page_no, page_text in normalised:`, `normalised: list[tuple[Optional[int], str]]`) and L428 (`for idx, ((page_no, chunk), blob) in …`, `chunks_with_pages: list[tuple[Optional[int], str]]`) try to bind `int | None` to the already-`int` name. Renaming the L387 binding frees `page_no` so its first binding becomes the `Optional[int]` one. The value flows into a nullable SQL column (`page_number`, L435), so `Optional[int]` is correct.
- **L427, L775 `[arg-type]`:** `int(cur.lastrowid)` where `sqlite3.Cursor.lastrowid` is `int | None`. After a successful `INSERT`, `lastrowid` is always a non-`None` int. Use `cast(int, …)` — zero runtime effect (preserves the existing `int()` call and the exact `TypeError`-on-None behavior).

- [ ] **Step 1: Add `cast` to the typing import**

Change line 25 (the existing `typing` import — NOT line 11, which is `from __future__ import annotations`):
```python
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, cast
```
(was `from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple`)

- [ ] **Step 2: Rename the input-loop variable (lines 387–390)**

Replace:
```python
            for page_no, page_text in pages:
                cleaned_page = (page_text or "").strip()
                if cleaned_page:
                    normalised.append((int(page_no), cleaned_page))
```
with:
```python
            for raw_page_no, page_text in pages:
                cleaned_page = (page_text or "").strip()
                if cleaned_page:
                    normalised.append((int(raw_page_no), cleaned_page))
```

- [ ] **Step 3: `cast` the two `lastrowid` reads (lines 427 and 775)**

Line 427:
```python
            doc_id = int(cast(int, cur.lastrowid))
```
(was `doc_id = int(cur.lastrowid)`)

Line 775:
```python
            msg_id = int(cast(int, cur.lastrowid))
```
(was `msg_id = int(cur.lastrowid)`)

- [ ] **Step 4: Verify kb_store.py reports 0 mypy errors**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'kb_store\.py'
```
Expected: **no output**.

- [ ] **Step 5: Verify behavior unchanged**

```powershell
py -3 -m pytest -k "kb_store" -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 6: Commit**

```powershell
git add app/services/kb_store.py
git commit -m "chore(mypy): drive app/services/kb_store.py to zero" -m "Rename the pages= input loop var (raw_page_no) so per-page page_no first-binds as Optional[int] (clears L407/L428 [assignment]); cast(int, cur.lastrowid) for the two INSERT id reads (clears L427/L775 [arg-type]). Type-only; runtime SQL and behavior unchanged."
```

---

## Task 3: `app/api/v1/search.py` → 0 (narrow `dict[str, object]` hits)

**Files:**
- Modify: `app/api/v1/search.py` (add a typing import; lines 66–72 and 96)
- Verify against: `tests/test_api_v1_search.py`

**Background — the 5 errors:** `search()` returns `List[dict[str, object]]`, so each `item.get(...)` is `object`. `SearchHit` expects `file: Optional[str]`, `page: Optional[int]`, `score: float`, `text: str` (L68–71 `[arg-type]`). Separately, L96 builds `[item.model_dump() for item in models]` (a `list[dict]`) inside an `and/or` expression whose inferred type clashes (`[misc]`). Casting the hit fields fixes L68–71; rewriting the `and/or` as an explicit conditional makes the `sources` list unambiguously `list[dict]`.

⚠️ **L96 is the one fix not experimentally pre-verified this session.** Step 5 (mypy → 0) is its gate: if a residual `[misc]` remains at L96 after the rewrite, re-read the surrounding lines and the `write_rag_run` call before adjusting — do not add a blanket ignore.

- [ ] **Step 1: Add the typing import**

After the `from __future__ import annotations` line (line 3), add:
```python
from typing import cast
```

- [ ] **Step 2: Narrow the `SearchHit` field sources (lines 66–72)**

Replace:
```python
    models = [
        SearchHit(
            file=item.get("file"),
            page=item.get("page"),
            score=float(item.get("score", 0.0)),
            text=item.get("text", ""),
        )
        for item in hits
    ]
```
with:
```python
    models = [
        SearchHit(
            file=cast("str | None", item.get("file")),
            page=cast("int | None", item.get("page")),
            score=float(cast("float", item.get("score", 0.0))),
            text=cast("str", item.get("text", "")),
        )
        for item in hits
    ]
```
(`cast` has no runtime effect; the values reaching Pydantic and the `float()` coercion are identical to before.)

- [ ] **Step 3: Make the `sources` list type explicit (line 96)**

Replace:
```python
                sources=models and [item.model_dump() for item in models] or [],
```
with:
```python
                sources=[item.model_dump() for item in models] if models else [],
```
(Behaviour-identical: empty `models` → `[]`; non-empty → the dumped dicts. `write_rag_run` is resolved via `getattr` and accepts any list.)

- [ ] **Step 4: Run the existing search tests (behavior pin)**

```powershell
py -3 -m pytest tests/test_api_v1_search.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 5: Verify search.py reports 0 mypy errors**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'v1\\search\.py'
```
Expected: **no output**. If a `[misc]` at L96 persists, STOP and re-read lines 89–98 plus the `write_rag_run` signature before changing the fix.

- [ ] **Step 6: Commit**

```powershell
git add app/api/v1/search.py
git commit -m "chore(mypy): drive app/api/v1/search.py to zero" -m "Cast the dict[str, object] hit fields to SearchHit's expected types (L68-71) and replace the models and [...] or [] idiom with an explicit conditional so sources is unambiguously list[dict] (L96). Type-only; same values reach Pydantic and write_rag_run."
```

---

## Task 4: `app/services/vectorstore.py` → 0 (hoist `meta` narrowing)

**Files:**
- Modify: `app/services/vectorstore.py` (line 159)
- Verify against: `tests/test_services_vectorstore.py`

**Background — the 8 errors:** the in-memory fallback search does
`meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}` (line 159). mypy cannot narrow `meta` because the `isinstance`-checked expression (`chunk.get("meta")`, call #1) is textually distinct from the assigned expression (call #2) — narrowing only flows through the *same* expression. So `meta` stays `object` and every later `meta.get(...)` (L162, 167, 172, 177, 180, 182, 191, 193) is `[attr-defined]`. Hoisting to a local both narrows the type and removes a redundant double `.get` call. **Runtime behavior identical** (same key, same default `{}`).

- [ ] **Step 1: Hoist the `meta` lookup (line 159)**

Replace:
```python
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
```
with:
```python
        raw_meta = chunk.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
```

- [ ] **Step 2: Verify vectorstore.py reports 0 mypy errors**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'services\\vectorstore\.py'
```
Expected: **no output** (all 8 `[attr-defined]` cleared).

- [ ] **Step 3: Verify behavior unchanged**

```powershell
py -3 -m pytest tests/test_services_vectorstore.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 4: Commit**

```powershell
git add app/services/vectorstore.py
git commit -m "chore(mypy): drive app/services/vectorstore.py to zero" -m "Hoist chunk.get('meta') into a local so isinstance() narrows meta to dict, clearing the 8 [attr-defined] 'object has no attribute get' errors in the in-memory fallback search (L162-193). Removes a redundant double .get; runtime behavior identical."
```

---

## Task 5: `app/api/v1/users.py` → 0 (+ characterization test)

**Files:**
- Create: `tests/test_v1_users_endpoint.py`
- Modify: `app/api/v1/users.py` (add typing import; lines 26–29, 76, 79–82)
- Pattern reference: `tests/test_admin_audit_endpoint.py`

**Background — the 5 errors:** `UserRecord` has `id: Optional[int]`, `email: Optional[str]`, `role: str`. The response/log sinks want non-optional types: `UserResponse.email: str` (L27, L80 `[arg-type]`), `UserResponse.role: UserRole` (L29, L82), `log_security_event(user_id: str | None)` (L76, where `record.id` is `int | None`). All three are coerced safely:
- `role`: `UserRole(user.role)` — explicit version of the str→enum coercion Pydantic already does (`UserRole` is `str, Enum`); same result for a valid role string.
- `email`: `cast(str, …)` — zero runtime effect; if `email` were `None` the `UserResponse` validation would still raise exactly as today.
- `user_id`: `cast("str | None", record.id)` — preserves the current behavior of logging the raw id value; annotation-only.

The characterization test pins `create_user`'s response shape on the *current* code first (so it must pass before the edits), guarding the one caller-visible refactor.

- [ ] **Step 1: Write the characterization test**

Create `tests/test_v1_users_endpoint.py`:
```python
"""Characterization test for the v1 create_user endpoint.

Pins the UserResponse shape so the mypy-cleanup refactor in app/api/v1/users.py
(UserRole coercion + casts) provably does not change caller-visible behavior.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api.v1.users import router
from app.models.entities import TenantRecord


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def app(engine):
    from app.core.auth import require_admin_user
    from app.core.deps import get_ingest_session

    fastapi_app = FastAPI()
    fastapi_app.include_router(router)

    def fake_admin():
        return {"id": "test-admin", "role": "admin"}

    def fake_session():
        with Session(engine) as s:
            yield s

    fastapi_app.dependency_overrides[require_admin_user] = fake_admin
    fastapi_app.dependency_overrides[get_ingest_session] = fake_session
    return fastapi_app


def test_create_user_returns_expected_response(app, engine):
    with Session(engine) as s:
        s.add(TenantRecord(tenant_id="acme", slug="acme", name="Acme Inc"))
        s.commit()

    client = TestClient(app)
    resp = client.post(
        "/users",
        json={
            "email": "alice@example.com",
            "full_name": "Alice Example",
            "password": "hunter2hunter2",
            "role": "manager",
            "is_active": True,
            "tenant_slug": "acme",
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert body["full_name"] == "Alice Example"
    assert body["role"] == "manager"
    assert body["is_active"] is True
    assert body["tenant_slug"] == "acme"
    assert isinstance(body["id"], int) and body["id"] >= 1
    # Password must never be echoed back.
    assert "password" not in body and "hashed_password" not in body
```

- [ ] **Step 2: Run the characterization test on CURRENT code — expect PASS**

```powershell
py -3 -m pytest tests/test_v1_users_endpoint.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0` (it characterizes existing behavior — must pass *before* the refactor). If it fails for an environment reason (e.g. `TenantRecord` requires extra non-null fields in this checkout), fix the test fixture to satisfy the real schema (read `app/models/entities.py`) until it passes on unmodified app code — do NOT modify `users.py` yet.

- [ ] **Step 3: Add the typing import to users.py**

After `from __future__ import annotations` (line 3), add:
```python
from typing import cast
```

- [ ] **Step 4: Coerce the three sinks**

In `list_users` (lines 26–29), replace:
```python
            id=user.id or 0,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
```
with:
```python
            id=user.id or 0,
            email=cast(str, user.email),
            full_name=user.full_name,
            role=UserRole(user.role),
```

In `create_user`, replace line 76:
```python
        log_security_event("role_change", user_id=cast("str | None", record.id), new_role=str(record.role))
```
(was `… user_id=record.id …`)

In `create_user`'s response (lines 79–82), replace:
```python
        id=record.id or 0,
        email=record.email,
        full_name=record.full_name,
        role=record.role,
```
with:
```python
        id=record.id or 0,
        email=cast(str, record.email),
        full_name=record.full_name,
        role=UserRole(record.role),
```

- [ ] **Step 5: Verify users.py reports 0 mypy errors**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'v1\\users\.py'
```
Expected: **no output**.

- [ ] **Step 6: Verify behavior unchanged (characterization test still green)**

```powershell
py -3 -m pytest tests/test_v1_users_endpoint.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 7: Commit**

```powershell
git add app/api/v1/users.py tests/test_v1_users_endpoint.py
git commit -m "chore(mypy): drive app/api/v1/users.py to zero" -m "Coerce UserRecord (id Optional[int], email Optional[str], role str) into the UserResponse/log_security_event sinks via UserRole(...) and casts. Adds a characterization test pinning create_user's response shape. Type-only; same values, same validation behavior."
```

---

## Task 6: CI ratchet gate

**Files:**
- Create: `scripts/check_mypy_ratchet.py`
- Modify: `.github/workflows/ci.yml` (insert a step after the advisory MyPy step at lines 204–206)
- Test: `tests/test_check_mypy_ratchet.py`

**Background:** CI runs `mypy .` with `continue-on-error: true` (`.github/workflows/ci.yml:204-206`) — advisory only, so a cleaned file can silently regress. This gate runs mypy, parses the `path:line: error:` lines, and exits non-zero if any file in an explicit `CLEAN_FILES` list has >0 errors. It asserts nothing about other files, so the 200-error baseline never fails CI.

- [ ] **Step 1: Write the ratchet script test**

Create `tests/test_check_mypy_ratchet.py`:
```python
"""Unit tests for the mypy ratchet gate's parsing/decision logic."""

from __future__ import annotations

from scripts.check_mypy_ratchet import CLEAN_FILES, offending_files


def test_clean_files_list_is_nonempty_and_normalised():
    assert CLEAN_FILES, "CLEAN_FILES must not be empty"
    for path in CLEAN_FILES:
        assert path == path.replace("\\", "/"), f"{path} must use forward slashes"
        assert path.endswith(".py")


def test_offending_files_flags_a_clean_file_with_errors():
    sample = (
        "app/core/deps.py:10: error: boom  [arg-type]\n"
        "app/other/file.py:3: error: ignore me  [misc]\n"
    )
    result = offending_files(sample, clean_files=["app/core/deps.py"])
    assert result == {"app/core/deps.py": 1}


def test_offending_files_ignores_non_clean_files():
    sample = "app/other/file.py:3: error: ignore me  [misc]\n"
    result = offending_files(sample, clean_files=["app/core/deps.py"])
    assert result == {}


def test_offending_files_handles_windows_separators_in_mypy_output():
    sample = "app\\core\\deps.py:10: error: boom  [arg-type]\n"
    result = offending_files(sample, clean_files=["app/core/deps.py"])
    assert result == {"app/core/deps.py": 1}
```

- [ ] **Step 2: Run the test — expect failure (module missing)**

```powershell
py -3 -m pytest tests/test_check_mypy_ratchet.py -q --ignore=backend
$LASTEXITCODE
```
Expected: non-zero (`ModuleNotFoundError: scripts.check_mypy_ratchet`).

- [ ] **Step 3: Write the ratchet script**

Create `scripts/check_mypy_ratchet.py`:
```python
"""Fail CI if any file proven mypy-clean has regressed to >0 errors.

This is a *ratchet*: it asserts only about files listed in CLEAN_FILES, so the
repo's overall mypy baseline can keep shrinking independently. Each drive-to-zero
pass appends the files it cleaned to CLEAN_FILES in the same PR.

Usage:  py -3 scripts/check_mypy_ratchet.py
Exit 0 if all CLEAN_FILES report 0 errors; exit 1 (with a report) otherwise.
"""

from __future__ import annotations

import subprocess
import sys

# Files proven to report 0 mypy errors. Append to this list as more are cleaned.
CLEAN_FILES: list[str] = [
    # PR #567 (deps + file_stats safe pass)
    "app/core/deps.py",
    "app/services/file_stats.py",
    # This pass (object/None-narrowing clean-set)
    "app/llm/lora_runtime.py",
    "app/services/kb_store.py",
    "app/api/v1/search.py",
    "app/services/vectorstore.py",
    "app/api/v1/users.py",
]


def offending_files(mypy_output: str, clean_files: list[str]) -> dict[str, int]:
    """Return {clean_file: error_count} for clean files with >0 errors."""
    wanted = set(clean_files)
    counts: dict[str, int] = {}
    for line in mypy_output.splitlines():
        if ": error:" not in line:
            continue
        path = line.split(":", 1)[0].replace("\\", "/")
        if path in wanted:
            counts[path] = counts.get(path, 0) + 1
    return counts


def run_mypy() -> str:
    """Run the repo's configured mypy over app/ and return combined output."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "app"],
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr


def main() -> int:
    output = run_mypy()
    offenders = offending_files(output, CLEAN_FILES)
    if offenders:
        print("mypy ratchet FAILED — these clean files regressed:")
        for path, count in sorted(offenders.items()):
            print(f"  {path}: {count} error(s)")
        print("\nFix the new errors or, if intentional, remove the file from")
        print("CLEAN_FILES in scripts/check_mypy_ratchet.py (discouraged).")
        return 1
    print(f"mypy ratchet OK — all {len(CLEAN_FILES)} clean files report 0 errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit test — expect PASS**

```powershell
py -3 -m pytest tests/test_check_mypy_ratchet.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 5: Run the gate end-to-end — expect PASS (all 7 files clean after Tasks 1–5)**

```powershell
py -3 scripts/check_mypy_ratchet.py
$LASTEXITCODE
```
Expected: `mypy ratchet OK — all 7 clean files report 0 errors.` and `$LASTEXITCODE` is `0`.
(If it fails, a prior task's file is not actually at 0 — return to that task; do NOT remove the file from `CLEAN_FILES`.)

- [ ] **Step 6: Wire the gate into CI**

In `.github/workflows/ci.yml`, immediately after the existing MyPy step (lines 204–206):
```yaml
      - name: MyPy (types)
        continue-on-error: true
        run: mypy .
```
add (same indentation, leave the advisory step unchanged):
```yaml

      - name: MyPy ratchet (clean-file regression gate)
        run: python scripts/check_mypy_ratchet.py
```

- [ ] **Step 7: Commit**

```powershell
git add scripts/check_mypy_ratchet.py tests/test_check_mypy_ratchet.py .github/workflows/ci.yml
git commit -m "chore(ci): add mypy ratchet gate for clean files" -m "scripts/check_mypy_ratchet.py fails CI if any file in CLEAN_FILES (deps, file_stats + this pass's 5 files) regresses above 0 mypy errors. Added as a non-advisory CI step alongside the existing advisory 'mypy .'. Closes the continue-on-error gap for already-clean files."
```

---

## Task 7: Whole-suite verification

**Files:** none.

- [ ] **Step 1: Confirm the seven files are at zero and the global count dropped**

```powershell
py -3 -m mypy app 2>&1 | Select-String 'deps\.py|file_stats\.py|lora_runtime\.py|kb_store\.py|v1\\search\.py|services\\vectorstore\.py|v1\\users\.py'
```
Expected: **no output**.

```powershell
py -3 -m mypy app 2>&1 | Select-String 'Found \d+ errors'
```
Expected: a total **noticeably below 225** (≈201; exact number confirmed here). Record it.

- [ ] **Step 2: Run the affected test slice**

```powershell
py -3 -m pytest tests/test_services_vectorstore.py tests/test_services_files.py tests/test_api_v1_search.py tests/test_v1_users_endpoint.py tests/test_check_mypy_ratchet.py -k "" -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 3: Broader regression check (mirror CI: skip Postgres-marked, ignore backend)**

```powershell
py -3 -m pytest -q -m "not requires_postgres" --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0` (the repo's ~6 intentional skips are fine).

- [ ] **Step 4: Lint and format gates**

```powershell
py -3 -m ruff check .
py -3 -m black --check app/llm/lora_runtime.py app/services/kb_store.py app/api/v1/search.py app/services/vectorstore.py app/api/v1/users.py tests/test_v1_users_endpoint.py scripts/check_mypy_ratchet.py tests/test_check_mypy_ratchet.py
```
Expected: both clean. If `black --check` flags a file, run `py -3 -m black <file>` and amend that file's commit.

- [ ] **Step 5: Finish the branch**

Use the **superpowers:finishing-a-development-branch** skill. PR title: `chore(mypy): ratchet clean-set (5 files to zero) + CI regression gate`. The PR body links this plan and the spec, states the verified before/after global mypy totals, and notes the explicitly-deferred follow-ups (`models/__init__.py` guarded pass, `kb_mvp.py` async-generator typing).

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** spec §2 clean-set (5 files) → Tasks 1–5; §3 fix shapes → the per-task Background + edits; §4 CI ratchet gate → Task 6; §5 verification (whole-package mypy, behavior tests, characterization test, ratchet run, ruff/black) → Tasks 0–7; §6 risks (kb_store `or 0` ambiguity → resolved via `cast`, not `or 0`; users.py output → characterization test) → Tasks 2 & 5; §7 deliverable → Task 7 Step 5. Deferred set (`models/__init__.py`, `kb_mvp.py`) is excluded as specified.
- **Placeholder scan:** every code step shows exact old→new text; every command shows expected output. The two conditional instructions (Task 1 Step 4 lora test name; Task 5 Step 2 fixture schema) are explicit recovery procedures, not "TODO"s.
- **Type/name consistency:** `cast` imported before use in `kb_store.py`, `search.py`, `users.py`; `Any` added in `lora_runtime.py`; `raw_page_no`/`raw_meta` introduced and used locally; `CLEAN_FILES`/`offending_files` names match between `scripts/check_mypy_ratchet.py` and `tests/test_check_mypy_ratchet.py`; the 7 seeded files match Tasks 1–5 plus PR #567's two.
- **Trust caveat (session-specific):** authored under intermittent tool-output corruption; all line numbers/types are cross-verified, but the executor MUST treat each `mypy <file>` → 0 step as a hard gate and re-read on any mismatch. `search.py:96` is the single fix not experimentally pre-verified.
```
