# mypy Safe Pass (deps.py + file_stats.py) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive `app/core/deps.py` and `app/services/file_stats.py` to **0 mypy errors** and delete the provably-dead Pydantic-v1 decorator fallback in `app/core/config.py`, with **zero behavioral change**.

**Architecture:** Three independent, type-only edits. Each is verified by (a) the touched file reporting 0 mypy errors under the whole-package `mypy app` check, and (b) the existing behavioral tests for that module still passing. One new characterization test pins the single call-site behavior change (extension normalisation). The two load-bearing-fallback files (`config.py` `BaseSettings` branch, `chunking.py`) are explicitly out of scope.

**Tech Stack:** Python 3.13, mypy (config in `pyproject.toml` `[tool.mypy]`), pytest, SQLModel/SQLAlchemy, Pydantic v2. Windows: invoke via the `py -3` launcher (no venv).

**Spec:** `docs/superpowers/specs/2026-05-31-mypy-safe-pass-clean-files-design.md`
**Branch:** `chore/mypy-safe-pass-deps-filestats` (already created off `main`).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `app/core/config.py` | App settings + helpers | Modify: delete dead v1 decorator fallback (lines 22–45) |
| `app/core/deps.py` | Shared FastAPI dependencies | Modify: typing fixes (10 errors → 0) |
| `app/services/file_stats.py` | Per-tenant file aggregation | Modify: `col()`-wrap query (cluster → 0) |
| `tests/test_core_deps.py` | Unit tests for `deps.py` | Modify: add one characterization test |

No new files. No `backend/**` changes (so no `legacy-path-approved` label needed).

---

## Task 1: Remove the dead Pydantic-v1 decorator fallback in `config.py`

**Files:**
- Modify: `app/core/config.py:22-45`
- Verify against: `tests/test_config_env_aliases.py`, `tests/test_config_flatten_aliases.py`, `tests/test_config_version_info.py`

**Background:** `pydantic~=2.11` / `pydantic==2.11.10` are hard-pinned, so `computed_field`, `field_validator`, `model_validator` always import successfully. The `except ImportError` body defines no-op fallbacks that never run, and mypy flags them as `[no-redef]` at lines 27/35/41. Removing the fallback is dead-code deletion — it cannot change runtime behavior. **Do NOT touch** the `pydantic_settings` try/except or the hand-rolled `BaseSettings` class (lines 48–52, 250–305): that is the live light-install code path.

- [ ] **Step 1: Capture the red baseline**

Run:
```powershell
py -3 -m mypy app 2>&1 | Select-String 'config\.py:(27|35|41)'
```
Expected: three `[no-redef]` lines for `computed_field` (27), `field_validator` (35), `model_validator` (41).

Then confirm the config tests are green before the change:
```powershell
py -3 -m pytest tests/test_config_env_aliases.py tests/test_config_flatten_aliases.py tests/test_config_version_info.py -q
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 2: Delete the fallback block**

In `app/core/config.py`, replace this entire block (lines 22–45):
```python
try:  # pragma: no cover - optional features may be unavailable in tests
    from pydantic import BaseModel, computed_field, field_validator, model_validator
except ImportError:  # pragma: no cover - provide light-weight fallbacks
    from pydantic import BaseModel  # type: ignore[assignment]

    def computed_field(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return property(func)

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    def field_validator(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

    def model_validator(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator
```
with this single line:
```python
from pydantic import BaseModel, computed_field, field_validator, model_validator
```

- [ ] **Step 3: Verify the no-redef errors are gone (and nothing new appeared)**

Run:
```powershell
py -3 -m mypy app 2>&1 | Select-String 'config\.py:(27|35|41)'
```
Expected: **no output** (the three errors are cleared).

Confirm the still-out-of-scope errors remain (sanity — we did not accidentally touch the BaseSettings branch):
```powershell
py -3 -m mypy app 2>&1 | Select-String 'config\.py:(250|309|338|385)'
```
Expected: those `[no-redef]`/`[arg-type]` lines still present (unchanged — out of scope).

- [ ] **Step 4: Verify behavior unchanged**

Run:
```powershell
py -3 -m pytest tests/test_config_env_aliases.py tests/test_config_flatten_aliases.py tests/test_config_version_info.py -q
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 5: Commit**

```powershell
git add app/core/config.py
git commit -m @'
chore(mypy): drop dead Pydantic-v1 decorator fallback in config.py

pydantic~=2.11 is hard-pinned, so computed_field/field_validator/
model_validator always import; the except-ImportError fallback was dead
code flagged [no-redef] at lines 27/35/41. The pydantic_settings /
hand-rolled BaseSettings light-install path is untouched.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
'@
```

---

## Task 2: Drive `app/core/deps.py` to 0 mypy errors

**Files:**
- Modify: `app/core/deps.py` (lines 40, 50, 95, 99, 108, 111, 117, 121, 129, 137, 151, 165)
- Test: `tests/test_core_deps.py`

**Background — the 10 errors:** six `[assignment]` "implicit Optional" on `request: Request = None` (lines 111/121/129/137/151/165); two `[arg-type]` `float(object)` (lines 42/52); one `[arg-type]` `allowed_extensions` `str | None` vs `set[str]` (line 108); one `[union-attr]` `.strip()` on `str | None` (line 117). Every fix widens or clarifies a type; none narrows runtime acceptance.

- [ ] **Step 1: Write the characterization test (pins the one behavior we refactor)**

Append to `tests/test_core_deps.py`:
```python
def test_get_upload_limits_parses_explicit_extensions(
    deps_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit UPLOAD_ALLOWED_EXTS list parses into the expected set."""

    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("UPLOAD_MAX_SIZE", raising=False)
    monkeypatch.setenv("UPLOAD_ALLOWED_EXTS", "pdf,docx")

    limits = deps_module.get_upload_limits()

    assert limits.allowed_extensions == {"pdf", "docx"}
```

- [ ] **Step 2: Run the characterization test — expect PASS on current code**

Run:
```powershell
py -3 -m pytest tests/test_core_deps.py::test_get_upload_limits_parses_explicit_extensions -q
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0` (it characterizes existing behavior — must pass *before* the refactor).

- [ ] **Step 3: Apply the typing fixes**

Edit 3a — narrow the two numeric helpers (`app/core/deps.py` lines 40 and 50):
```python
    @staticmethod
    def _normalise_max_mb(value: str | int | float) -> int:
```
```python
    @staticmethod
    def _bytes_to_mb(value: str | int | float) -> int:
```

Edit 3b — narrow the env guards in `get_upload_limits` so `str | None` becomes `str` before those helper calls. Line 95:
```python
    if raw_max_mb:
```
(was `if raw_max_mb not in {None, ""}:`). Line 99:
```python
        if legacy:
```
(was `if legacy not in {None, ""}:`). These are behaviour-identical for `str | None` (both reject `None` and `""`).

Edit 3c — normalise extensions at the call site (line 108):
```python
    return UploadLimits(
        max_upload_mb=max_upload_mb,
        allowed_extensions=UploadLimits._normalise_extensions(extensions),
    )
```
(was `allowed_extensions=extensions`). `_normalise_extensions` returns `set[str]`; `__post_init__` re-normalises idempotently on a set — same result.

Edit 3d — make the six dependency signatures honestly Optional:
```python
def get_tenant(request: Request | None = None) -> str:
```
```python
def get_ingest_service(request: Request | None = None) -> IngestService:
```
```python
def get_ingest_session(request: Request | None = None) -> Iterator[Session]:
```
```python
def get_file_store(request: Request | None = None) -> FileStore:
```
```python
def get_ingest_queue(request: Request | None = None) -> IngestQueue:
```
```python
def get_lora_manager(request: Request | None = None) -> LlamaLoraManager:
```

Edit 3e — guarantee a non-`None` string before `.strip()` in `get_tenant` (line 117):
```python
    tenant = (header_value or os.getenv("DEFAULT_TENANT", "default") or "default").strip()
```
(was `(header_value or os.getenv("DEFAULT_TENANT", "default")).strip()`). The trailing `or "default"` makes the value `str`; line 118 `return tenant or "default"` already handled the empty case, so behavior is unchanged.

- [ ] **Step 4: Verify deps.py reports 0 mypy errors**

Run:
```powershell
py -3 -m mypy app 2>&1 | Select-String 'core\\deps\.py'
```
Expected: **no output** (all 10 errors cleared, no new ones).

- [ ] **Step 5: Verify behavior unchanged**

Run the full deps test module (includes the new characterization test):
```powershell
py -3 -m pytest tests/test_core_deps.py -q
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 6: Commit**

```powershell
git add app/core/deps.py tests/test_core_deps.py
git commit -m @'
chore(mypy): drive app/core/deps.py to zero mypy errors

Honest Optional on the six request= dependencies, narrowed numeric-helper
inputs (str|int|float) with truthiness env guards, call-site extension
normalisation, and a None-safe tenant .strip(). Type-only; adds a
characterization test pinning UPLOAD_ALLOWED_EXTS parsing.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
'@
```

---

## Task 3: Drive `app/services/file_stats.py` to 0 mypy errors via `col()`

**Files:**
- Modify: `app/services/file_stats.py` (import line 10; query block lines 54–81)
- Verify against: `tests/test_services_files.py::test_compute_file_stats_handles_mixed_statuses`

**Background:** SQLModel exposes model attributes as their Python type, so `FileRecord.tenant_id == tenant_id` is inferred as `bool` and `func.count(FileRecord.id)` passes `int | None` — producing the `select`/`exec` `[call-overload]`, the `where` `[arg-type]`, the `count` `[arg-type]`, and the `[var-annotated]` on `timestamp_stmt`. SQLModel's `col()` helper (confirmed importable in pinned `sqlmodel 0.0.25`) re-types a model attribute as a `ColumnElement`, which satisfies the `select()`/`func.*`/`where()` overloads and lets the statement types infer. Runtime SQL is identical (`col(x)` returns `x`).

- [ ] **Step 1: Confirm the behavior pin passes on current code**

Run:
```powershell
py -3 -m pytest tests/test_services_files.py::test_compute_file_stats_handles_mixed_statuses -q
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 2: Import `col`**

In `app/services/file_stats.py`, change line 10:
```python
from sqlmodel import Session, col
```
(was `from sqlmodel import Session`).

- [ ] **Step 3: Wrap model attributes in `col()` in both statements**

Replace the aggregation statement (lines 54–63):
```python
    aggregation_stmt = (
        select(
            col(FileRecord.status),
            func.count(col(FileRecord.id)),
            func.coalesce(func.sum(col(FileRecord.size)), 0),
            func.coalesce(func.sum(col(FileRecord.chunks)), 0),
        )
        .where(col(FileRecord.tenant_id) == tenant_id)
        .group_by(col(FileRecord.status))
    )
```

Replace the timestamp statement (lines 78–81):
```python
    timestamp_stmt = select(
        func.min(col(FileRecord.created_at)),
        func.max(col(FileRecord.created_at)),
    ).where(col(FileRecord.tenant_id) == tenant_id)
```

(Leave the loop, unpacking, `_normalise_timestamp`, and `FileStats` construction exactly as they are.)

- [ ] **Step 4: Verify file_stats.py reports 0 mypy errors**

Run:
```powershell
py -3 -m mypy app 2>&1 | Select-String 'file_stats\.py'
```
Expected: **no output**. (The `col()` wrapping makes `select()` infer `Select[tuple[...]]`, which also clears the `[var-annotated]` on `timestamp_stmt` — no explicit annotation needed. If a lone `[var-annotated]` somehow persists, add `timestamp_stmt: Select[...]` by importing `Select` from `sqlalchemy` and annotating with the inferred tuple type from the mypy note — do **not** use a blanket `# type: ignore`.)

- [ ] **Step 5: Verify behavior unchanged**

Run:
```powershell
py -3 -m pytest tests/test_services_files.py -q
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 6: Commit**

```powershell
git add app/services/file_stats.py
git commit -m @'
chore(mypy): drive app/services/file_stats.py to zero via sqlmodel.col()

Wrap FileRecord attributes in col() inside select/where/func.*/group_by so
the SQLAlchemy overloads match and the statement types infer. Runtime SQL
unchanged; existing compute_file_stats test still green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
'@
```

---

## Task 4: Whole-package verification

**Files:** none (verification only).

- [ ] **Step 1: Confirm the two target files are at zero and the global count dropped**

Run:
```powershell
py -3 -m mypy app 2>&1 | Select-String 'core\\deps\.py|file_stats\.py'
```
Expected: **no output**.

```powershell
py -3 -m mypy app 2>&1 | Select-String 'Found \d+ errors'
```
Expected: a "Found N errors in M files" line with **N noticeably below 244** (roughly ~222; the exact number depends on how many residual config.py lines remain). Compare against the pre-change baseline of `244 errors in 48 files`.

- [ ] **Step 2: Run the affected test slice**

Run:
```powershell
py -3 -m pytest tests/test_core_deps.py tests/test_services_files.py tests/test_config_env_aliases.py tests/test_config_flatten_aliases.py tests/test_config_version_info.py tests/test_api_v1_upload.py -q --ignore=backend
$LASTEXITCODE
```
Expected: `$LASTEXITCODE` is `0`.

- [ ] **Step 3: Lint and format gates**

Run:
```powershell
py -3 -m ruff check .
py -3 -m black --check app/core/config.py app/core/deps.py app/services/file_stats.py tests/test_core_deps.py
```
Expected: both report no issues. If `black --check` flags any of the four files, run `py -3 -m black <file>` and amend the relevant commit.

- [ ] **Step 4: Finish the branch**

Use the **superpowers:finishing-a-development-branch** skill to choose how to integrate (push + open PR, or merge). The PR title should be `chore(mypy): safe pass — deps.py + file_stats.py to zero` and the body should link the spec and note the explicitly-deferred follow-ups (config BaseSettings branch, chunking optional-dep guards, qdrant_client shim retirement).

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** §3.1 config → Task 1; §3.2 deps (all four error groups) → Task 2; §3.3 file_stats → Task 3; §4 verification (mypy whole-package, behavior tests, characterization test, ruff/black) → Tasks 1–4; §6 deliverable → Task 4 Step 4. No spec section is unmapped.
- **Placeholder scan:** every code step shows exact old→new text; every command shows expected output. The only conditional (Task 3 Step 4 fallback annotation) is fully specified, not a "TODO".
- **Type consistency:** `_normalise_extensions` / `_normalise_max_mb` / `_bytes_to_mb` signatures and the `col()` import name match across tasks; `request: Request | None` is applied uniformly to all six dependencies; error line numbers match the captured `mypy app` output.
