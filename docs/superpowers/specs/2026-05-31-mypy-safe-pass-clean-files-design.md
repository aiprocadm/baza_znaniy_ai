# Code Health — Safe mypy Pass: `deps.py` + `file_stats.py` (Design)

**Date:** 2026-05-31
**Author scope:** technical design for a zero-behavioral-change typing cleanup
**Status:** Design document. Subordinate to repo conventions in `CLAUDE.md` and `CONTRIBUTING.md`.

> Этот документ описывает безопасный «drive-to-zero» проход по двум модулям с
> чистым типовым долгом плюс удаление одной заведомо мёртвой ветки совместимости.
> Цель — измеримое падение mypy-baseline **без единого изменения поведения** и
> повторяемый шаблон для будущих проходов по остальным ~225 ошибкам.

---

## 1. Context

`py -3 -m mypy app` reports a pre-existing baseline of **244 errors across 48
files** (config: `[tool.mypy]` in `pyproject.toml`; `backend/` excluded). Per
`CLAUDE.md`, a change is judged by *new errors on touched lines*, not the global
total, and the correct unit of cleanup is **whole-file drive-to-zero** — a
partially-fixed file just re-accrues noise on the next edit.

Profiling the baseline shows the debt is concentrated in a handful of files, and
that the three largest each hide an **intentional, load-bearing fallback** that
must NOT be deleted as "dead code":

| File | mypy errors | Nature of the errors | This pass? |
|---|---|---|---|
| `app/core/config.py` | ~68 diag lines | Hand-rolled `BaseSettings` fallback for installs *without* `pydantic-settings` (live on light/MVP installs — `pydantic-settings` is absent from every `requirements*.txt`) **plus** a dead Pydantic-v1 decorator fallback | **Partial** — only the dead v1 decorator fallback removed |
| `app/ingest/chunking.py` | ~9 errors | `try/except ImportError` graceful-degradation guards for optional parsers (`fitz`, `pdfminer`, `pypdf`, `openpyxl`, `pptx`, `tiktoken`, `PIL`…) | **No** — deferred to a guarded pass |
| `app/core/deps.py` | ~11 errors | Pure typing hygiene (FastAPI dependency defaults, env parsing) | **Yes** |
| `app/services/file_stats.py` | ~11 errors | SQLModel query-building typing (the `col()` gap) | **Yes** |

Only `deps.py` and `file_stats.py` are pure typing debt with no behavioral
subtlety. They are the safe, high-signal targets for the first pass.

## 2. Goal & non-goals

**Goal.** Drive `app/core/deps.py` and `app/services/file_stats.py` to **0 mypy
errors**, and delete the provably-dead Pydantic-v1 decorator fallback in
`app/core/config.py`. Zero behavioral change. Establish a repeatable per-file
drive-to-zero template for subsequent guarded passes.

**Non-goals (explicitly deferred — binding for this PR):**

- The `config.py` `BaseSettings` dual-branch (the live light-install path). A
  later pass must first add a guard test that pins *"Settings constructs with
  `pydantic-settings` absent"* before any typing rework there.
- `chunking.py` optional-dependency fallbacks (same family as the above).
- Removing the `app/qdrant_client.py` / `app/models/qdrant_client.py` shims —
  still covered by `tests/test_qdrant_client.py` and used by
  `tests/test_worker_main.py`. `CLAUDE.md` freezes *new* imports of them; it does
  not mandate deletion.
- The `srv/projects/kb/` legacy tree — pinned by `tests/test_chat_formatting.py`
  and `tests/test_memory_env_vars.py`; treat like `backend/` (intentional legacy).
- Reducing the *global* error count beyond the touched files. Expected delta:
  ~244 → ~225.

## 3. Per-file changes

### 3.1 `app/core/config.py` — remove the dead Pydantic-v1 decorator fallback

Current `config.py` lines 22–45 are a `try/except ImportError` block that defines
no-op fallback decorators (`computed_field`, `field_validator`, `model_validator`)
for the Pydantic-v1 case.

**Change.** Collapse to the direct import; delete the `except` body:

```python
from pydantic import BaseModel, computed_field, field_validator, model_validator
```

**Why this is safe.** `pydantic~=2.11` (`requirements-runtime.txt`) and
`pydantic==2.11.10` (`requirements.txt`) are hard-pinned. All four symbols exist
in Pydantic v2; the `except ImportError` branch is unreachable in every supported
install (Pydantic v1 never had `computed_field`).

**Do NOT touch in this pass:**

- The `importlib_metadata` shim (lines 13–16) — a real Python-3.10 concern.
- The `pydantic_settings` try/except and the hand-rolled `BaseSettings` class
  (lines 48–52 and 250–305). That fallback is the **live code path** on installs
  that do not have `pydantic-settings` (it is in no requirements file).

**Errors cleared.** The `computed_field` / `field_validator` / `model_validator`
`[no-redef]` errors. The remaining `config.py` errors (`[arg-type]`,
`[prop-decorator]`, `[call-overload]` in the live `BaseSettings` branch) are out
of scope and remain in the baseline.

### 3.2 `app/core/deps.py` → 0 errors

| Location | mypy error | Fix |
|---|---|---|
| `get_tenant`, `get_ingest_service`, `get_ingest_session`, `get_file_store`, `get_ingest_queue`, `get_lora_manager` (the `request: Request = None` params) | `[assignment]` "Incompatible default `None` for argument of type `Request`" (×6) | Annotate `request: Request \| None = None`. Runtime already guards every use with `if request is None` / `if request and …`, so this only makes the signature honest. |
| `UploadLimits._normalise_max_mb`, `_bytes_to_mb` (`int(float(value))`) | `[arg-type]` "`float` has incompatible type `object`" (×2) | Narrow the helper input to `str \| int \| float`. This shifts the obligation to the env-var call sites in `get_upload_limits`, where mypy does **not** narrow `x not in {None, ""}`: switch those guards to a truthiness check (`if raw_max_mb:` / `if legacy:`) so `str \| None` narrows to `str` before the call (behaviour-identical for strings). Keep the defensive `except (TypeError, ValueError)`. |
| `get_upload_limits` call to `UploadLimits(allowed_extensions=extensions)` | `[arg-type]` "`str \| None` vs `set[str]`" | Normalise at the call site: pass `UploadLimits._normalise_extensions(extensions)` (returns `set[str]`). `__post_init__` re-runs `_normalise_extensions`, which is idempotent on a `set` — no behavior change. |
| A `str \| None` value reaching `.strip()` (header/env tenant resolution in `get_tenant`) | `[union-attr]` / `[str]` | Add an explicit None-guard or default before `.strip()`; confirm the exact site against mypy output during implementation. |

Every change widens or clarifies a type; none narrows what is accepted at runtime.

### 3.3 `app/services/file_stats.py` → 0 errors

**Root cause.** SQLModel exposes model attributes as their Python type, so
`FileRecord.tenant_id == tenant_id` is inferred as `bool` (not
`ColumnElement[bool]`) and `func.count(FileRecord.id)` passes `int | None`. This
produces the `[bool]`, `[arg-type]`, and `[call-overload]` cluster.

**Fix.** Wrap model attributes in SQLModel's `col()` helper (confirmed importable
in the pinned `sqlmodel 0.0.25`): `from sqlmodel import Session, col`, then:

```python
func.count(col(FileRecord.id))
func.coalesce(func.sum(col(FileRecord.size)), 0)
func.coalesce(func.sum(col(FileRecord.chunks)), 0)
... .where(col(FileRecord.tenant_id) == tenant_id)
... .group_by(col(FileRecord.status))
func.min(col(FileRecord.created_at)), func.max(col(FileRecord.created_at))
```

- Annotate `timestamp_stmt` to clear `[var-annotated]`.
- For the multi-column `session.exec(<select>)` `[call-overload]`: prefer
  `sqlmodel.select`. If the overload for a multi-column tuple select still cannot
  be satisfied without contortion, apply a **single** targeted
  `# type: ignore[call-overload]` with a one-line justification (documented
  SQLModel limitation) — never a blanket file-level ignore.

## 4. Verification

1. **mypy (whole package).** `py -3 -m mypy app`. Acceptance: `app/core/deps.py`
   and `app/services/file_stats.py` report **0 errors**, and the three
   v1-decorator `[no-redef]` errors in `config.py` are gone; global total drops
   (~244 → ~225). A single-file check (`mypy app/foo.py`) is NOT used — per
   `CLAUDE.md` it under-reports because it follows only that file's imports.
2. **Behavior-preservation.** `py -3 -m pytest -k "deps or file_stat or config or
   upload or tenant" -q --ignore=backend`. Confirm pass via exit code
   (`$LASTEXITCODE` / `EXIT=$?`) — piping pytest drops the final "N passed" line
   (repo gotcha).
3. **Lint/format.** `py -3 -m ruff check .` and `py -3 -m black --check .` clean.
4. **TDD-lite.** Add a characterization test only for a change observable by a
   caller. Sole candidate: `UploadLimits(allowed_extensions="pdf,docx")` →
   `{"pdf", "docx"}` (guards the call-site normalisation in §3.2). The
   `Request | None` changes are strictly more permissive, so no new test.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Call-site extension normalisation subtly changes behavior | Characterization test (§4.4); `_normalise_extensions` is idempotent on a `set`. |
| Removing the v1 decorator fallback breaks an unforeseen install | None supported: all `requirements*.txt` pin Pydantic v2, which always has these symbols; v1 never had `computed_field`. |
| `col()` not satisfying the multi-column `exec()` overload | Documented as acceptable: one justified `# type: ignore[call-overload]`, scoped to the single statement. |
| Scope creep into the deferred traps (`config` BaseSettings, `chunking`, shims, `srv/`) | §2 non-goals are binding for this PR. |

## 6. Deliverable

One PR off `chore/mypy-safe-pass-deps-filestats`, Conventional-Commit
`chore(mypy): …`, ≤400 LoC, no `backend/**` change (so no `legacy-path-approved`
label needed). The PR establishes the per-file drive-to-zero template for the
next, guarded passes (`config.py` BaseSettings branch, `chunking.py`, shim
retirement).

## 7. Open questions

None at design stage. The exact `exec()` overload resolution in §3.3 is settled
during the writing-plans / implementation phase; `col()` availability is already
confirmed (`sqlmodel 0.0.25`).
