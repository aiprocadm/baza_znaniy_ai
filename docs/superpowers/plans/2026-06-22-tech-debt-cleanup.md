# Tech-Debt Cleanup — Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove accumulated technical debt — delete the deprecated `backend/`
path, pin the existing config-footgun warning with a test, make the intentional
dual-surface architecture obvious in-code, and shrink the mypy baseline — all
without changing product behaviour.

**Architecture:** Four independent streams, each its own PR. Stream 1 (backend/
delete) edits CI and is the highest-value cleanup. Stream 2 is test-only (the
runtime guards already exist). Stream 3 is docs/comments only. Stream 4 is an
iterative mypy ratchet pass. Recommended order 1 → 2 → 3 → 4; stream 4 never
blocks the others.

**Tech Stack:** Python 3.13 (invoke via `py -3.13`), FastAPI, pytest, mypy,
ruff, black, GitHub Actions CI. Spec: `docs/superpowers/specs/2026-06-22-tech-debt-cleanup-design.md`.

**Windows note:** there is no venv. Use `py -3.13 -m <tool>`; do NOT use the
POSIX `make` targets. Confirm pass/fail via exit code (`echo "EXIT=$?"`) because
piping pytest output drops the final summary line.

---

## Task 1: Delete the deprecated `backend/` path

`backend/` (46 `.py` files) is not imported by `app/`, `tests/`, or `scripts/`
(verified). But three CI mechanisms in `.github/workflows/ci.yml` reference it
and MUST be removed together, or downstream jobs break on dangling `needs:`.

**Files:**
- Delete: `backend/` (entire directory)
- Modify: `.github/workflows/ci.yml`
- Modify: `Makefile` (line ~51, `python -m backend.app.db.seed`)
- Modify: `pytest.ini` (comment mentioning legacy `backend/tests/`)
- Modify: `README.md` (legacy-OpenAPI wording the guard asserted on)
- Modify: `CLAUDE.md` (the `backend/` deprecation/guard paragraph)

- [ ] **Step 1: Re-verify nothing imports `backend`**

Run:
```bash
grep -rEn "(from|import)\s+backend" app tests scripts
```
Expected: no output. If anything prints, STOP — resolve the import first; the
deletion is not safe.

- [ ] **Step 2: Delete the directory**

Run:
```bash
git rm -r backend/
```
Expected: `rm 'backend/...'` lines for ~46 files. (Use `git rm` so the deletion
is staged in one move.)

- [ ] **Step 3: Remove the three legacy CI jobs**

In `.github/workflows/ci.yml`, delete these three job blocks entirely (match by
the `name:` anchor, not line number — line numbers shift as you edit):
- `legacy-path-guard:` (name: "Guard legacy path changes")
- `openapi-primary-guard:` (name: "Guard OpenAPI primary publication")
- `legacy-compatibility-tests:` (name: "Compatibility tests (legacy backend/app/* path)")

- [ ] **Step 4: Remove the `legacy` filter/output from `path-classifier`**

In the `path-classifier` job: delete the `legacy_changed: ${{ steps.filter.outputs.legacy }}`
output line and the `- 'backend/**'` filter entry that feeds it. Leave the
other outputs (`app_changed`, `frontend_changed`, `eval_changed`) untouched.

- [ ] **Step 5: Drop the deleted jobs from every downstream `needs:`**

These jobs currently declare `needs: [path-classifier, legacy-path-guard, openapi-primary-guard]`.
Edit each to `needs: [path-classifier]`:
`python-ci`, `eval-gate`, `docker-lint-build`, `web-ci`, `shell-lint`,
`python-warnings-smoke`.

- [ ] **Step 6: Remove `legacy_changed` clauses from `if:` conditions**

`docker-lint-build` and `shell-lint` have `if:` conditions that OR-in
`needs.path-classifier.outputs.legacy_changed == 'true'`. Remove just that
clause from each condition (keep the rest of the boolean expression intact).

- [ ] **Step 7: Drop `--ignore=backend` from the test run**

In the `python-ci` job, change:
```yaml
          coverage run -m pytest -q --ignore=backend
```
to:
```yaml
          coverage run -m pytest -q
```

- [ ] **Step 8: Fix the `Makefile` seed target**

`Makefile` line ~51 calls `python -m backend.app.db.seed`, which no longer
exists. Remove that line (and its target if the target becomes empty). Verify no
other `backend` reference remains:
```bash
grep -n backend Makefile
```
Expected: no output.

- [ ] **Step 9: Update `pytest.ini` and docs wording**

- `pytest.ini`: the `testpaths`/comment block references the legacy
  `backend/tests/` rationale and the `--ignore=backend` gating job. Trim those
  sentences so they no longer describe a removed path. Keep `testpaths = tests`.
- `README.md`: remove/rewrite the "Source-of-truth backend entrypoint" and
  "legacy" lines that named `backend/app/main.py` (these were asserted by the now
  deleted `openapi-primary-guard`). The source-of-truth is `app/api/main.py`.
- `CLAUDE.md`: delete the "Legacy: `backend/app/*` is deprecated … the guard in
  `.github/workflows/ci.yml` will fail the build" paragraph — the guard is gone.

- [ ] **Step 10: Sweep for any remaining `backend` references**

Run:
```bash
grep -rIn "backend/" . --include=*.yml --include=*.ini --include=*.md --include=Makefile -l 2>/dev/null
grep -rIn "legacy_changed\|legacy-path-guard\|openapi-primary-guard\|legacy-compatibility" .github/
```
Expected: no matches (ignore `.git/` and historical spec/plan docs that merely
describe the cleanup). Fix any stragglers.

- [ ] **Step 11: Run the full local suite**

Run:
```bash
py -3.13 -m pytest -q; echo "EXIT=$?"
```
Expected: `EXIT=0`. (No `--ignore=backend` needed now.) If failures appear,
confirm they are pre-existing/unrelated by checking out `backend/` temporarily —
but do NOT restore it; fix forward.

- [ ] **Step 12: Lint + format gates**

Run:
```bash
py -3.13 -m ruff check . ; echo "EXIT=$?"
py -3.13 -m black --check . ; echo "EXIT=$?"
```
Expected: both `EXIT=0`.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "chore: remove deprecated backend/ legacy path and its CI guards"
```

---

## Task 2: Pin the hashing-embedder warning with a characterization test

The warning already exists in `app/services/kb_embeddings.py:_build_from_env()`:
it logs a `WARNING` when falling back to the hashing embedder while `KB_API_KEY`
is set. There is no test guarding it. Add one. **No production code changes** —
this is a characterization test of existing behaviour.

**Files:**
- Create: `tests/test_embedder_hash_warning.py`

- [ ] **Step 1: Write the characterization test**

Create `tests/test_embedder_hash_warning.py`:
```python
"""Characterization test: the hashing-embedder fallback warns loudly when a
production-like config (KB_API_KEY set, no real embedder backend) would silently
degrade semantic search to near-random results. Pins existing behaviour in
app/services/kb_embeddings.py:_build_from_env — do not let it regress silently."""

import logging

from app.services.kb_embeddings import _build_from_env


def test_hash_fallback_warns_when_api_key_set(caplog):
    env = {"KB_API_KEY": "secret", "KB_EMBEDDINGS_BACKEND": ""}
    with caplog.at_level(logging.WARNING, logger="app.services.kb_embeddings"):
        embedder = _build_from_env(env)
    assert embedder.name == "hash"
    assert any(
        "hashing embedder" in rec.getMessage() and "near-random" in rec.getMessage()
        for rec in caplog.records
    ), "expected a loud WARNING about the hashing fallback"


def test_hash_fallback_silent_without_api_key(caplog):
    env = {"KB_EMBEDDINGS_BACKEND": ""}
    with caplog.at_level(logging.WARNING, logger="app.services.kb_embeddings"):
        embedder = _build_from_env(env)
    assert embedder.name == "hash"
    assert not any(
        "hashing embedder" in rec.getMessage() for rec in caplog.records
    ), "no KB_API_KEY → no warning noise"
```

- [ ] **Step 2: Run the test — expect PASS (behaviour already exists)**

Run:
```bash
py -3.13 -m pytest tests/test_embedder_hash_warning.py -v ; echo "EXIT=$?"
```
Expected: `EXIT=0`, both tests pass. If `test_hash_fallback_warns...` FAILS,
the implicit ST embedder may be resolving instead of hash (weights present on
this machine). In that case add `"ST_EMBED_MODEL": "/nonexistent"` to the `env`
dict in both tests to force the ST probe to fail and fall through to hash, then
re-run.

- [ ] **Step 3: Confirm no production code changed**

Run:
```bash
git status --short
```
Expected: only `tests/test_embedder_hash_warning.py` is new. No `app/` changes.

- [ ] **Step 4: Lint + commit**

```bash
py -3.13 -m ruff check tests/test_embedder_hash_warning.py ; echo "EXIT=$?"
py -3.13 -m black tests/test_embedder_hash_warning.py
git add tests/test_embedder_hash_warning.py
git commit -m "test: pin hashing-embedder fallback warning against regression"
```

---

## Task 3: Make the dual-surface architecture obvious in-code

Add one-glance banners at the four entry points so contributors stop editing the
wrong twin (a failure mode called out in CLAUDE.md). Docs/comments only — no
logic, no tests. Also fix one stale CLAUDE.md reference discovered during
planning.

**Files:**
- Modify: `app/api/kb_mvp/__init__.py` (already has a docstring — extend it)
- Modify: `app/api/v1/__init__.py`
- Modify: `data/www/index.html` (HTML comment)
- Modify: `frontend/README.md`
- Modify: `docs/architecture.md` (navigator table)
- Modify: `CLAUDE.md` (stale `kb_mvp.py` single-file reference)

- [ ] **Step 1: Banner in the MVP API package**

`app/api/kb_mvp/__init__.py` opens with a module docstring. Append this
paragraph to it (keep the existing text):
```
TWIN-SURFACE NOTE — do not unify. This is the SINGLE-TENANT MVP surface
(/api/kb/*, one KB_API_KEY, SQLite state). Its deliberate twin is the
MULTI-TENANT mature surface in app/api/v1/* (JWT/RBAC, Postgres + Qdrant).
Merging them is a known anti-pattern (forces MVP installs to carry ~2 GB of
multi-tenant deps). See docs/architecture.md before touching either.
```

- [ ] **Step 2: Banner in the v1 API package**

`app/api/v1/__init__.py`: add (or extend) the module docstring with:
```python
"""Multi-tenant mature API surface, mounted under ``/api/v1/*``.

TWIN-SURFACE NOTE — do not unify. This is the MULTI-TENANT surface (JWT/RBAC,
Postgres + Qdrant). Its deliberate twin is the SINGLE-TENANT MVP surface in
``app/api/kb_mvp/`` (/api/kb/*, one KB_API_KEY, SQLite). Merging them is a known
anti-pattern — see ``docs/architecture.md``.
"""
```
If `__init__.py` already has a docstring or imports, prepend the docstring above
them (a module docstring must be the first statement).

- [ ] **Step 3: Banner in the end-user UI**

`data/www/index.html`: add an HTML comment immediately after the opening
`<!DOCTYPE html>`/`<html ...>` line:
```html
<!-- TWO-FRONTENDS NOTE: this is the BUILT-IN END-USER MVP UI, served directly
     by FastAPI (vanilla HTML + i18n JSON in data/www/i18n/). Its twin is the
     admin/ops React app in frontend/ ("operations-console"). End-user chat
     changes go HERE; admin/diagnostic changes go in frontend/. See
     docs/architecture.md. -->
```

- [ ] **Step 4: Banner in the ops-console frontend**

`frontend/README.md`: add this note near the top (after the first heading):
```markdown
> **TWO-FRONTENDS NOTE:** this `frontend/` app is the **operations-console**
> (admin/diagnostic UI — TS, Vitest, Tailwind live here). Its twin is the
> built-in **end-user MVP UI** in `data/www/` (vanilla HTML, served by FastAPI).
> Admin/ops changes go here; end-user chat changes go in `data/www/`. See
> `docs/architecture.md`.
```

- [ ] **Step 5: Navigator table in the architecture doc**

In `docs/architecture.md`, add a short "Where do I edit X?" table (place it near
the existing two-path rationale; do not duplicate that prose):
```markdown
## Where do I edit X?

| I need to change… | Edit… |
|---|---|
| End-user chat / citation UI | `data/www/` |
| Admin / ops / diagnostics UI | `frontend/` |
| Single-tenant MVP API (`/api/kb/*`) | `app/api/kb_mvp/` |
| Multi-tenant API (`/api/v1/*`) | `app/api/v1/` |
```

- [ ] **Step 6: Fix the stale CLAUDE.md reference**

CLAUDE.md describes `app/api/kb_mvp.py` as a "large file, ~1200 LoC". It has
since been split into the `app/api/kb_mvp/` package. Update that line to:
```
- `/api/kb/*` — single-tenant MVP, single `KB_API_KEY` env, SQLite state. Source: `app/api/kb_mvp/` (package; split from the former single-file kb_mvp.py).
```

- [ ] **Step 7: Verify banners are present and nothing else changed**

Run:
```bash
grep -rn "TWIN-SURFACE NOTE\|TWO-FRONTENDS NOTE\|Where do I edit X" app data/www frontend docs CLAUDE.md
```
Expected: one hit per file edited above.

- [ ] **Step 8: Sanity — app still imports (no syntax slip in docstrings)**

Run:
```bash
py -3.13 -c "import app.api.kb_mvp, app.api.v1; print('ok')" ; echo "EXIT=$?"
```
Expected: `ok` and `EXIT=0`.

- [ ] **Step 9: Commit**

```bash
git add app/api/kb_mvp/__init__.py app/api/v1/__init__.py data/www/index.html frontend/README.md docs/architecture.md CLAUDE.md
git commit -m "docs: mark the two API surfaces and two frontends to prevent wrong-twin edits"
```

---

## Task 4: One mypy ratchet drive-to-zero pass

Shrink the baseline by proving a batch of currently-dirty files clean and adding
them to the ratchet allowlist `CLEAN_FILES` in `scripts/check_mypy_ratchet.py`.
This task is one pass; repeat it as more background PRs.

**Files:**
- Modify: target source files under `app/` (chosen in Step 2)
- Modify: `scripts/check_mypy_ratchet.py` (append cleaned files to `CLEAN_FILES`)

- [ ] **Step 1: Capture the current baseline**

Run:
```bash
py -3.13 -m mypy app 2>&1 | tail -1
```
Expected: a `Found N errors in M files` line. Record N (baseline ≈ 244).

- [ ] **Step 2: Pick a batch of low-error files NOT already in `CLEAN_FILES`**

List per-file error counts, smallest first, and pick ~5 files with 1–3 errors
each that are not already listed in `scripts/check_mypy_ratchet.py:CLEAN_FILES`:
```bash
py -3.13 -m mypy app 2>&1 | grep ": error:" \
  | sed 's/:[0-9].*//' | sort | uniq -c | sort -n | head -20
```
Prefer files already touched by Tasks 1/3 so the cleanup compounds. Write the
chosen paths down — they are the batch for this pass.

- [ ] **Step 3: Fix the type errors in the batch (annotations/narrowing only)**

For each file in the batch, run a focused check and fix every reported error by
adding annotations or narrowing — **never by changing runtime behaviour**, and
never with a blanket `# type: ignore` unless the error is a known stub gap (then
use a specific `# type: ignore[code]` with a one-line reason):
```bash
py -3.13 -m mypy app 2>&1 | grep "^app/path/to/file.py:"
```
Watch the embedder-Protocol lockstep: the `dimension` read-only `@property` on
`Embedder` / `_EmbedderLike` / `_HasSignature` must stay aligned, or the ratchet
goes red elsewhere.

- [ ] **Step 4: Verify the batch is clean**

Run (substitute your batch paths):
```bash
py -3.13 -m mypy app 2>&1 | grep -E "app/(fileA|fileB|fileC).py:" ; echo "EXIT=$?"
```
Expected: no error lines for the batch files.

- [ ] **Step 5: Add the batch to `CLEAN_FILES`**

In `scripts/check_mypy_ratchet.py`, append the cleaned paths to the
`CLEAN_FILES` list with a comment marking this pass:
```python
    # 2026-06-22 drive-to-zero pass
    "app/path/to/fileA.py",
    "app/path/to/fileB.py",
    # ... rest of the batch
```

- [ ] **Step 6: Run the ratchet gate**

Run:
```bash
py -3.13 scripts/check_mypy_ratchet.py ; echo "EXIT=$?"
```
Expected: `mypy ratchet OK — all <N> clean files report 0 errors.` and `EXIT=0`.

- [ ] **Step 7: Confirm the baseline strictly dropped**

Run:
```bash
py -3.13 -m mypy app 2>&1 | tail -1
```
Expected: `Found <N-k> errors` where `<N-k>` < the Step 1 baseline.

- [ ] **Step 8: Run the affected tests + format gate**

Run:
```bash
py -3.13 -m pytest -q ; echo "EXIT=$?"
py -3.13 -m black --check . ; echo "EXIT=$?"
```
Expected: both `EXIT=0` (type-only edits must not break tests).

- [ ] **Step 9: Commit**

```bash
git add scripts/check_mypy_ratchet.py app/
git commit -m "refactor(types): drive-to-zero mypy pass on <batch> (ratchet +<k> files)"
```

---

## Self-review notes

- **Spec coverage:** Stream 1 → Task 1; Stream 2 (narrowed to 2a test, 2b already
  done) → Task 2; Stream 3 → Task 3; Stream 4 → Task 4. The stale `kb_mvp.py`
  CLAUDE.md reference found in planning is folded into Task 3 Step 6.
- **No new product behaviour:** Task 2 is test-only; Task 3 is docs/comments;
  Task 4 is annotations-only; Task 1 removes dead code + its CI.
- **Order:** 1 → 2 → 3 → 4. Each task is an independent PR and can be reordered;
  Task 4 repeats as background passes.
