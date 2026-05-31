# Code Health — mypy Ratchet: "object/None-narrowing" clean-set + CI gate (Design)

**Date:** 2026-05-31
**Status:** Design document. Subordinate to repo conventions in `CLAUDE.md` and `CONTRIBUTING.md`.
**Predecessor:** `docs/superpowers/specs/2026-05-31-mypy-safe-pass-clean-files-design.md` (#11 / PR #567 — drove `deps.py` + `file_stats.py` to zero and established the per-file drive-to-zero template).

> Второй проход «drive-to-zero» по типовому долгу. Берём пять файлов с *чистым*
> (поведенчески-нейтральным) долгом семейства «`object`/`None`-narrowing», доводим их
> до нуля ошибок mypy **без изменения поведения**, и — главное — добавляем CI-гейт,
> который не даёт уже-чистым файлам деградировать обратно. Сегодня `mypy` в CI
> запускается с `continue-on-error: true`, то есть ничего не мешает регрессии.

---

## 1. Context

`py -3 -m mypy app` reports **225 errors across 45 files** (config: `[tool.mypy]` in
`pyproject.toml`; `backend/` excluded). Per `CLAUDE.md`, a change is judged by *new
errors on touched lines*, and the correct unit of cleanup is **whole-file
drive-to-zero** — a partially-fixed file re-accrues noise on the next edit.

Two structural facts shape this pass:

1. **The error-code histogram is dominated by type-flow problems, not missing
   annotations.** Top codes: `arg-type` 52, `attr-defined` 50, `misc` 22,
   `call-overload` 20, `assignment` 19. (`var-annotated` is **1**; `no-any-return`
   is **0** — there is no "add the obvious annotation" sweep to be had here.) The
   tractable subset is a recurring micro-pattern: **values typed `object` (or
   `X | None`) flowing into typed sinks** — `.get()`, `int()`, `float()`, model
   constructors.

2. **CI mypy is advisory, not gating.** `.github/workflows/ci.yml` runs the MyPy
   step with `continue-on-error: true`. Nothing prevents an already-clean file
   from regressing. The #11 pass made `deps.py`/`file_stats.py` clean, but that
   cleanliness is currently "enforced socially" only.

This design (a) extends the clean set by five tractable files and (b) closes the
regression gap with an explicit ratchet gate.

## 2. Goal & non-goals

**Goal.** Drive five files to **0 mypy errors** with **zero behavioral change**,
and add a CI ratchet gate that fails the build if any file in a documented
clean-set has >0 errors.

Clean-set for this pass (24 errors, verified self-validating — every cited line is
within its file's real length):

| File | errs | Error family |
|---|---|---|
| `app/llm/lora_runtime.py` | 2 | `int(object)` / `float(object)` |
| `app/services/kb_store.py` | 4 | `int \| None` narrowing |
| `app/api/v1/search.py` | 5 | `object` → `SearchHit` fields + list-comp |
| `app/services/vectorstore.py` | 8 | `object.get` (isinstance-narrowing idiom) |
| `app/api/v1/users.py` | 5 | `str\|None`→`str`, `str`→`UserRole`, `int\|None`→`int` |

Expected global delta: **225 → ~201** (exact count confirmed during implementation;
narrowing one `object` source can clear several dependent errors at once).

**Non-goals (binding for this PR):**

- **`app/models/__init__.py` (6 errors).** Verified to be an optional-import
  fallback — `try: from .lora import …; except Exception: LoraAdapterInfo = None;
  LoraAdapterName = None; LoraStatusResponse = None` (`__init__.py:12-17`). The
  `[assignment]`/`[misc] "Cannot assign to a type"` errors are the **same
  load-bearing-fallback family** as `config.py`'s `BaseSettings` branch and
  `chunking.py`'s optional-parser guards. Deferred to a *guarded* pass that first
  pins the reduced-dependency import behavior with a test. **This corrects the #11
  spec, which listed `models/__init__.py` as a plain "future candidate".**
- **`app/api/kb_mvp.py` (5 errors).** The cluster is async-generator typing
  (`[misc]` async-generator return type at L1095; `[attr-defined] "bool" has no
  attribute "__aiter__"` at L1121/1131/1138) plus one `[assignment]`. This is a
  genuine type subtlety (a function inferred as `bool` where an async-iterable is
  expected), and `kb_mvp.py` is the 1249-LoC MVP surface — low ROI to touch for 5
  errors now, and it deserves a careful look rather than a mechanical fix.
- **The `config.py` `BaseSettings` branch, `chunking.py` guards, the
  `qdrant_client.py` / `models/qdrant_client.py` shims, `srv/`, `backend/`.**
  Unchanged from #11's deferred list.
- **Reducing the global count beyond the clean-set files.**

## 3. Approach

**Per-file drive-to-zero, fix at the idiom level.** Where a file's errors share a
single cause, fix the cause once rather than suppressing each site. No blanket
`# type: ignore`; a targeted, single-line, justified ignore is permitted only where
a documented third-party/stdlib overload genuinely cannot be satisfied.

Fix strategy per cluster (exact edits are settled in the implementation plan; these
are the behavior-preserving shapes):

- **`vectorstore.py` (8 × `object.get`).** The in-memory fallback search assigns
  `meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}`
  (`vectorstore.py:159`). mypy keeps `meta` as `object` because the
  isinstance-checked expression is a *separate call* from the assigned one.
  Hoist to a local: `m = chunk.get("meta"); meta = m if isinstance(m, dict) else {}`.
  This narrows `meta` to `dict[...]`, clears the dependent `.get` errors, and
  removes a real double-call smell. Apply the same hoist to the `chunk`-level
  accesses as needed. **Runtime behavior identical** (same keys read, same
  defaults).
- **`search.py` (5).** The fallback `SearchHit` construction reads `object`-typed
  fields (`file`, `page`, `text`) and `float(object)`; the list-comp then yields
  `dict` where `list[SearchHit]` is expected. Narrow at the source (the same
  hoist/`isinstance` idiom and explicit `str(...)`/`int(...)` coercions already
  used elsewhere in the file), so the constructed `SearchHit` types line up.
- **`kb_store.py` (4 × `int | None`).** Counts/sizes coming back as `int | None`
  feed `int(...)` / int-typed assignments. Provide an explicit non-None default
  (`value or 0` / a local guard) before the `int()` call / assignment. Defensive
  coercions stay.
- **`lora_runtime.py` (2).** An `object`-typed config/env value feeds
  `int(...)`/`float(...)`. Narrow the source to `str | int | float` (or guard)
  before coercion — the same shape used in `deps.py` under #11.
- **`users.py` (5).** `create_user` builds a `UserResponse` from a `UserRecord`
  whose `email` is `str | None`, `role` is `str`, and `id` is `int | None`, while
  the response model / `log_security_event` expect non-optional `str` / `UserRole`
  / `int`. Resolve with explicit guards/`or`-defaults and an enum coercion
  (`UserRole(record.role)`), matching the already-present `id=record.id or 0`
  idiom (`users.py:79`). Because this touches model construction, it is pinned by
  a **characterization test** (§5) asserting `create_user`'s output is byte-for-byte
  the same for a representative record before and after.

## 4. CI ratchet gate

**New file `scripts/check_mypy_ratchet.py`.** A small, dependency-free script:

1. Holds an explicit `CLEAN_FILES` list (normalised, forward-slash paths).
2. Runs `mypy app` once (reusing the repo's `[tool.mypy]` config), parses the
   `path:line: error:` lines.
3. Exits non-zero with a clear message listing any `CLEAN_FILES` entry that has
   >0 errors (and which lines); exits 0 otherwise. It does **not** assert anything
   about non-clean files, so the global baseline can keep shrinking independently.

Seed `CLEAN_FILES` with the files known-clean after this pass:
`app/core/deps.py`, `app/services/file_stats.py` (from #11), plus
`app/llm/lora_runtime.py`, `app/services/kb_store.py`, `app/api/v1/search.py`,
`app/services/vectorstore.py`, `app/api/v1/users.py` → **7 files locked.**

**CI wiring (`.github/workflows/ci.yml`).** Add a new step **without**
`continue-on-error` that runs `python scripts/check_mypy_ratchet.py`, placed in the
same job that already invokes mypy. Leave the existing advisory `mypy .` step
exactly as-is (it remains the whole-repo trend signal). Net effect: the repo gains
a hard gate scoped *only* to files already proven clean — zero risk of failing CI
on the 200-error baseline, full protection against regressing the clean set.

**Extensibility.** Each future drive-to-zero pass appends its files to
`CLEAN_FILES` in the same PR that cleans them. The list is the living ledger of
what's locked.

## 5. Verification

1. **mypy (whole package).** `py -3 -m mypy app`. Acceptance: each of the five
   clean-set files reports **0 errors**, no new errors appear in untouched files,
   and the global total drops from 225 to ~201. A single-file check
   (`mypy app/foo.py`) is **not** used — per `CLAUDE.md` it under-reports because
   it follows only that file's imports.
2. **Behavior-preservation.** `py -3 -m pytest -k "vectorstore or kb_store or
   search or lora or users" -q --ignore=backend`, confirmed via exit code
   (`$LASTEXITCODE` / `EXIT=$?`) — piping pytest drops the final "N passed" line
   (repo gotcha).
3. **Characterization test (new).** One test pinning `create_user`'s `UserResponse`
   output for a representative `UserRecord`, asserted to pass on current code
   *before* the §3 `users.py` refactor (it characterises existing behavior).
4. **Ratchet gate.** `python scripts/check_mypy_ratchet.py` exits 0 with all seven
   files clean. Sanity-check the gate *fails* if a known error is reintroduced
   (manual one-off during implementation; not committed).
5. **Lint/format.** `py -3 -m ruff check .` and `py -3 -m black --check .` clean.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `users.py` enum/guard fixes subtly change `create_user` output | Characterization test (§5.3) pins the full response object; `UserRole(record.role)` round-trips an already-valid string. |
| Narrowing an `object` source changes which branch runs | Each fix is a pure type-hoist of an *existing* expression (e.g. `vectorstore.py:159`), reading the same keys with the same defaults — verified by the module's existing tests. |
| A `kb_store.py` `or 0` default masks a real `None` that mattered | Inspect each site during planning; only apply where the column is non-nullable in practice or `0` is already the semantic default. If ambiguous, defer that single line rather than guess. |
| Ratchet script drift (a file is removed/renamed) | Script treats a missing `CLEAN_FILES` path as a hard error, so renames must update the list — failing loudly, not silently. |
| Tool-output corruption observed earlier in the session feeding wrong data into the plan | All scoping data here was re-derived with a **self-validating** run (rejects any error citing a line beyond the file's length) and ground-truth `ast.parse`/`py_compile` checks; implementation re-runs mypy per file, so any bad datum fails its own acceptance check. |

## 7. Deliverable

- **PR1:** the five-file clean-set + `scripts/check_mypy_ratchet.py` + the CI step,
  seeded with all seven clean files. Conventional-Commit `chore(mypy): …`,
  ≤400 LoC, no `backend/**` change (no `legacy-path-approved` label needed).
  One commit per clean-set file (each independently verified at 0) plus one for the
  gate, mirroring #11's commit granularity.
- The PR body links this spec and the #11 spec, and notes the explicitly-deferred
  follow-ups (`models/__init__.py` guarded pass, `kb_mvp.py` async-generator typing).

## 8. Open questions

None at design stage. Exact per-line edits, the precise `kb_store.py` default
choices, and the `mypy`-output parsing in `check_mypy_ratchet.py` are settled
during the writing-plans / implementation phase. The five files' error counts and
families are confirmed; `models/__init__.py` and `kb_mvp.py` are confirmed deferred.
