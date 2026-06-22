# Tech-Debt Cleanup — Round 1

**Status:** approved (brainstorm 2026-06-22)
**Scope:** four independent debt streams, each shipped as one or more small PRs
(~400 LoC ceiling per CONTRIBUTING.md). No new product features — this round is
purely code health and footgun removal.

## Goal

Reduce accumulated technical debt without changing product behaviour. Four
streams, ordered high-value-and-isolated → monotonous-and-ongoing:

1. **Delete `backend/`** — remove the deprecated legacy path and the three CI
   mechanisms wired to it.
2. **Config footguns** — make the two known "silent garbage" config traps loud
   at startup.
3. **Two-versions markers** — make the intentional dual-surface architecture
   obvious in-code so contributors stop editing the wrong twin.
4. **mypy ratchet drive-to-zero** — shrink the 244-error baseline file-batch by
   file-batch, appending to the existing ratchet allowlist.

Streams are independent: they can ship in any interleaving, but the recommended
delivery order is 1 → 2 → 3 → 4 (stream 4 runs in the background and never
blocks the others).

### Non-goals (explicitly out of scope)

- **Not** merging the two API surfaces or the two frontends. The dual-surface
  design is intentional (`docs/architecture.md`); stream 3 *documents* it, it
  does not unify it.
- **Not** migrating SQLite → Postgres, adding features, or touching the
  `experiments/pravo_nn` research line.
- **Not** a blanket refactor. Each change stays scoped to the debt it removes.

---

## Stream 1 — Delete `backend/` (1 PR)

`backend/` is 46 `.py` files. Verified: **nothing in `app/`, `tests/`, or
`scripts/` imports it.** But three CI mechanisms in `.github/workflows/ci.yml`
are wired to it and must be removed in the same PR (or CI will reference dead
paths):

- `legacy-compatibility-tests` job — runs `backend/tests/test_contract_schemathesis.py`
  and `backend/tests/test_health.py`.
- `legacy-path-guard` job/step — gates PRs that touch `backend/**`.
- `openapi-primary-guard` — the part that references `backend/app/main.py` as
  `LEGACY_OPENAPI`, plus the README-content assertions it enforces.
- `path-classifier` `legacy_changed` output — remove if it has no other consumer
  after the above are gone.

Also clean up:

- `--ignore=backend` from the pytest/coverage invocation (~line 221 of ci.yml)
  and any mirror in `pytest.ini` / `Makefile`.
- `README.md` lines the `openapi-primary-guard` asserted on (the
  "Source-of-truth backend entrypoint" / "legacy" wording referencing
  `backend/app/main.py`).
- `CLAUDE.md` "Legacy: `backend/app/*` is deprecated …" paragraph — the
  CI-guard sentence becomes false once the guard is gone.

**Risk:** medium — this edits CI. **Verification:** run the full local suite
(`py -3.13 -m pytest -q`, no longer needs `--ignore=backend`); confirm no
remaining CI job references a deleted job via `needs:`; confirm `grep -rn
backend` over `app tests scripts .github README.md CLAUDE.md Makefile pytest.ini`
returns only intentional residue (none expected). History is preserved in git;
the deletion is recoverable via the commit before it.

**Definition of done:** `backend/` gone; CI green with no orphaned `needs:`;
docs no longer claim a backend/ guard exists.

---

## Stream 2 — Config footguns (1 PR, characterization test only)

**Investigation finding (2026-06-22):** both config traps are ALREADY guarded in
the codebase. This stream is reduced to closing the single remaining coverage
gap — no new runtime behaviour.

### 2b. Dimension-mismatch guard — ALREADY DONE, do not reimplement

`app/services/embedder_signature.py:verify_or_store()` raises
`EmbedderMismatchError` (with a `kb-cli reindex --embedder <name>` instruction)
when the active embedder's signature disagrees with the stored index signature.
It is wired into `KnowledgeBaseStore.__init__` via `_verify_embedder_signature()`
(`app/services/kb_store.py:232`), so it fires when the store opens (startup /
first use), including a TOCTOU lock and a legacy-index backfill path. The
`/api/kb` health endpoint additionally reports `EMBEDDING_DIM_MISMATCH`
(CRITICAL), and the FAISS/Qdrant backends raise on dimension mismatch.
Covered by **7 passing tests** in `tests/test_embedder_signature.py` (unit +
store-level integration + legacy-upgrade path). **Leave it untouched.**

### 2a. Hashing-embedder warning — exists, missing a test

`app/services/kb_embeddings.py:_build_from_env()` already logs a `WARNING` when it
falls back to the hashing embedder while `KB_API_KEY` is set (gated on the key so
unit tests stay quiet; fires lazily at first `get_embedder()` build). The only
gap: **no test pins this behaviour.** Add a characterization test so the working
warning cannot silently regress. Do NOT broaden the trigger or move it to
startup — that was considered and rejected as YAGNI (it would add test noise for
no real-world gain).

**Risk:** minimal (test-only). **Definition of done:** a test asserts the warning
fires when falling back to hash with `KB_API_KEY` set, and does NOT fire without
the key. No production code changes.

---

## Stream 3 — Two-versions markers (1 PR, docs/comments only)

Make the intentional dual-surface fork obvious so the "edited the wrong twin"
failure mode (called out in CLAUDE.md) stops happening. No logic changes.

- Banner docstrings at the four entry points, each stating *what I am, when to
  edit me, who my twin is*:
  - `app/api/kb_mvp.py` — single-tenant MVP surface.
  - `app/api/v1/*` — multi-tenant mature surface.
  - `frontend/` — operations-console (admin/ops UI).
  - `data/www/` — built-in end-user MVP UI.
- A short "need to edit X → go to file Y" navigator table in
  `docs/architecture.md` (extend, don't duplicate, the existing rationale).

**Risk:** minimal (comments + docs). **Definition of done:** each of the four
entry points carries a one-glance banner; architecture doc has the navigator.

---

## Stream 4 — mypy ratchet drive-to-zero (N small PRs, background)

The mechanism already exists: `scripts/check_mypy_ratchet.py` holds a
`CLEAN_FILES` allowlist (currently 7 files) that may never regress to >0 errors,
while the overall baseline (244 errors / 48 files) shrinks independently.

Each pass:

1. Pick a batch of ~5–10 files — **prioritise files already touched by streams
   1–3** so the cleanup compounds.
2. `py -3.13 -m mypy app`; fix errors **only** in those files.
3. Append the cleaned files to `CLEAN_FILES`; ratchet goes green → one PR.

**Constraint:** type fixes must not change runtime behaviour (annotations and
narrowing only). Watch the embedder-protocol lockstep recorded in repo memory —
the `dimension` read-only `@property` across the three embedder Protocols must
stay aligned or the ratchet gate goes red.

**Risk:** low but monotonous and effectively endless — runs in the background and
never gates streams 1–3. **Definition of done (per pass):** N more files in
`CLEAN_FILES`, ratchet green, baseline count strictly lower than before.

---

## Delivery & verification summary

| Stream | PRs | Risk | Primary verification |
|--------|-----|------|----------------------|
| 1 backend/ delete | 1 | medium (CI) | full suite green, no orphaned `needs:` |
| 2 config footguns | 1 | minimal | characterization test for hash warning (2b already done) |
| 3 markers | 1 | minimal | manual read of four banners |
| 4 mypy ratchet | N | low | ratchet green, baseline strictly down |

Each PR follows the repo discipline: TDD where logic changes (stream 2, 4),
Conventional Commits, `ruff + black + pytest` green, ~400 LoC ceiling. On
Windows, invoke the underlying commands directly (`py -3.13 -m …`) rather than
the POSIX `make` targets.
