# Vectorstore SearchFilters tests refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable the 7 `@pytest.mark.skip`-ed tests across `tests/test_services_vectorstore.py` and `tests/test_vector_stores.py` so they validate the current production retrieval contract (`SearchFilters` dataclass, `tenant_id` requirement, new `qdrant_client.http.models.Match*` shape). Production code already works; this is a tests-only effort.

**Why now:** The skips were marked in commits `432133d test(vectorstore): skip stale tests pending SearchFilters rewrite` and `0104fef test(misc): one targeted fix + skip 13 stale tests with explicit reasons`. They are documented technical debt explicitly flagged for follow-up. Closing them returns the test suite to "every test runs" state, which is a v1.x quality bar.

**Scope:**
- **In:** rewrite test stubs (`DummyVectorStore`, `_StubQdrant`, `_StubFaiss`, helper fixtures), update calls to match the production `vectorstore.search(...)` signature, align `qdrant_client.http.models.*` mock shapes with the real package's current API.
- **Out:** Do **not** change `app/services/vectorstore.py`, `app/retriever/vector_store.py`, `app/retriever/qdrant.py`, `app/retriever/faiss.py`. Production code is correct against the deployed `qdrant-client~=1.11` and the live SearchFilters contract; if a test surfaces a real bug, open a separate issue.

**Tech context:**
- Production `vectorstore.search(query, *, top_k, owner, tags, ..., tenant_id)` → builds `SearchFilters.from_input(...)` → `store.search(query, top_k=top_k, filters=filters)`.
- `SearchFilters` is a frozen dataclass in `app/retriever/vector_store.py:79-119` with fields `tenant_id, owner, tags (tuple), act_type, issuer, reg_number, is_active, revision_mode`.
- `VectorStore.search` protocol signature: `search(self, query: str, top_k: int, *, filters: SearchFilters) -> list[dict[str, object]]`.
- `tests/stubs/qdrant_client/http/models.py` (see [repo-test-stubs](../../../memory/repo_test_stubs.md)) — extend this if the on-disk stub is missing the current Match* shape.

**Plans dir conventions:** See `docs/superpowers/plans/2026-05-25-mvp-completion.md` for the canonical style. Each task = failing test first, then implementation, then commit. Per-task atomic commits using `git commit -m @'…'@` here-strings on PowerShell (see [repo-pythonenv-py-launcher](../../../memory/repo_pythonenv_py_launcher.md) for the `py -3` launcher).

---

## File Structure

**Files modified (tests only):**
- `tests/test_services_vectorstore.py` — rewrite `DummyVectorStore` + `ExplodingVectorStore` + 3 skipped tests.
- `tests/test_vector_stores.py` — rewrite `_StubQdrant` Match* mocks + 4 skipped tests.

**Files possibly modified (if needed for stub parity):**
- `tests/stubs/qdrant_client/http/models.py` — extend `MatchValue`/`MatchText` if the on-disk stub drifted behind real `qdrant-client~=1.11`.

**Files NOT modified:**
- `app/services/vectorstore.py`, `app/retriever/*.py` — production code stays as-is.

---

## Sprint 1 — Service-level fallback tests (~2-3 h)

**Goal:** All 3 tests in `tests/test_services_vectorstore.py` pass against the live `vectorstore.search` signature with `tenant_id`.

**Abort point:** None — this sprint is one cohesive unit; partial completion isn't useful.

### Task 1.1: Rewrite `DummyVectorStore` to the current `VectorStore` protocol

**Files:**
- Modify: `tests/test_services_vectorstore.py` (lines 20-44 — class `DummyVectorStore`)

- [ ] **Step 1: Read the current `VectorStore` protocol**

  `app/retriever/vector_store.py:11-30` — confirm signature is `search(self, query: str, top_k: int, *, filters: SearchFilters)`. Note `top_k` is positional after `query` (NOT keyword-only).

- [ ] **Step 2: Rewrite the `DummyVectorStore.search` signature**

  Replace lines 35-44 with:
  ```python
  def search(
      self,
      query: str,
      top_k: int,
      *,
      filters: SearchFilters,
  ) -> List[dict[str, object]]:
      self.search_calls.append((query, top_k, filters))
      return self.results[:top_k]
  ```
  Also: change `self.search_calls` type annotation from `List[tuple[str, int, str | None, list[str] | None]]` to `List[tuple[str, int, SearchFilters]]`.

- [ ] **Step 3: Add the import**

  At the top of the file add: `from app.retriever.vector_store import SearchFilters`.

- [ ] **Step 4: Repeat for `ExplodingVectorStore`** (lines 47-68 — same signature change; it still raises, just match the protocol).

- [ ] **Step 5: Verify file still parses**

  ```powershell
  py -c "import tests.test_services_vectorstore" 2>&1
  ```
  Expected: no SyntaxError. Tests still skipped at this point, so pytest is irrelevant until Task 1.2.

- [ ] **Step 6: Commit**

  ```powershell
  git add tests/test_services_vectorstore.py
  git commit -m @'
  test(vectorstore): align DummyVectorStore with current protocol

  Rewrite DummyVectorStore and ExplodingVectorStore search() signature
  to accept (query, top_k, *, filters: SearchFilters) instead of the
  pre-SearchFilters (owner, tags) form. No tests un-skipped yet; that
  happens in subsequent tasks.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  '@
  ```

### Task 1.2: Un-skip and rewrite `test_index_chunks_success`

**Files:**
- Modify: `tests/test_services_vectorstore.py` (test at lines ~85-105)

- [ ] **Step 1: Run the test in its current skipped state**

  ```powershell
  py -m pytest tests/test_services_vectorstore.py::test_index_chunks_success -v
  ```
  Expected: 1 skipped. (Baseline before un-skipping.)

- [ ] **Step 2: Remove the `@pytest.mark.skip` decorator from `test_index_chunks_success`**.

- [ ] **Step 3: Rewrite the test body**

  - Call `vectorstore.search("anything", top_k=5, tenant_id="t1")` (tenant_id is required).
  - Assert that `dummy.search_calls` last entry has the expected `SearchFilters` shape (e.g. `filters.tenant_id == "t1"`, `filters.owner is None`).

- [ ] **Step 4: Run the test in isolation**

  ```powershell
  py -m pytest tests/test_services_vectorstore.py::test_index_chunks_success -v
  ```
  Expected: PASS. If it fails: read the assertion error, adjust the test (NOT the production code).

- [ ] **Step 5: Commit**

  ```powershell
  git add tests/test_services_vectorstore.py
  git commit -m @'
  test(vectorstore): un-skip test_index_chunks_success

  Calls vectorstore.search with tenant_id="t1" and asserts the
  SearchFilters object reaching DummyVectorStore.search().

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  '@
  ```

### Task 1.3: Un-skip `test_index_chunks_fallback_and_search_order`

Same recipe as Task 1.2:
- [ ] Remove skip decorator.
- [ ] Adjust calls — pass `tenant_id=`.
- [ ] Adjust assertions to read `filters` instead of `owner`/`tags`.
- [ ] `py -m pytest tests/test_services_vectorstore.py::test_index_chunks_fallback_and_search_order -v` → PASS.
- [ ] Commit.

### Task 1.4: Un-skip `test_fallback_search_filters`

This test specifically exercises the in-memory fallback path. The fallback function `_search_fallback` (in `app/services/vectorstore.py:200+`) already accepts the filter fields as keyword args, so this test should map cleanly.

- [ ] Remove skip decorator.
- [ ] Pass `tenant_id="t1"` (mandatory in `SearchFilters.from_input`).
- [ ] Verify chunks with non-matching tenant are filtered out.
- [ ] `py -m pytest tests/test_services_vectorstore.py -v` → all 3 PASS.
- [ ] Commit.

### Task 1.5: Sprint 1 regression sweep

- [ ] **Step 1: Run the full audit + vectorstore-related test set**

  ```powershell
  py -m pytest tests/test_services_vectorstore.py tests/test_audit_db.py tests/test_audit_middleware.py -v
  ```
  Expected: 3 + 13 = 16 PASS (audit tests as smoke against fixture neighbours).

- [ ] **Step 2: Run lint**

  ```powershell
  py -m ruff check tests/test_services_vectorstore.py
  py -m black --check tests/test_services_vectorstore.py
  ```
  Expected: clean.

---

## Sprint 2 — Backend-level vector store tests (~3-4 h)

**Goal:** All 4 tests in `tests/test_vector_stores.py` pass against the current `qdrant_client.http.models` API and `SearchFilters` shape.

**Abort point:** After Task 2.2 — first two qdrant tests + faiss baseline gives most of the value if time is short.

### Task 2.1: Audit current `qdrant_client.http.models` shape

**Files:**
- Read: `tests/stubs/qdrant_client/http/models.py` (in-tree stub)
- Read: `app/retriever/qdrant.py` (production usage of `qmodels.*`)

- [ ] **Step 1: List all symbols the production code imports from `qdrant_client.http.models`**

  ```powershell
  py -c "import re; src = open('app/retriever/qdrant.py').read(); print(set(re.findall(r'qmodels\.(\w+)', src)))"
  ```

- [ ] **Step 2: Confirm each symbol exists in `tests/stubs/qdrant_client/http/models.py`**

  If anything is missing — that's a stub-drift bug; add it. Per [repo-test-stubs](../../../memory/repo_test_stubs.md), `tests/test_prometheus_stub_contract.py` is the regression-sentinel pattern; mirror it as `tests/test_qdrant_stub_contract.py` if you find gaps.

- [ ] **Step 3: Document the current `MatchValue` / `MatchText` shape in a short note in the stub file's docstring** (so the next person doesn't have to re-investigate).

- [ ] **Step 4: Commit** (if any stub changes; otherwise skip to Task 2.2).

### Task 2.2: Un-skip `test_qdrant_upsert_batches_embeddings`

This test inlines its own `_PointStruct`, `_VectorParams`, etc. via `monkeypatch.setattr`. The shape should still work; the failure is likely an indirect effect of the SearchFilters drift propagating into the test setup.

- [ ] Remove skip decorator at line 206.
- [ ] Run in isolation: `py -m pytest tests/test_vector_stores.py::test_qdrant_upsert_batches_embeddings -v`.
- [ ] Inspect the actual error. Most likely cause: missing `SearchFilters` import or an outdated kwarg in `store.upsert`. Fix the test, not prod.
- [ ] Commit.

### Task 2.3: Un-skip `test_faiss_search_returns_payload`

The faiss backend has its own `search(filters: SearchFilters)` signature. The test needs:
- [ ] Construct a `SearchFilters(tenant_id="t1")` to pass through `store.search`.
- [ ] Verify the returned payload contains expected fields.
- [ ] Commit.

### Task 2.4: Un-skip `test_qdrant_search_builds_filter_parity`

This test asserts that qdrant's `Filter`/`FieldCondition`/`MatchValue` are built consistently from a given `SearchFilters` input. Key validation:
- [ ] `tenant_id` maps to a `FieldCondition(key="tenant_id", match=MatchValue(value="t1"))`.
- [ ] `tags` maps to multiple `MatchValue`s (one per tag), combined with `must`.
- [ ] `is_active=False` correctly serialises (bool ≠ None case).
- [ ] Commit.

### Task 2.5: Un-skip `test_qdrant_and_faiss_apply_same_filters`

The parity test. Both backends must produce the same `hit` ordering given the same `SearchFilters`.

- [ ] Adjust to call `.search(query, top_k=5, filters=SearchFilters(tenant_id="t1", ...))`.
- [ ] Assert `[hit["sha256"] for hit in faiss_hits] == [hit["sha256"] for hit in qdrant_hits]`.
- [ ] Commit.

### Task 2.6: Sprint 2 regression sweep + delete the `_BACKEND_REFACTOR_SKIP` constant

- [ ] All 4 tests pass: `py -m pytest tests/test_vector_stores.py -v`.
- [ ] Delete `_BACKEND_REFACTOR_SKIP = (...)` block at the top of the file (lines 7-12); also remove `_VECTORSTORE_REFACTOR_SKIP` at the top of `tests/test_services_vectorstore.py` (lines 11-17).
- [ ] Lint: `py -m ruff check tests/ && py -m black --check tests/`.
- [ ] Final full suite: `py -m pytest -q --ignore=backend`.
- [ ] Commit.

---

## Sprint 3 — Documentation + release-readiness (~30 min)

- [ ] **Step 1: Update `README.md` (if applicable)** — if the README references `tenant_id` semantics anywhere, ensure they match.
- [ ] **Step 2: Open the PR** with a body summarising:
  - Why these tests were skipped (`432133d`, `0104fef`).
  - What was changed (signature alignment, no prod code touched).
  - Verification: `py -m pytest tests/test_services_vectorstore.py tests/test_vector_stores.py -v` → 7/7 PASS.
- [ ] **Step 3: Squash the per-task commits** if maintainer prefers small history; otherwise leave atomic per-task.

---

## Acceptance criteria

- [ ] Zero `@pytest.mark.skip` decorators remain in `tests/test_services_vectorstore.py` (was 3) and `tests/test_vector_stores.py` (was 4).
- [ ] `py -m pytest tests/test_services_vectorstore.py tests/test_vector_stores.py -v` → **7/7 pass**.
- [ ] `py -m pytest -q --ignore=backend` — full suite still green (no regression).
- [ ] `py -m ruff check . && py -m black --check .` — clean.
- [ ] `py -m mypy .` — error count not worse than `568` (PR #552 baseline).
- [ ] PR body explicitly states "no production code changed" and points to commits `432133d` / `0104fef` as historical context.

## Out of scope (open separate issues if encountered)

- Anything the test surfaces as a real production bug → new issue with a regression test.
- Stub-vs-real package drift for symbols other than `Match*` and `SearchFilters` → file under [repo-test-stubs](../../../memory/repo_test_stubs.md) pattern.
- `mypy` errors on the affected files that aren't introduced by this refactor.

## Estimated effort

- Sprint 1: 2-3 hours (3 tests, tight scope)
- Sprint 2: 3-4 hours (4 tests, more stub-archaeology)
- Sprint 3: 30 min
- **Total: ~6-8 hours of focused TDD work**
