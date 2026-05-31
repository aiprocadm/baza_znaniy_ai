# Consolidated Plan Status — All 11 Plans Merged (2026-05-31)

> **What this is:** a single merged view of every plan under `docs/superpowers/plans/`.
> It supersedes the individual checkbox tracking in those files (the per-plan `- [ ]`
> boxes were authored as TDD specs and **never ticked back**, so they are *not* a
> reliable completion signal — git history and the code are).
>
> **Headline:** all 11 plans are **implemented and their tests pass.** There is no
> substantial code left to write. The only unmerged work is the local mypy branch
> (`chore/mypy-safe-pass-deps-filestats`), which is itself complete and green.

---

## How this was verified (gates run 2026-05-31)

| Gate | Command | Result |
|---|---|---|
| Tests | `py -3 -m pytest -q -m "not requires_postgres" --ignore=backend` | **exit 0** — full suite green, ~6 intentional skips, 0 failures |
| Types | `py -3 -m mypy app` | **226 errors in 46 files** (down from 244 baseline); `core/deps.py` + `services/file_stats.py` at **0** |
| Branch | `git log --oneline main..HEAD` / `HEAD..main` | **6 ahead, 0 behind** — current branch is a strict superset of `main` |
| Tree | `git status --short` | **clean** |
| Release | `git tag -l v1.0.0` | **present** |

The 226 residual mypy errors are the explicitly-deferred out-of-scope set (config `BaseSettings`
light-install branch, `chunking.py` optional-dep guards, `qdrant_client` shims) — see the mypy plan.

---

## Status matrix

| # | Plan | Goal (one line) | Status | Shipped via |
|---|---|---|---|---|
| 1 | `2026-05-22-foundation-cleanup-audit-i18n` | Repo tidy-up, DB-persisted audit log + admin endpoint, i18n scaffold (ru) | ✅ DONE | many commits 2026-05-22 |
| 2 | `2026-05-22-pdf-citation-viewer` | Clickable `[file.pdf, p.12]` citations → modal PDF.js viewer | ✅ DONE | ~20 commits → main |
| 3 | `2026-05-25-mvp-completion` | kb-cli (backup/restore/reindex/health), UI polish, OSS-ready, `v1.0.0` | ✅ DONE | ~20 commits, tag `v1.0.0` |
| 4 | `2026-05-25-w1-synthetic-qa-generation` | Teacher-LLM → JSONL synthetic Q&A dataset generator | ✅ DONE | W1 commit chain |
| 5 | `2026-05-28-vectorstore-searchfilters-tests-refactor` | Re-enable 7 skipped SearchFilters/backend tests, no prod change | ✅ DONE | PR #556 |
| 6 | `2026-05-29-w3-rag-aware-fine-tuning` | 4-variant RAG-aware SFT dataset + `train_lora --prompt-mode rag` | ✅ DONE | PR #558 / #559 |
| 7 | `2026-05-29-retrieval-degradation-visibility` | `RetrievalReport` contract + health/Prometheus surfacing (ops-loud) | ✅ DONE | PR #562 |
| 8 | `2026-05-29-w4-dpo-post-training` | DPO dataset synth + `train_dpo` CLI + live feedback collection | ✅ DONE | PR #560 |
| 9 | `2026-05-29-retrieval-degradation-customer-loud` | Per-query degradation banner on `/api/kb/ask` (MVP UI) | ✅ DONE | PR #563 |
| 10 | `2026-05-30-retrieval-degradation-v1-chat` | Carry `RetrievalReport` onto v1 multi-tenant `ChatResponse` | ✅ DONE | PR #565 |
| 11 | `2026-05-31-mypy-safe-pass-clean-files` | `deps.py` + `file_stats.py` → 0 mypy; drop dead config v1 fallback | ✅ DONE (local, **unmerged**) | this branch |

---

## Dependency graph (why the plans were sequenced as they were)

```
ML "Pack B++" strengthening:      W1 (synthetic QA)  →  W3 (RAG-aware SFT)  →  W4 (DPO post-train)
                                  #4 feeds #6 (seed JSONL)        #6 feeds #8 (SFT adapter + pairs)

Retrieval-degradation visibility: PR1 visibility (#7, retrieval_health contract)
                                       ├──→ PR2  customer-loud  (#9, MVP /api/kb/ask banner)
                                       └──→ PR2b v1-chat        (#10, v1 ChatResponse field)
                                  Both consumers reuse report_payload() / current_report() from PR1.

mypy safe pass (#11):             independent; depends only on pinned pydantic 2.11 + sqlmodel 0.0.25.
```

The two-path API design (`/api/kb/*` MVP vs `/api/v1/*` multi-tenant) is preserved throughout —
PR2 (MVP) and PR2b (v1) are deliberately *separate* implementations of the same signal, per
`docs/architecture.md`.

---

## Per-plan detail + evidence

### 1. foundation-cleanup-audit-i18n — ✅ DONE
Tech-debt removal (`dev_kb_only.py` → `scripts/dev_server_mvp.py`, `docs/architecture.md`, `.gitignore`),
DB-persisted audit log (`alembic/versions/20260522_01_audit_log.py`, `app/models/audit.py`,
`app/core/audit_db.py`, `app/core/audit_middleware.py`, `app/api/v1/admin_audit.py`), i18n scaffold
(`data/www/i18n/ru.json` + `_loader.js`, wired into `index.html`/`admin.html`). 22 audit/i18n tests pass.

### 2. pdf-citation-viewer — ✅ DONE
Migration `20260522_02_pdf_citation.py` (page_number / has_original_file / file_relpath); `kb_store`
page-aware `SearchHit`; `kb_mvp.py` `_parse_file_bytes_with_pages`, blob upload, `get_document_file`
(path-traversal guarded) + cascade delete; lazy-loaded PDF.js viewer (`data/www/js/pdf-viewer.js`,
`vendor/pdfjs/`), citation rendering, scan-PDF fallback banner, 14 i18n keys. 15 test files.

### 3. mvp-completion — ✅ DONE
`scripts/kb_cli.py` + `scripts/cli/{backup,restore,reindex,health}.py`; extended `/api/kb/health`
(`kb_stats`, `compliance_mode`); debug pills behind `?debug=1`; outsider-first README; OSS files
(LICENSE Apache-2.0, CONTRIBUTING, SECURITY, ROADMAP, `install.sh`); `kb-cli` entry point; `v1.0.0` tag.
⚠️ The 3 screenshots in `docs/screenshots/` are **68-byte 1×1 placeholders** (see loose ends).

### 4. w1-synthetic-qa-generation — ✅ DONE
`app/services/synthetic_qa.py` (QAPair, GenerationMode, SyntheticQAGenerator, quality filters, cost guard,
resume) + `scripts/generate_synthetic_qa.py`. 46 unit + 3 CLI tests. Runtime needs an LLM API key
(DeepSeek/Groq/OpenRouter/OpenAI) and a populated KB — not a code blocker.

### 5. vectorstore-searchfilters-tests-refactor — ✅ DONE (PR #556)
7 previously-skipped tests re-enabled (3 in `test_services_vectorstore.py`, 4 in `test_vector_stores.py`);
`DummyVectorStore`/stubs realigned to the current `VectorStore` Protocol + `SearchFilters`. **Zero**
`@pytest.mark.skip` / skip-constants remain. No production code touched.

### 6. w3-rag-aware-fine-tuning — ✅ DONE (PR #558/#559)
`app/services/rag_dataset.py` (RELEVANT/IRRELEVANT/PARTIAL/EMPTY variants, proportions, 4 builders) +
`scripts/generate_rag_dataset.py`; `scripts/train_lora.py --prompt-mode rag` with `PROMPT_TEMPLATE_RAG`.
Consumes W1 seed JSONL. ~20–25 tests across 4 files.

### 7. retrieval-degradation-visibility — ✅ DONE (PR #562)
`app/observability/retrieval_health.py` (RetrievalReport contract, ContextVar, Prometheus gauge,
`report_payload()`/`current_report()`); reasons reported from `vectorstore.py` (VECTOR_BACKEND_DOWN) and
`kb_store.py` (HASHING_EMBEDDER / EMBEDDING_DIM_MISMATCH / SEARCH_TRUNCATED); surfaced in `/api/kb/health`
and `/ops/health/dependencies`; SLO doc alert rule. The shared foundation for #9 and #10.

### 8. w4-dpo-post-training — ✅ DONE (PR #560)
`app/services/dpo_dataset.py` (RejectStrategy, DPOPair, builders) + `scripts/generate_dpo_pairs.py` +
`scripts/train_dpo.py` (wraps `trl.DPOTrainer`, `trl~=0.11` optional + `tests/stubs/trl/`) +
`app/api/kb_feedback.py` (POST/GET) + `kb_store` feedback table/`store_feedback`/`iter_feedback_pairs`.
Reuses W3's `format_prompt`; Sprint-0 made `apportion_counts` generic + `strip_citations` public.

### 9. retrieval-degradation-customer-loud — ✅ DONE (PR #563)
`RetrievalReportOut` on `AskResponse`; `ask()` + `ask_stream()` populate via `report_payload()`;
severity-colored dismissible banner (`renderDegradation()` in `index.html`) with 5 i18n keys. Builds on #7.

### 10. retrieval-degradation-v1-chat — ✅ DONE (PR #565)
`RetrievalReasonOut`/`RetrievalReportOut` + optional `retrieval` field on `ChatResponse`
(`app/models/__init__.py`); `chat_orchestrator.handle_chat()` populates it. 4 tests. Builds on #7
(independent of #9).

### 11. mypy-safe-pass-clean-files — ✅ DONE (local, unmerged on this branch)
`config.py` dead Pydantic-v1 decorator fallback deleted (137393a); `deps.py` → 0 (5c1a9b9, with d3394ff
keeping bare `Request` for FastAPI injection + targeted `# type: ignore[assignment]`); `file_stats.py` → 0
via `sqlmodel.col()` (6a8e3a8). Verified: both files 0 errors, whole-package 244→226, behavior tests green.

---

## Genuine remaining loose ends (the only "unfinished" items)

These are small and were *not* task steps in any plan — they are gaps surfaced by cross-checking.

1. **Placeholder screenshots** — `docs/screenshots/{chat-with-citations,pdf-viewer-modal,upload-flow}.png`
   are 68-byte 1×1 PNGs. The README references them. They need real captures (requires running the MVP UI).
   *Not source code — an asset gap.*

2. ~~**`AUDIT_LOG_RETENTION_DAYS`** — named in the foundation plan's file-structure table but never given a
   task step and absent from `app/core/config.py`.~~ **DONE 2026-05-31** (TDD): added the
   `audit_log_retention_days` setting (default `0` = disabled/keep-forever, opt-in purging) +
   `purge_audit_log()` helper in `app/core/audit_db.py` + `.env.example` doc. New tests:
   `tests/test_config_audit_retention.py` (3) and 2 added to `tests/test_audit_db.py`. Net-zero mypy (226).
   **Wired end-to-end** via `POST /api/v1/admin/audit/purge` (admin-only; uses `AUDIT_LOG_RETENTION_DAYS`,
   optional `?days=N` override; no-op when disabled) — 5 endpoint tests in `tests/test_admin_audit_endpoint.py`.
   A cron can hit it the same way the `kb-cli health` command polls the API. A non-empty purge is itself
   recorded as an `audit_log_purged` audit event (actor + rows-removed), so destroying history leaves a trail.

3. **This branch is unmerged.** `chore/mypy-safe-pass-deps-filestats` is 6 commits ahead of `main` and
   complete/green. Per the mypy plan's Task 4 Step 4 it should be finished via
   **superpowers:finishing-a-development-branch** (push + open PR titled
   `chore(mypy): safe pass — deps.py + file_stats.py to zero`).

4. ~~**Pre-existing mypy error in `app/core/audit_db.py:71`**~~ **DONE 2026-05-31**: wrapped
   `order_by(col(AuditLog.timestamp).desc())` (the `file_stats.py` idiom). `audit_db.py` is now
   mypy-clean; whole-package count 226 → **225** (45 files).

---

## Recommendation

No code needs to be "finished" — the plans are realized and the suite is green. Actionable next steps,
in priority order:

1. **Finish this branch** → push + PR (item 3). This is the one piece of in-flight workflow.
2. *(optional)* Capture the 3 real screenshots (item 1).
3. *(optional)* Add `AUDIT_LOG_RETENTION_DAYS` + a retention helper if a cleanup job is wanted (item 2).
