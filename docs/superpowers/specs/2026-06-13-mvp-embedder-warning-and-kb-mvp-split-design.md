# Design — MVP embedder warning banner + `kb_mvp.py` package split

**Date:** 2026-06-13
**Status:** Approved (brainstorming)
**Scope:** Two independent, sequential changes to the single-tenant MVP path.

This spec covers two unrelated improvements that were requested together. They are
implemented and committed **separately** (a user-facing feature, then a pure
refactor) so the refactor can be reverted without losing the feature.

---

## Part A — Hashing-embedder warning banner (admin console)

### Problem

`KB_EMBEDDINGS_BACKEND` defaults to a **hashing embedder** when no real backend is
configured (`app/services/kb_embeddings.py`). Hashing embeddings give near-random
semantic matches. A new operator who skips the docs gets nonsense search results
and blames the product. There is a server-side `LOGGER.warning` today, but nothing
visible in the UI.

### Goal

When the system is running on the hashing fallback, the **admin console**
(`data/www/admin.html`) shows a prominent amber banner explaining the situation and
giving the exact fix command. When a real embedder is configured, no banner appears.

Decision (from brainstorming): banner lives **only in the admin console**, not the
end-user chat UI — the admin is the person who can actually fix the `.env` and run
the reindex.

### Existing infrastructure we reuse (no backend changes)

- `GET /api/kb/health` is a **public** endpoint (no auth — `public` sub-router in
  `app/api/kb_mvp.py`). It already returns an `embedder` object via
  `kb_embeddings.embedder_status()`, whose `name` field is `"hash"` for the
  hashing fallback.
- `admin.html` already loads `/i18n/_loader.js`, which replaces `textContent` of
  any `[data-i18n]` element from `data/www/i18n/ru.json` after `DOMContentLoaded`,
  and exposes a `window.t(key, fallback, vars)` helper.

### Changes (frontend only — 2 files)

1. **`data/www/admin.html`**
   - Add a hidden banner element at the top of `<main>`, before the first
     `<section>`:
     ```html
     <div id="embedder-warning" class="warning hidden" role="alert">
       <strong data-i18n="admin.embedder.warn.title">Поиск работает в демо-режиме</strong>
       <p data-i18n="admin.embedder.warn.body">...</p>
       <code data-i18n="admin.embedder.warn.fix">KB_EMBEDDINGS_BACKEND=ollama ...</code>
     </div>
     ```
   - Add a `.warning` CSS rule (amber background `#fef3c7`, amber-700 border/text,
     rounded, padded; dark-mode-friendly via the existing `color-scheme` approach).
   - Add a small inline JS block (runs on load, no auth needed):
     ```js
     fetch('/api/kb/health')
       .then(r => r.ok ? r.json() : null)
       .then(d => {
         if (d && d.embedder && d.embedder.name === 'hash') {
           document.getElementById('embedder-warning').classList.remove('hidden');
         }
       })
       .catch(() => {});  // never break the page on a health probe failure
     ```
     This runs independently of the admin token (health is public). Failure is
     swallowed silently — a missing banner is acceptable; a broken admin page is not.

2. **`data/www/i18n/ru.json`** — add keys:
   - `admin.embedder.warn.title` — "Поиск работает в демо-режиме"
   - `admin.embedder.warn.body` — explanation that hashing = near-random results.
   - `admin.embedder.warn.fix` — the exact fix: set `KB_EMBEDDINGS_BACKEND=ollama`
     (+ `OLLAMA_EMBED_MODEL`) or `=api` (+ `EMBEDDINGS_API_BASE_URL`), then run
     `kb-cli reindex`.

### Behavior

- Hashing backend → banner visible.
- Real backend (ollama/api/st) → `embedder.name != "hash"` → banner stays hidden.
- Health endpoint unreachable / network error → banner stays hidden, page works.

### Verification

- Manual / browser-preview: start the MVP dev server
  (`uvicorn scripts.dev_server_mvp:app --port 8001`), open `/admin.html`.
  - Default env (hashing) → screenshot showing the banner.
  - With `KB_EMBEDDINGS_BACKEND=st` (or a stub real embedder) → reload →
    screenshot showing no banner.
- Confirm `GET /api/kb/health` returns `embedder.name` without auth (it already
  does; existing tests like `test_kb_mvp_health_retrieval.py` cover the endpoint).

### Out of scope (A)

- No banner in the end-user chat UI (`data/www/index.html`).
- No dismiss/snooze button — the banner auto-disappears once the embedder is fixed,
  so persistence-until-fixed is the desired behavior.
- No backend changes.

---

## Part B — Split `app/api/kb_mvp.py` (1255 LoC) into a `kb_mvp/` package

### Problem

`app/api/kb_mvp.py` is a 1255-line "god file" mixing Pydantic schemas, RAG prompt
logic, file parsing, and four feature areas of HTTP endpoints. Large files are
harder to read, review, and test.

### Goal

Convert the single module into a package `app/api/kb_mvp/` of focused submodules,
**with zero change to public behavior or import paths**. Every existing
`from app.api.kb_mvp import X` must keep working.

### Hard constraint: preserve the public import surface

Code outside the module imports these names from `app.api.kb_mvp` and they MUST
remain importable from the package root (`kb_mvp/__init__.py` re-exports them):

- `router` — used by `app/api/router.py` and ~10 test files.
- `_RAG_SYSTEM_PROMPT` — imported by `tests/test_eval_generation.py` and mirrored
  (drift-tested, must stay byte-identical) by `app/eval/generation_eval.py`.
- `_parse_file_bytes_with_pages` — imported by
  `tests/test_parse_file_bytes_with_pages.py`.
- Also keep importable (referenced/mirrored by `app/eval/*`): `_build_rag_prompt`,
  `_format_context`, `ask`, and all Pydantic model classes
  (`RetrievalReportOut`, etc.).

The 28 test files that touch `kb_mvp` are the safety net (characterization tests).

### Target layout

```
app/api/kb_mvp/
  __init__.py     # assembles `router`, includes kb_feedback, re-exports public surface
  schemas.py      # all Pydantic models (DocumentCreate … ConversationDetail)
  common.py       # LOGGER, MAX_UPLOAD_BYTES, SUPPORTED_UPLOAD_EXT,
                  #   public/protected APIRouter objects, _store_for,
                  #   converters (_doc_to_out, _hit_to_out, _conversation_to_out,
                  #   _sources_payload_to_hit_out, _message_to_out, _format_history),
                  #   path resolvers (_resolve_data_dir, _resolve_kb_files_dir),
                  #   file helpers (_extension_for, _decode_text, _parse_file_bytes,
                  #   _parse_file_bytes_with_pages)
  rag.py          # _retrieve_with_rerank, _format_context, _extractive_answer,
                  #   _RAG_SYSTEM_PROMPT, _build_rag_prompt, _generate_answer
  documents.py    # create_document, upload_document, list_documents, get_document,
                  #   get_document_file, delete_document
  search.py       # search_documents
  chat.py         # ask, ask_stream, _sse_event, _stream_extractive, _stream_legacy,
                  #   conversation CRUD endpoints
  health.py       # health, providers (public endpoints)
```

### Mechanics

- `common.py` owns the **single** `public` and `protected` `APIRouter` instances.
  Every endpoint module imports those same objects and decorates them — so all
  routes register on the shared routers regardless of file.
- `__init__.py` imports every endpoint submodule (import side-effect = route
  registration), then builds the top-level `router` exactly as the old file did
  (include `public`, `protected`, and the kb_feedback router under the same paths),
  and finally re-exports the public surface listed above.
- Pure code movement. No renames, no signature changes, no logic edits. Prompt
  strings and route paths byte-identical.

### Verification (TDD safety net)

1. **Baseline:** run the full suite first and record the pass count:
   `py -3.13 -m pytest -q --ignore=backend` → note "N passed".
2. Perform the split.
3. **Re-run** the same command → **same N passed, zero new failures**. A red test
   almost certainly means a missing re-export in `__init__.py` — fix the export,
   not the test.
4. `py -3.13 -m ruff check app/api/kb_mvp` and `py -3.13 -m black --check
   app/api/kb_mvp` clean.

### Out of scope (B)

- No moving business logic into `app/services` (that was the rejected "deep"
  option — higher risk).
- No merging of the `/api/kb/*` and `/api/v1/*` paths (explicit anti-pattern per
  `docs/architecture.md`).
- No behavior, route, or response-shape changes.

---

## Sequence & commits

1. Part A → commit (`feat(mvp): warn in admin console when hashing embedder is active`).
2. Part B → commit (`refactor(kb-mvp): split kb_mvp.py into a kb_mvp/ package`).

Separate commits so the refactor is independently revertible.
