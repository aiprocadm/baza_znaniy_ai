# MVP embedder warning banner + `kb_mvp.py` package split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface an admin-console warning when the hashing-embedder fallback is active, then split the 1255-line `app/api/kb_mvp.py` into a behavior-identical `kb_mvp/` package.

**Architecture:** Part A is frontend-only — the admin page queries the already-public `/api/kb/health`, reads `embedder.name`, and reveals an amber banner when it equals `"hash"`. Part B is a pure code move into submodules; `kb_mvp/__init__.py` re-exports the public surface so every existing `from app.api.kb_mvp import X` keeps working. The full pytest suite is the refactor's safety net.

**Tech Stack:** FastAPI (Python 3.13 via `py -3.13`), vanilla JS + JSON i18n in `data/www/`, pytest.

**Environment note:** This repo has no venv. Use `py -3.13` (bare `py -3` resolves to 3.14 without pytest). Mirror CI with `--ignore=backend`. Piping pytest output drops the final summary line — rely on the exit code.

---

## Part A — Hashing-embedder warning banner (admin console)

### Task A1: Lock the health-endpoint contract the banner depends on

**Files:**
- Test: `tests/test_kb_mvp_health_embedder_name.py` (Create)

- [ ] **Step 1: Write the test that asserts `/api/kb/health` exposes `embedder.name`**

```python
"""The admin embedder-warning banner reads health().embedder.name === 'hash'.
Lock that contract so a future refactor can't silently drop the field."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("KB_API_KEY", raising=False)  # keep /health open
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_exposes_embedder_name(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/api/kb/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "embedder" in body
    assert "name" in body["embedder"]
    # Default config with no real backend → hashing fallback.
    assert body["embedder"]["name"] == "hash"
```

- [ ] **Step 2: Run the test — expect PASS (contract already holds today)**

Run: `py -3.13 -m pytest tests/test_kb_mvp_health_embedder_name.py -v`
Expected: PASS. (This is a characterization test — it documents the contract the banner relies on. If it FAILS, the health shape differs from the spec; stop and re-check `app/services/kb_embeddings.py:embedder_status` before touching the frontend.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_kb_mvp_health_embedder_name.py
git commit -m "test(mvp): lock /api/kb/health embedder.name contract for admin banner"
```

---

### Task A2: Add the i18n strings for the banner

**Files:**
- Modify: `data/www/i18n/ru.json` (add 3 keys after the existing `admin.*` block, ~line 52)

- [ ] **Step 1: Add the three keys**

Find the existing admin block:
```json
  "admin.section.upload": "Загрузка документа",
  "admin.section.history": "История загрузок",
  "admin.section.docs": "Управление документами",
```
Add immediately after the `admin.section.docs` line (keep valid JSON — the preceding line needs a trailing comma):
```json
  "admin.embedder.warn.title": "Поиск работает в демо-режиме",
  "admin.embedder.warn.body": "Сейчас используется hashing-эмбеддер — результаты семантического поиска почти случайны. Чтобы поиск заработал нормально, настройте реальный эмбеддер в .env и переиндексируйте базу.",
  "admin.embedder.warn.fix": "Укажите в .env: KB_EMBEDDINGS_BACKEND=ollama (+ OLLAMA_EMBED_MODEL) или =api (+ EMBEDDINGS_API_BASE_URL), затем выполните: kb-cli reindex",
```

- [ ] **Step 2: Verify the JSON still parses**

Run: `py -3.13 -c "import json; json.load(open('data/www/i18n/ru.json', encoding='utf-8')); print('OK')"`
Expected: `OK` (a trailing-comma or missing-comma mistake prints a `json.decoder.JSONDecodeError` instead).

- [ ] **Step 3: Commit**

```bash
git add data/www/i18n/ru.json
git commit -m "i18n(admin): add embedder-warning banner strings"
```

---

### Task A3: Render the banner in the admin console

**Files:**
- Modify: `data/www/admin.html` (markup at top of `<main>` ~line 120; CSS in `<style>`; JS in the existing inline `<script>` IIFE)

- [ ] **Step 1: Add the `.warning` CSS rule**

In the `<style>` block, add after the `.danger` rule (search for `.danger {`):
```css
  .warning {
    margin-bottom: 1.5rem;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    background: #fef3c7;            /* amber-100 */
    border: 1px solid #f59e0b;      /* amber-500 */
    color: #92400e;                 /* amber-800 */
  }
  .warning strong { display: block; margin-bottom: 0.4rem; font-size: 1.05rem; }
  .warning p { margin: 0 0 0.6rem; }
  .warning code {
    display: block;
    white-space: pre-wrap;
    font-size: 0.85rem;
    background: rgba(146, 64, 14, 0.08);
    padding: 0.5rem 0.6rem;
    border-radius: 8px;
  }
  .hidden { display: none; }
```
Note: `admin.html` already uses a `.hidden` class on sections; if a `.hidden { display: none; }` rule already exists in the file, do NOT add a second one — keep just the `.warning` rules.

- [ ] **Step 2: Add the banner markup as the first child of `<main>`**

Immediately after the `<main>` opening tag (before the first `<section>`):
```html
  <div id="embedder-warning" class="warning hidden" role="alert">
    <strong data-i18n="admin.embedder.warn.title">Поиск работает в демо-режиме</strong>
    <p data-i18n="admin.embedder.warn.body">Сейчас используется hashing-эмбеддер — результаты семантического поиска почти случайны.</p>
    <code data-i18n="admin.embedder.warn.fix">KB_EMBEDDINGS_BACKEND=ollama (+ OLLAMA_EMBED_MODEL) или =api; затем: kb-cli reindex</code>
  </div>
```

- [ ] **Step 3: Add the health probe to the inline script**

Inside the existing `<script>` IIFE, near the top (right after the `const ... = document.getElementById(...)` block, before the token logic), add:
```js
  // Reveal the embedder warning if the server reports the hashing fallback.
  // /api/kb/health is public (no token needed) — run this immediately.
  fetch('/api/kb/health')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (d) {
      if (d && d.embedder && d.embedder.name === 'hash') {
        var el = document.getElementById('embedder-warning');
        if (el) { el.classList.remove('hidden'); }
      }
    })
    .catch(function () { /* never break the admin page on a health probe failure */ });
```
The i18n loader (`/i18n/_loader.js`, loaded at the bottom) fills the `data-i18n` text on `DOMContentLoaded`; the inline fallback text covers the brief window before that and the offline case.

- [ ] **Step 4: Start the MVP dev server in the background**

Run: `py -3.13 -m uvicorn scripts.dev_server_mvp:app --port 8001`
(Start it with the Bash tool's `run_in_background`, or via the preview tools. Confirm it is serving before continuing.)

- [ ] **Step 5: Verify the banner SHOWS on the hashing default**

Using the browser-preview tools: open `http://localhost:8001/admin.html`, then `preview_snapshot` / `preview_screenshot`.
Expected: the amber "Поиск работает в демо-режиме" banner is visible at the top. Also `preview_console_logs` shows no errors.

- [ ] **Step 6: Verify the banner HIDES with a real embedder**

Stop the server. Restart it with a non-hashing backend:
Run: `$env:KB_EMBEDDINGS_BACKEND="st"; py -3.13 -m uvicorn scripts.dev_server_mvp:app --port 8001`
Reload `http://localhost:8001/admin.html` in preview.
Expected: NO banner (the `embedder-warning` div stays `hidden`). If the `st` backend is unavailable in the environment and falls back to hash, instead confirm via the network panel that `/api/kb/health` returned `embedder.name !== "hash"`; if it can't be made non-hash locally, document that step 5 + the A1 contract test together cover the behavior. Stop the server and unset the env var afterward (`Remove-Item Env:KB_EMBEDDINGS_BACKEND`).

- [ ] **Step 7: Commit**

```bash
git add data/www/admin.html
git commit -m "feat(mvp): warn in admin console when hashing embedder is active"
```

---

## Part B — Split `app/api/kb_mvp.py` into a `kb_mvp/` package

**Do not start Part B until Part A is committed.** Part B is a pure move: copy each symbol verbatim from the current `kb_mvp.py` into its new home — no logic, string, route-path, or signature changes.

### Task B0: Record the green baseline

**Files:** none (measurement only)

- [ ] **Step 1: Run the full suite and record the pass count**

Run: `py -3.13 -m pytest -q --ignore=backend; echo "EXIT=$LASTEXITCODE"`
Expected: `EXIT=0`. Write down the "N passed" number (this is the target B must preserve). If the baseline is NOT green, stop — fix or note the pre-existing failures before refactoring, otherwise you can't tell what the refactor broke.

---

### Task B1: Create the package skeleton with `schemas.py`

**Files:**
- Create: `app/api/kb_mvp/__init__.py` (temporary minimal version, replaced in B7)
- Create: `app/api/kb_mvp/schemas.py`
- Note: `app/api/kb_mvp.py` still exists at this point. A file and a package directory of the same name cannot coexist — so this task is "prepare the new files in a scratch location is NOT possible". Instead: do B1–B6 by creating the new modules' CONTENT in your editor/notes, then in B7 perform the swap atomically. **Concretely:** delete `app/api/kb_mvp.py` and create the `app/api/kb_mvp/` directory only at the START of B1, immediately creating `__init__.py` so imports keep resolving. Follow the steps below in order without running the suite until B7.

- [ ] **Step 1: Convert the module into a package directory**

```bash
git rm app/api/kb_mvp.py
mkdir app/api/kb_mvp
```
(The code from the old file is preserved in git history / your open buffer — you will paste each section into its new module.)

- [ ] **Step 2: Create `app/api/kb_mvp/schemas.py`**

Paste verbatim, from the old file, lines 64–235 (all Pydantic model classes: `DocumentCreate` … `ConversationDetail`) under this header:
```python
"""Pydantic request/response models for the MVP /api/kb endpoints."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.services.kb_store import (
    DEFAULT_HISTORY_LIMIT,
    MAX_CONVERSATION_TITLE,
    MAX_QUERY_LEN,
    MAX_TEXT_LEN,
)

# <-- paste class DocumentCreate ... class ConversationDetail here, unchanged -->
```

- [ ] **Step 3: Create a temporary `app/api/kb_mvp/__init__.py`**

```python
"""MVP knowledge-base endpoints mounted under /api/kb (package split of the
former kb_mvp.py module). Public import surface is preserved here."""
```
(This stub keeps the directory importable while you build the other modules. It is replaced wholesale in B7.)

---

### Task B2: Create `common.py` (shared infra, converters, file helpers)

**Files:**
- Create: `app/api/kb_mvp/common.py`

- [ ] **Step 1: Create the module with this header and paste the listed symbols**

```python
"""Shared infrastructure for the MVP /api/kb endpoint modules:
routers, constants, store accessor, model converters, file parsing."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.kb_auth import require_api_key
from app.services.kb_store import (
    Conversation as StoredConversation,
    Document as StoredDocument,
    KnowledgeBaseStore,
    Message as StoredMessage,
    SearchHit,
    get_store,
)

from .schemas import ConversationOut, DocumentOut, HitOut, MessageOut

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["kb-mvp"])
public = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB
SUPPORTED_UPLOAD_EXT = {
    "pdf", "docx", "pptx", "xlsx", "txt", "md", "markdown", "html", "htm",
}

# <-- paste, unchanged, from the old file: -->
#   _store_for                      (lines 238-249)
#   _doc_to_out                     (252-262)
#   _hit_to_out                     (265-276)
#   _conversation_to_out            (279-286)
#   _sources_payload_to_hit_out     (289-319)
#   _message_to_out                 (322-332)
#   _format_history                 (335-346)
#   _extension_for                  (477-481)
#   _decode_text                    (484-490)
#   _resolve_data_dir               (493-510)
#   _resolve_kb_files_dir           (513-517)
#   _parse_file_bytes               (520-560)
#   _parse_file_bytes_with_pages    (563-610)
```

Note: the converter functions reference `DocumentOut`, `HitOut`, `ConversationOut`, `MessageOut` — now imported from `.schemas`. `_parse_file_bytes*` reference `HTTPException`/`status` (imported above) and do a local `from app.ingest.chunking import parse_document` (keep that local import inside the function unchanged).

---

### Task B3: Create `rag.py` (retrieval + answer generation)

**Files:**
- Create: `app/api/kb_mvp/rag.py`

- [ ] **Step 1: Create the module**

```python
"""Retrieval (bi-encoder + optional cross-encoder rerank) and answer
generation for the MVP /api/kb endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import Request

from app.services import kb_llm, kb_rerank
from app.services.kb_store import KnowledgeBaseStore, SearchHit

from .common import LOGGER
from .schemas import RerankInfo

# <-- paste, unchanged, from the old file: -->
#   _retrieve_with_rerank   (349-383)
#   _format_context         (386-391)
#   _extractive_answer      (394-403)
#   _RAG_SYSTEM_PROMPT      (406-412)
#   _build_rag_prompt       (415-426)
#   _generate_answer        (429-474)
```

Note: `_RAG_SYSTEM_PROMPT` MUST stay byte-identical — it is drift-tested against `app/eval/generation_eval.py`. Copy it character-for-character.

---

### Task B4: Create `health.py` (public endpoints)

**Files:**
- Create: `app/api/kb_mvp/health.py`

- [ ] **Step 1: Create the module**

```python
"""Public (no-auth) MVP endpoints: /health and /providers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import Request

from app.api.kb_auth import auth_status
from app.observability import retrieval_health
from app.services import kb_embeddings, kb_llm, kb_rerank

from .common import public, _store_for

# <-- paste, unchanged, from the old file: -->
#   @public.get("/health")  def health(...)   (613-693)
#   @public.get("/providers") def providers(...) (696-700)
```

Note: keep the local `import shutil as _shutil` / `import sqlite3 as _sqlite3` inside `health()` unchanged.

---

### Task B5: Create `documents.py` and `search.py`

**Files:**
- Create: `app/api/kb_mvp/documents.py`
- Create: `app/api/kb_mvp/search.py`

- [ ] **Step 1: Create `documents.py`**

```python
"""Document CRUD + upload endpoints (protected)."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, List, Optional

from fastapi import File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from .common import (
    LOGGER,
    MAX_UPLOAD_BYTES,
    SUPPORTED_UPLOAD_EXT,
    protected,
    _doc_to_out,
    _extension_for,
    _parse_file_bytes_with_pages,
    _resolve_data_dir,
    _resolve_kb_files_dir,
    _store_for,
)
from .schemas import DocumentCreate, DocumentListItem, DocumentOut

# <-- paste, unchanged, from the old file: -->
#   create_document       (703-716)
#   upload_document       (719-797)
#   list_documents        (800-815)
#   get_document          (818-826)
#   get_document_file     (829-869)
#   delete_document       (872-911)
```

- [ ] **Step 2: Create `search.py`**

```python
"""Similarity search endpoint (protected)."""
from __future__ import annotations

from fastapi import Request

from .common import protected, _hit_to_out, _store_for
from .rag import _retrieve_with_rerank
from .schemas import SearchRequest, SearchResponse

# <-- paste, unchanged, from the old file: -->
#   search_documents      (914-924)
```

---

### Task B6: Create `chat.py` (ask, streaming, conversations)

**Files:**
- Create: `app/api/kb_mvp/chat.py`

- [ ] **Step 1: Create the module**

```python
"""Ask / streaming-ask / conversation endpoints (protected)."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, List, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.observability import retrieval_health
from app.services import kb_llm
from app.services.kb_store import Conversation as StoredConversation

from .common import (
    LOGGER,
    protected,
    _conversation_to_out,
    _format_history,
    _hit_to_out,
    _message_to_out,
    _store_for,
)
from .rag import (
    _RAG_SYSTEM_PROMPT,
    _build_rag_prompt,
    _extractive_answer,
    _generate_answer,
    _retrieve_with_rerank,
)
from .schemas import (
    AskRequest,
    AskResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationRename,
    RetrievalReportOut,
)

# <-- paste, unchanged, from the old file: -->
#   _sse_event              (1000-1004)
#   _stream_extractive      (1007-1010)
#   _stream_legacy          (1013-1024)   (keep local `import asyncio` inside)
#   ask                     (927-992)
#   ask_stream              (1027-1176)
#   create_conversation     (1184-1194)
#   list_conversations      (1197-1203)
#   get_conversation_detail (1206-1218)
#   rename_conversation     (1221-1231)
#   delete_conversation     (1234-1241)
```

---

### Task B7: Assemble `__init__.py` with the preserved public surface

**Files:**
- Modify: `app/api/kb_mvp/__init__.py` (replace the B1 stub entirely)

- [ ] **Step 1: Write the final `__init__.py`**

```python
"""MVP knowledge-base endpoints mounted under ``/api/kb``.

Auth-free contract for the simple frontend in ``data/www/index.html``.
The full multi-tenant API stays under ``/api/v1/*``. ``/ask`` prefers
:func:`app.services.kb_llm.select_provider`, falls back to
``state.llm_provider`` (legacy), then to an extractive answer.

This package is the split of the former single-file ``kb_mvp.py``. The
public import surface (``router`` plus the helpers/models/prompt that
tests and ``app/eval`` import) is re-exported here so
``from app.api.kb_mvp import X`` keeps working unchanged.
"""
from __future__ import annotations

from .common import router, public, protected

# Importing the endpoint modules registers their routes on the shared
# ``public`` / ``protected`` routers via decorator side-effects.
from . import health, documents, search, chat  # noqa: F401,E402

# Wire sub-routers into the top-level router (same order/paths as before).
router.include_router(public)
router.include_router(protected)

# W4 — live feedback collection endpoints
from app.api.kb_feedback import router as kb_feedback_router  # noqa: E402

router.include_router(kb_feedback_router)

# ---- Re-export the public import surface (back-compat with the old module) ----
from .common import (  # noqa: E402,F401
    LOGGER,
    MAX_UPLOAD_BYTES,
    SUPPORTED_UPLOAD_EXT,
    _conversation_to_out,
    _decode_text,
    _doc_to_out,
    _extension_for,
    _format_history,
    _hit_to_out,
    _message_to_out,
    _parse_file_bytes,
    _parse_file_bytes_with_pages,
    _resolve_data_dir,
    _resolve_kb_files_dir,
    _sources_payload_to_hit_out,
    _store_for,
)
from .rag import (  # noqa: E402,F401
    _RAG_SYSTEM_PROMPT,
    _build_rag_prompt,
    _extractive_answer,
    _format_context,
    _generate_answer,
    _retrieve_with_rerank,
)
from .health import health, providers  # noqa: E402,F401
from .documents import (  # noqa: E402,F401
    create_document,
    delete_document,
    get_document,
    get_document_file,
    list_documents,
    upload_document,
)
from .search import search_documents  # noqa: E402,F401
from .chat import (  # noqa: E402,F401
    ask,
    ask_stream,
    create_conversation,
    delete_conversation,
    get_conversation_detail,
    list_conversations,
    rename_conversation,
    _sse_event,
    _stream_extractive,
    _stream_legacy,
)
from .schemas import (  # noqa: E402,F401
    AskRequest,
    AskResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationRename,
    DocumentCreate,
    DocumentListItem,
    DocumentOut,
    HitOut,
    MessageOut,
    RerankInfo,
    RetrievalReasonOut,
    RetrievalReportOut,
    SearchRequest,
    SearchResponse,
)

__all__ = ["router"]
```

Note: `__all__ = ["router"]` matches the original (it only governs `import *`; the explicit re-imports above are what keep `from app.api.kb_mvp import _RAG_SYSTEM_PROMPT` etc. working).

---

### Task B8: Verify behavior is identical and clean up

**Files:** none (verification)

- [ ] **Step 1: Confirm the package imports and the public surface resolves**

Run:
```
py -3.13 -c "from app.api.kb_mvp import router, _RAG_SYSTEM_PROMPT, _parse_file_bytes_with_pages, _build_rag_prompt, _format_context, ask, RetrievalReportOut; print('imports OK')"
```
Expected: `imports OK`. An `ImportError` here names the symbol you forgot to re-export in B7 — add it and re-run.

- [ ] **Step 2: Run the full suite — must match the B0 baseline**

Run: `py -3.13 -m pytest -q --ignore=backend; echo "EXIT=$LASTEXITCODE"`
Expected: `EXIT=0` and the SAME "N passed" count recorded in B0. A new failure almost always means a missing re-export (fix B7) or a copy-paste omission (compare the module against the old file's line ranges). Do not edit tests to make them pass.

- [ ] **Step 3: Lint and style the new package**

Run: `py -3.13 -m ruff check app/api/kb_mvp; py -3.13 -m black --check app/api/kb_mvp`
Expected: both clean. If ruff flags an unused import in a submodule, it is genuinely unused — remove it (the `__init__` re-exports use `# noqa: F401` deliberately; submodules should not).

- [ ] **Step 4: Confirm no stale references to the old module path remain**

Run: `py -3.13 -m pytest tests/test_eval_generation.py tests/test_parse_file_bytes_with_pages.py tests/test_kb_mvp_ask_retrieval.py -v`
Expected: PASS (these import the underscore-prefixed internals and the router directly — the tightest check on the re-export surface).

- [ ] **Step 5: Commit**

```bash
git add app/api/kb_mvp
git commit -m "refactor(kb-mvp): split kb_mvp.py into a kb_mvp/ package

Pure code move — zero behavior, route, or response-shape changes. The
public import surface (router + helpers/models/prompt imported by tests
and app/eval) is re-exported from kb_mvp/__init__.py."
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** A1 locks the health contract; A2/A3 implement the banner (admin-only, amber, fix command) — Part A covered. B1–B7 create exactly the 8 modules named in the spec; B8 verifies the preserved import surface and identical behavior — Part B covered. Sequence (A then B) and separate commits — covered by section order and the explicit per-part commits.
- **Placeholder scan:** The `<-- paste lines X-Y -->` markers are explicit, line-referenced move instructions against the read source file, not vague TODOs. All new module headers and the full `__init__.py` are written out completely.
- **Type/name consistency:** Re-export names in B7 match the symbols defined in B2–B6 and the test imports found in the repo (`router`, `_RAG_SYSTEM_PROMPT`, `_parse_file_bytes_with_pages`, `_build_rag_prompt`, `_format_context`, `ask`, `RetrievalReportOut`). Router/`public`/`protected` live solely in `common.py`; every endpoint module decorates those same instances.
- **Risk note:** No circular imports — dependency order is schemas → common → rag → {health, documents, search, chat} → `__init__`.
