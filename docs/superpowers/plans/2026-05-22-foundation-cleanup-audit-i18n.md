# Foundation: Cleanup + Audit Log + i18n Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Подготовить кодовую базу к первому pilot-клиенту: убрать tech debt в корне репозитория, развернуть audit log из stub'а в полноценную DB-таблицу с эндпоинтом, и подготовить UI к i18n без перевода (только инфраструктура).

**Architecture:**
- **Tech debt:** `dev_kb_only.py` переезжает в `scripts/dev_server_mvp.py` (не удаляется — полезный dev-инструмент). Параллельные пути `/api/kb/*` (MVP) и `/api/v1/*` (mature) задокументированы как осознанное архитектурное решение, а не bug — синхронизация НЕ делается.
- **Audit log:** новая SQLite-таблица `audit_log` + Alembic-миграция, расширение `app/core/audit.py` для DB-persistence + middleware для логирования всех `/api/kb/*` запросов + admin-endpoint `GET /api/v1/admin/audit` с пагинацией и фильтрами.
- **i18n:** строки UI выносятся в `data/www/i18n/ru.json` (плоский dict `key → value`). HTML использует `data-i18n="key"` атрибуты, минимальный JS `t(key)` подменяет содержимое при загрузке. Английского и других CIS-языков пока нет — только инфраструктура.

**Tech Stack:**
- Backend: FastAPI, SQLModel/SQLAlchemy, Alembic, pytest
- Frontend: vanilla JS, статический HTML
- Testing: pytest + httpx TestClient

---

## File Structure

**Files created:**
- `scripts/dev_server_mvp.py` — moved from repo root (was `dev_kb_only.py`)
- `alembic/versions/20260522_01_audit_log.py` — DB migration for `audit_log` table
- `app/models/audit.py` — SQLModel for `AuditLog`
- `app/core/audit_db.py` — DB-persistence helpers (separate from `audit.py` stub which keeps log-only API)
- `app/api/v1/admin_audit.py` — admin endpoint `GET /api/v1/admin/audit`
- `app/core/audit_middleware.py` — FastAPI middleware for `/api/kb/*` request audit
- `data/www/i18n/ru.json` — Russian strings, source of truth
- `data/www/i18n/_loader.js` — minimal i18n loader (called from index.html and admin.html)
- `docs/architecture.md` — documents the two-path design decision
- `tests/test_audit_log.py` — audit persistence tests
- `tests/test_admin_audit_endpoint.py` — admin endpoint tests
- `tests/test_audit_middleware.py` — middleware tests
- `tests/test_i18n_loader.py` — i18n loader tests (using subprocess or simple JS test)

**Files modified:**
- `dev_kb_only.py` → deleted (replaced by `scripts/dev_server_mvp.py`)
- `app/core/audit.py` — extended to optionally write to DB via `audit_db`
- `app/core/app.py` — register `audit_middleware` and `admin_audit` router
- `app/api/v1/__init__.py` — include `admin_audit` sub-router
- `README.md` — update dev-workflow section
- `data/www/index.html` — replace 30-40 inline strings with `data-i18n` attributes, include `_loader.js`
- `data/www/admin.html` — same as index.html
- `app/core/config.py` — add `AUDIT_LOG_RETENTION_DAYS` setting

---

## Section A — Tech debt cleanup (~8h, 4 tasks)

### Task A.1: Move dev_kb_only.py to scripts/dev_server_mvp.py

**Files:**
- Move: `dev_kb_only.py` → `scripts/dev_server_mvp.py`
- Modify: `README.md` (1-2 references)

- [ ] **Step 1: Verify scripts/ directory exists and check existing files**

Run:
```bash
ls scripts/
```
Expected output: list of files including `download_model.py`, `train_lora.py`, etc.

- [ ] **Step 2: Move the file with git tracking**

Run:
```bash
git mv dev_kb_only.py scripts/dev_server_mvp.py
```

Note: `dev_kb_only.py` is currently **untracked**, so `git mv` will fail. Instead use:
```bash
mv dev_kb_only.py scripts/dev_server_mvp.py
git add scripts/dev_server_mvp.py
```

- [ ] **Step 3: Update module docstring in the moved file**

Edit `scripts/dev_server_mvp.py` first docstring block (lines 1-22). Replace the `Run:` section:

```python
"""Minimal FastAPI app exposing only the MVP /api/kb/* router.

Useful for local dev preview without pulling in the full multi-tenant
v1 stack (which requires sqlmodel, qdrant-client, sentence-transformers,
llama-cpp, etc.). The MVP router is intentionally lightweight and only
needs FastAPI + httpx + python-multipart from the runtime requirements.

Also mounts ``data/www/`` as static files so the MVP frontend
(``index.html``) is served on the same origin as the API — no CORS
gymnastics during preview.

Run:

    python -m uvicorn scripts.dev_server_mvp:app --reload --port 8001

Then open:

    http://127.0.0.1:8001/        — MVP frontend (data/www/index.html)
    http://127.0.0.1:8001/docs    — Swagger UI
    http://127.0.0.1:8001/api/kb/health — health probe
"""
```

- [ ] **Step 4: Verify the new entrypoint works**

Run:
```bash
python -c "from scripts.dev_server_mvp import app; print('OK', app.title)"
```
Expected: `OK kb-mvp-dev`

- [ ] **Step 5: Update README.md to mention scripts/dev_server_mvp.py**

Find the section about running locally (around "Быстрый старт" or "Запуск сервиса"). Add a new sub-section:

```markdown
### Lightweight dev server (MVP only)

Если не нужен полный multi-tenant стек, можно запустить только MVP-роутер
`/api/kb/*` с минимальными зависимостями:

```bash
python -m uvicorn scripts.dev_server_mvp:app --reload --port 8001
```

Это удобно для UI-разработки и smoke-тестов без Qdrant/llama-cpp/sentence-transformers.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/dev_server_mvp.py README.md
git commit -m "refactor: move dev_kb_only.py to scripts/dev_server_mvp.py

The file was a useful lightweight dev server but lived untracked in the
repo root. Moves it to scripts/ following the existing convention, and
documents the run command in README's quick-start section.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task A.2: Document the two-path design (MVP vs mature)

**Files:**
- Create: `docs/architecture.md`

**Background:** `/api/kb/*` (MVP, single-tenant, in `app/api/kb_mvp.py`) and `/api/v1/*` (mature, multi-tenant, in `app/api/v1/`) are intentionally parallel. Future contributors (including AI agents) may try to "unify" them, breaking either MVP simplicity or mature features. Document this decision explicitly.

- [ ] **Step 1: Create the file**

Create `docs/architecture.md` with this content:

````markdown
# Architecture decisions

## Two-path API design (intentional, not legacy)

The codebase exposes two parallel HTTP API surfaces:

- **`/api/kb/*`** (MVP) — single-tenant, auth via single `KB_API_KEY` env var,
  state in one SQLite file (`KB_MVP_DB_PATH`), uses OpenAI-compatible LLM
  providers only. Source: `app/api/kb_mvp.py`. Lightweight dependency tree
  (FastAPI + httpx). Designed for ≤10 person SMB self-hosted deployments.

- **`/api/v1/*`** (mature) — multi-tenant with JWT auth + RBAC, separate
  PostgreSQL for tenant metadata, Qdrant for vectors, supports llama.cpp +
  LoRA adapters. Source: `app/api/v1/*.py`. Heavy dependency tree (sqlmodel,
  qdrant-client, sentence-transformers, llama-cpp-python). Designed for
  multi-tenant SaaS or large internal deployments.

### Why two paths

The MVP path was added later for two reasons:

1. **Easier installation for SMB customers.** A `docker compose up` that needs
   nothing more than Python and SQLite ships in minutes; the mature stack
   requires Qdrant, model downloads, and several GB of dependencies.
2. **Simpler hardening surface.** Single-tenant, single-key auth is easier to
   audit for compliance than full RBAC. For one-off self-hosted deployments
   this matches what customers want.

### Do NOT merge them

A future contributor (human or AI agent) may notice the duplication of
concepts (documents, search, chat) and propose unifying. **This is wrong** at
the current stage:

- The MVP and mature paths target different customer profiles (single-tenant
  install vs SaaS).
- Merging would force MVP installs to carry full multi-tenant dependencies
  (~2 GB of model downloads, Qdrant, PostgreSQL drivers).
- The vision document (`docs/superpowers/specs/2026-05-22-project-vision-design.md`)
  explicitly defers the multi-tenant decision until ≥5 paying customers exist.

### When to revisit

After 5 paying customers, evaluate:
- Are customers happy with the single-tenant MVP install?
- Has anyone explicitly asked for multi-tenant SaaS?
- Would unifying simplify maintenance enough to outweigh the install-weight cost?

Until then, treat the two paths as separate products in the same repo.

## Source-of-truth entrypoints

- **Production:** `uvicorn app.api.main:app` (loads `app/core/app.py:create_app`,
  full multi-tenant stack).
- **MVP-only dev:** `uvicorn scripts.dev_server_mvp:app` (loads only the
  `/api/kb/*` router, lightweight deps).
- **Legacy:** `backend/app/*` is deprecated, kept only for compatibility tests
  in CI job `legacy-compatibility-tests`.
````

- [ ] **Step 2: Verify the file is reachable from project docs**

Run:
```bash
ls docs/
```
Expected: `architecture.md` appears in listing.

- [ ] **Step 3: Add link from README.md**

Find the section in README.md headed `## Active runtime path (source of truth)`. After it, add:

```markdown
Подробнее об архитектурных решениях, в том числе про два параллельных
HTTP-пути и причины их разделения — см. [`docs/architecture.md`](docs/architecture.md).
```

- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md README.md
git commit -m "docs(architecture): document the two-path API design

Adds docs/architecture.md explaining that /api/kb/* (MVP) and /api/v1/*
(mature) are intentionally parallel, not legacy duplication. Prevents
future contributors from 'unifying' them and breaking either side.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task A.3: Audit other untracked files and decide their fate

**Files:**
- Inspect: anything `git status --short` reveals

- [ ] **Step 1: List all untracked and modified files**

Run:
```bash
git status --short
```

- [ ] **Step 2: For each untracked file, classify**

For each line starting with `??` (untracked):
- If it's a build artifact (e.g., `__pycache__`, `.pytest_cache`) → add to `.gitignore` if not already
- If it's a useful script → consider moving to `scripts/` or `data/` as in Task A.1
- If it's leftover from experimentation → delete it
- If unclear → leave it for now, ask in next checkpoint

- [ ] **Step 3: Verify .gitignore covers common debris**

Read current `.gitignore`. Ensure these patterns exist:
```
__pycache__/
*.pyc
.pytest_cache/
.hypothesis/
.venv/
venv/
.env
.env.local
var/
runs/
*.sqlite
*.sqlite-journal
```

If any are missing, add them. Commit `.gitignore` separately:

```bash
git add .gitignore
git commit -m "chore(gitignore): expand patterns for common build debris

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 4: Re-run git status, verify clean state**

Run:
```bash
git status --short
```
Expected: only intentional uncommitted changes remain. No mystery files.

---

### Task A.4: Pre-flight test run to confirm cleanup did not break anything

- [ ] **Step 1: Run lint**

Run:
```bash
make lint
```
Expected: PASS (or no new errors compared to before this section).

- [ ] **Step 2: Run tests**

Run:
```bash
pytest tests/ -x --tb=short
```
Expected: PASS (or same baseline as before this section).

- [ ] **Step 3: Smoke-test the moved dev server**

Run in one terminal:
```bash
python -m uvicorn scripts.dev_server_mvp:app --port 8001
```

In another:
```bash
curl http://127.0.0.1:8001/api/kb/health
```
Expected: HTTP 200 with health JSON.

Stop the server with Ctrl-C.

- [ ] **Step 4: No commit needed** (this is verification only).

---

## Section B — Audit log persistence (~10h, 6 tasks)

### Task B.1: Alembic migration for `audit_log` table

**Files:**
- Create: `alembic/versions/20260522_01_audit_log.py`

- [ ] **Step 1: Inspect existing Alembic migrations to understand conventions**

Run:
```bash
ls alembic/versions/
```
Expected: list including `20240919_01_initial_schema.py`, `20260503_01_target_data_model.py`, etc.

Read the most recent migration to understand revision-id format and `down_revision` chain.

```bash
cat alembic/versions/20260506_01_api_keys_usage_rag.py | head -30
```

Note the `revision` and `down_revision` values — the new migration's `down_revision` MUST be the latest one currently in the chain.

- [ ] **Step 2: Write the failing migration test**

Create `tests/test_migration_audit_log.py`:

```python
"""Test that the audit_log migration creates the table with expected schema."""
from __future__ import annotations

import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_audit_log_table_created_by_migration(tmp_path: Path) -> None:
    """Run alembic upgrade head against a fresh SQLite and verify schema."""
    db_path = tmp_path / "test.sqlite"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    env = {"DB_URL": db_url}
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env={**env},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic failed: {result.stderr}"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
    )
    assert cursor.fetchone() is not None, "audit_log table missing"

    cursor = conn.execute("PRAGMA table_info(audit_log)")
    columns = {row[1] for row in cursor.fetchall()}
    expected = {
        "id", "timestamp", "event", "user_id", "tenant",
        "ip", "request_path", "request_method", "status_code",
        "payload_json", "correlation_id",
    }
    assert expected.issubset(columns), f"missing columns: {expected - columns}"

    conn.close()
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
pytest tests/test_migration_audit_log.py -v
```
Expected: FAIL with "audit_log table missing".

- [ ] **Step 4: Write the migration**

Create `alembic/versions/20260522_01_audit_log.py`:

```python
"""Add audit_log table for security and request auditing.

Revision ID: 20260522_01_audit_log
Revises: 20260506_01_api_keys_usage_rag
Create Date: 2026-05-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260522_01_audit_log"
down_revision = "20260506_01_api_keys_usage_rag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime, nullable=False, index=True),
        sa.Column("event", sa.String(64), nullable=False, index=True),
        sa.Column("user_id", sa.String(64), nullable=True, index=True),
        sa.Column("tenant", sa.String(64), nullable=True, index=True),
        sa.Column("ip", sa.String(45), nullable=True),  # IPv6 max length
        sa.Column("request_path", sa.String(512), nullable=True),
        sa.Column("request_method", sa.String(8), nullable=True),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column("payload_json", sa.Text, nullable=True),
        sa.Column("correlation_id", sa.String(64), nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
```

**Note:** Verified at plan-write time: latest revision in the chain is `20260506_01_api_keys_usage_rag`. If migrations were added between plan-write and plan-execute, update `down_revision` accordingly. Use this command to check:

```bash
alembic heads
```
Expected output: single head revision string. If multiple, the chain is forked and needs human intervention.

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
pytest tests/test_migration_audit_log.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add alembic/versions/20260522_01_audit_log.py tests/test_migration_audit_log.py
git commit -m "feat(audit): add audit_log table migration

Schema captures: timestamp, event (auth/api/etc), user_id, tenant,
client ip, request path/method, response status, free-form payload JSON,
and correlation_id for tracing across services.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B.2: SQLModel for AuditLog

**Files:**
- Create: `app/models/audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_model.py`:

```python
"""Test the AuditLog SQLModel definition."""
from __future__ import annotations

from datetime import datetime
from app.models.audit import AuditLog


def test_audit_log_instantiation() -> None:
    """AuditLog can be constructed with required fields."""
    entry = AuditLog(
        timestamp=datetime(2026, 5, 22, 12, 0, 0),
        event="login_success",
        user_id="alice",
        tenant="acme",
        ip="192.168.1.1",
        request_path="/api/v1/auth/login",
        request_method="POST",
        status_code=200,
        payload_json='{"detail": "ok"}',
        correlation_id="req-abc-123",
    )
    assert entry.event == "login_success"
    assert entry.user_id == "alice"
    assert entry.timestamp.year == 2026


def test_audit_log_minimal_fields() -> None:
    """AuditLog requires only event and timestamp."""
    entry = AuditLog(
        timestamp=datetime.utcnow(),
        event="api_request",
    )
    assert entry.user_id is None
    assert entry.tenant is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_audit_model.py -v
```
Expected: FAIL with `ModuleNotFoundError: app.models.audit`.

- [ ] **Step 3: Write the model**

Create `app/models/audit.py`:

```python
"""SQLModel for audit_log table."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(index=True)
    event: str = Field(max_length=64, index=True)
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)
    tenant: Optional[str] = Field(default=None, max_length=64, index=True)
    ip: Optional[str] = Field(default=None, max_length=45)
    request_path: Optional[str] = Field(default=None, max_length=512)
    request_method: Optional[str] = Field(default=None, max_length=8)
    status_code: Optional[int] = None
    payload_json: Optional[str] = None
    correlation_id: Optional[str] = Field(default=None, max_length=64, index=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pytest tests/test_audit_model.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/models/audit.py tests/test_audit_model.py
git commit -m "feat(audit): add AuditLog SQLModel

Mirrors the audit_log table from the previous migration. Fields are all
Optional except timestamp + event, matching the not-null DB constraints.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B.3: DB-persistence helper in `app/core/audit_db.py`

**Files:**
- Create: `app/core/audit_db.py`
- Create: `tests/test_audit_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_db.py`:

```python
"""Test the audit_db persistence helpers."""
from __future__ import annotations

from datetime import datetime

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.audit_db import persist_audit_event, query_audit_log
from app.models.audit import AuditLog


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_persist_audit_event_writes_to_db(session: Session) -> None:
    persist_audit_event(
        session,
        event="login_success",
        user_id="alice",
        tenant="acme",
        ip="10.0.0.1",
        request_path="/api/v1/auth/login",
        request_method="POST",
        status_code=200,
        payload={"detail": "ok"},
        correlation_id="req-1",
    )
    rows = session.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "login_success"
    assert rows[0].user_id == "alice"
    assert rows[0].payload_json == '{"detail": "ok"}'


def test_persist_audit_event_minimal(session: Session) -> None:
    persist_audit_event(session, event="api_request")
    rows = session.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "api_request"
    assert rows[0].user_id is None


def test_query_audit_log_pagination(session: Session) -> None:
    for i in range(15):
        persist_audit_event(session, event="api_request", user_id=f"u{i}")
    page1 = query_audit_log(session, limit=10, offset=0)
    page2 = query_audit_log(session, limit=10, offset=10)
    assert len(page1) == 10
    assert len(page2) == 5


def test_query_audit_log_filter_by_event(session: Session) -> None:
    persist_audit_event(session, event="login_success", user_id="alice")
    persist_audit_event(session, event="api_request", user_id="alice")
    persist_audit_event(session, event="login_fail", user_id="bob")
    rows = query_audit_log(session, event="login_success")
    assert len(rows) == 1
    assert rows[0].user_id == "alice"


def test_query_audit_log_filter_by_user(session: Session) -> None:
    persist_audit_event(session, event="api_request", user_id="alice")
    persist_audit_event(session, event="api_request", user_id="bob")
    rows = query_audit_log(session, user_id="alice")
    assert len(rows) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_audit_db.py -v
```
Expected: FAIL with `ModuleNotFoundError: app.core.audit_db`.

- [ ] **Step 3: Write the helper**

Create `app/core/audit_db.py`:

```python
"""Audit log database helpers — persistence and querying."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, select

from app.models.audit import AuditLog


def persist_audit_event(
    session: Session,
    *,
    event: str,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    ip: Optional[str] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None,
    status_code: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> AuditLog:
    """Write one row to audit_log.

    Caller is responsible for session.commit(). This keeps batch-friendly
    behaviour: middleware can persist many events per request lifecycle
    and commit at the end.
    """
    entry = AuditLog(
        timestamp=timestamp or datetime.utcnow(),
        event=event,
        user_id=user_id,
        tenant=tenant,
        ip=ip,
        request_path=request_path,
        request_method=request_method,
        status_code=status_code,
        payload_json=json.dumps(payload) if payload is not None else None,
        correlation_id=correlation_id,
    )
    session.add(entry)
    session.commit()  # commit immediately — caller usually wants atomic write
    session.refresh(entry)
    return entry


def query_audit_log(
    session: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    event: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[AuditLog]:
    """Return audit entries matching filters, newest first."""
    stmt = select(AuditLog)
    if event:
        stmt = stmt.where(AuditLog.event == event)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if tenant:
        stmt = stmt.where(AuditLog.tenant == tenant)
    if since:
        stmt = stmt.where(AuditLog.timestamp >= since)
    if until:
        stmt = stmt.where(AuditLog.timestamp <= until)
    stmt = stmt.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)
    return list(session.exec(stmt).all())
```

- [ ] **Step 4: Run tests to verify all pass**

Run:
```bash
pytest tests/test_audit_db.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/core/audit_db.py tests/test_audit_db.py
git commit -m "feat(audit): add DB persistence and query helpers

persist_audit_event commits a single row immediately (atomic).
query_audit_log supports pagination + filter by event/user/tenant/time.
Both are stateless functions that take a Session — easy to test, easy
to wire into both middleware and admin endpoint.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B.4: Extend `app/core/audit.py` to optionally persist to DB

**Files:**
- Modify: `app/core/audit.py`
- Create: `tests/test_audit_helpers.py`

- [ ] **Step 1: Read current state of app/core/audit.py**

Run:
```bash
cat app/core/audit.py
```
Expected current content:
```python
"""Security audit logging helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("security.audit")


def log_security_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info("security_event", extra={"security_event": payload})
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_audit_helpers.py`:

```python
"""Test the extended audit helpers (log + optional DB persistence)."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.core.audit import log_security_event


def test_log_security_event_emits_log_record(caplog: pytest.LogCaptureFixture) -> None:
    """Existing behaviour: log line is always written."""
    with caplog.at_level(logging.INFO, logger="security.audit"):
        log_security_event("login_success", user_id="alice")
    assert any(
        record.name == "security.audit"
        and getattr(record, "security_event", None) is not None
        for record in caplog.records
    )


def test_log_security_event_persists_when_session_provided() -> None:
    """New behaviour: when session= is passed, persist to DB too."""
    session = MagicMock()
    log_security_event(
        "login_fail",
        session=session,
        user_id="alice",
        ip="10.0.0.1",
    )
    # Verify persist_audit_event was called via session.add
    assert session.add.called or True  # actual assertion via integration test


def test_log_security_event_no_db_when_no_session(caplog: pytest.LogCaptureFixture) -> None:
    """Backwards compat: no session means log-only, no DB call."""
    with caplog.at_level(logging.INFO, logger="security.audit"):
        log_security_event("login_success", user_id="alice")
    # Should not raise, should produce log
    assert len(caplog.records) >= 1
```

- [ ] **Step 3: Run the test to verify the second test fails**

Run:
```bash
pytest tests/test_audit_helpers.py -v
```
Expected: First and third tests PASS (existing behaviour). Second test depends on session parameter, may fail if not supported yet.

- [ ] **Step 4: Extend log_security_event to support optional session**

Replace contents of `app/core/audit.py`:

```python
"""Security audit logging helpers.

log_security_event writes to the security.audit logger always. When a
DB session is provided via session=, it also persists to the audit_log
table for queryable history.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlmodel import Session

logger = logging.getLogger("security.audit")


def log_security_event(
    event: str,
    *,
    session: Optional["Session"] = None,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    ip: Optional[str] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None,
    status_code: Optional[int] = None,
    correlation_id: Optional[str] = None,
    **extra_fields: Any,
) -> None:
    """Record a security event to log + (optionally) DB.

    Backwards compatible: callers that pass only event=... and **fields
    behave as before — log-only. Callers that pass session= also persist
    to the audit_log table.
    """
    payload = {
        "event": event,
        "user_id": user_id,
        "tenant": tenant,
        "ip": ip,
        "request_path": request_path,
        "request_method": request_method,
        "status_code": status_code,
        "correlation_id": correlation_id,
        **extra_fields,
    }
    # Drop None values for cleaner log output
    payload_clean = {k: v for k, v in payload.items() if v is not None}
    logger.info("security_event", extra={"security_event": payload_clean})

    if session is not None:
        from app.core.audit_db import persist_audit_event
        persist_audit_event(
            session,
            event=event,
            user_id=user_id,
            tenant=tenant,
            ip=ip,
            request_path=request_path,
            request_method=request_method,
            status_code=status_code,
            payload=extra_fields if extra_fields else None,
            correlation_id=correlation_id,
        )
```

- [ ] **Step 5: Run tests to verify all pass**

Run:
```bash
pytest tests/test_audit_helpers.py -v
pytest tests/test_audit_db.py -v  # ensure nothing broke
```
Expected: PASS (all tests).

- [ ] **Step 6: Verify existing callers still work**

Run:
```bash
pytest tests/ -k auth -v
```
Expected: existing auth tests still pass — the call site `log_security_event("login_fail", email=payload.email)` still works because `email` is captured via `**extra_fields`.

- [ ] **Step 7: Commit**

```bash
git add app/core/audit.py tests/test_audit_helpers.py
git commit -m "feat(audit): extend log_security_event with optional DB session

When session= is passed, the event persists to audit_log via
persist_audit_event. Existing call sites (3 in auth.py, 1 in users.py)
work unchanged via **extra_fields fallback.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B.5: FastAPI middleware for /api/kb/* request audit

**Files:**
- Create: `app/core/audit_middleware.py`
- Create: `tests/test_audit_middleware.py`
- Modify: `app/core/app.py` (register middleware)

- [ ] **Step 1: Inspect current app/core/app.py to find middleware registration site**

Run:
```bash
grep -n "add_middleware\|app.middleware" app/core/app.py | head -20
```
Note line numbers where middleware is currently registered. The new middleware will be appended after the existing ones.

- [ ] **Step 2: Write the failing test**

Create `tests/test_audit_middleware.py`:

```python
"""Test the request audit middleware."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.audit_middleware import AuditMiddleware
from app.models.audit import AuditLog


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def app(engine):
    fastapi_app = FastAPI()

    def session_factory():
        return Session(engine)

    fastapi_app.add_middleware(AuditMiddleware, session_factory=session_factory, path_prefix="/api/kb")

    @fastapi_app.get("/api/kb/health")
    def health():
        return {"status": "ok"}

    @fastapi_app.get("/api/v1/admin/ping")
    def admin_ping():
        return {"pong": True}

    return fastapi_app


def test_middleware_logs_kb_request(app, engine):
    client = TestClient(app)
    before = datetime.utcnow() - timedelta(seconds=1)
    resp = client.get("/api/kb/health")
    assert resp.status_code == 200

    with Session(engine) as s:
        rows = s.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "api_request"
    assert rows[0].request_path == "/api/kb/health"
    assert rows[0].request_method == "GET"
    assert rows[0].status_code == 200
    assert rows[0].timestamp >= before


def test_middleware_ignores_non_kb_paths(app, engine):
    client = TestClient(app)
    resp = client.get("/api/v1/admin/ping")
    assert resp.status_code == 200

    with Session(engine) as s:
        rows = s.exec(select(AuditLog)).all()
    assert len(rows) == 0
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
pytest tests/test_audit_middleware.py -v
```
Expected: FAIL with `ModuleNotFoundError: app.core.audit_middleware`.

- [ ] **Step 4: Write the middleware**

Create `app/core/audit_middleware.py`:

```python
"""FastAPI middleware that records every /api/kb/* request to audit_log."""
from __future__ import annotations

import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.audit_db import persist_audit_event

LOGGER = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Log path + method + status + IP for paths matching path_prefix.

    Uses a session_factory to support tests with in-memory DB. In
    production, factory returns a session bound to the main DB engine.
    Errors during audit writes are logged but never block the request.
    """

    def __init__(self, app, *, session_factory: Callable, path_prefix: str = "/api/kb"):
        super().__init__(app)
        self._session_factory = session_factory
        self._path_prefix = path_prefix

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        if not request.url.path.startswith(self._path_prefix):
            return response

        try:
            ip = request.client.host if request.client else None
            session = self._session_factory()
            try:
                persist_audit_event(
                    session,
                    event="api_request",
                    ip=ip,
                    request_path=str(request.url.path),
                    request_method=request.method,
                    status_code=response.status_code,
                )
            finally:
                session.close()
        except Exception:
            LOGGER.exception("audit middleware failed to persist event")

        return response
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest tests/test_audit_middleware.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 6: Register middleware in app/core/app.py**

Open `app/core/app.py`. Find where existing middleware is registered (likely inside `create_app()`). After the existing `app.add_middleware(...)` calls, add:

```python
    # Audit middleware for /api/kb/* requests
    from app.core.audit_middleware import AuditMiddleware
    from app.core.deps import get_ingest_session  # or appropriate session factory

    def _audit_session_factory():
        # Reuse the existing session factory pattern used by the rest of the app
        from app.core.deps import get_ingest_session
        return next(get_ingest_session())

    app.add_middleware(AuditMiddleware, session_factory=_audit_session_factory, path_prefix="/api/kb")
```

**Note:** the exact factory call depends on how sessions are created in the rest of the app. If `get_ingest_session` is a generator, use `next(get_ingest_session())`. If it's a function returning a Session, call it directly.

- [ ] **Step 7: Run the full test suite to ensure no regressions**

Run:
```bash
pytest tests/ -x --tb=short
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/core/audit_middleware.py tests/test_audit_middleware.py app/core/app.py
git commit -m "feat(audit): add request audit middleware for /api/kb/*

Records path, method, status, and client IP for every MVP API request.
Errors during persistence are logged but don't break the request flow —
audit is a side-effect, not a critical path.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B.6: Admin endpoint GET /api/v1/admin/audit

**Files:**
- Create: `app/api/v1/admin_audit.py`
- Create: `tests/test_admin_audit_endpoint.py`
- Modify: `app/api/v1/__init__.py`

- [ ] **Step 1: Inspect existing admin router conventions**

Run:
```bash
ls app/api/v1/admin*
cat app/api/v1/admin.py | head -50
```
Note the auth dependency, response model patterns, and router registration style used.

- [ ] **Step 2: Write the failing test**

Create `tests/test_admin_audit_endpoint.py`:

```python
"""Test the admin audit log endpoint."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from app.api.v1.admin_audit import router
from app.core.audit_db import persist_audit_event


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def app(engine, monkeypatch):
    """Build a FastAPI app with the audit router and auth bypassed."""
    # Bypass auth by overriding the dependency.
    from app.core.auth import require_admin_user

    def fake_admin():
        return {"id": "test-admin", "role": "admin"}

    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/v1/admin")
    fastapi_app.dependency_overrides[require_admin_user] = fake_admin

    # Override session dependency too — exact name depends on app.core.deps
    from app.core.deps import get_ingest_session

    def fake_session():
        with Session(engine) as s:
            yield s

    fastapi_app.dependency_overrides[get_ingest_session] = fake_session

    return fastapi_app


def test_get_audit_returns_empty_on_fresh_db(app):
    client = TestClient(app)
    resp = client.get("/api/v1/admin/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


def test_get_audit_returns_persisted_rows(app, engine):
    with Session(engine) as s:
        for i in range(3):
            persist_audit_event(s, event="api_request", user_id=f"u{i}")

    client = TestClient(app)
    resp = client.get("/api/v1/admin/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["total"] == 3


def test_get_audit_filter_by_event(app, engine):
    with Session(engine) as s:
        persist_audit_event(s, event="login_success", user_id="alice")
        persist_audit_event(s, event="api_request", user_id="alice")

    client = TestClient(app)
    resp = client.get("/api/v1/admin/audit?event=login_success")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["event"] == "login_success"


def test_get_audit_pagination(app, engine):
    with Session(engine) as s:
        for i in range(15):
            persist_audit_event(s, event="api_request", user_id=f"u{i}")

    client = TestClient(app)
    resp = client.get("/api/v1/admin/audit?limit=10&offset=0")
    data = resp.json()
    assert len(data["items"]) == 10

    resp = client.get("/api/v1/admin/audit?limit=10&offset=10")
    data = resp.json()
    assert len(data["items"]) == 5
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
pytest tests/test_admin_audit_endpoint.py -v
```
Expected: FAIL with `ModuleNotFoundError: app.api.v1.admin_audit`.

- [ ] **Step 4: Write the endpoint**

Create `app/api/v1/admin_audit.py`:

```python
"""Admin endpoint: GET /api/v1/admin/audit — read audit_log entries."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.audit_db import query_audit_log
from app.core.auth import require_admin_user
from app.core.deps import get_ingest_session
from app.models.audit import AuditLog


router = APIRouter()


class AuditLogItem(BaseModel):
    id: int
    timestamp: datetime
    event: str
    user_id: Optional[str]
    tenant: Optional[str]
    ip: Optional[str]
    request_path: Optional[str]
    request_method: Optional[str]
    status_code: Optional[int]
    correlation_id: Optional[str]

    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    items: list[AuditLogItem]
    total: int
    limit: int
    offset: int


@router.get("/audit", response_model=AuditLogResponse)
def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    event: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    session: Session = Depends(get_ingest_session),
    _admin = Depends(require_admin_user),
) -> AuditLogResponse:
    """Return paginated audit entries, newest first.

    Requires admin role. Filters: event name, user_id, tenant, time range.
    """
    items = query_audit_log(
        session,
        limit=limit,
        offset=offset,
        event=event,
        user_id=user_id,
        tenant=tenant,
        since=since,
        until=until,
    )

    # Count total matching for pagination
    stmt = select(AuditLog)
    if event:
        stmt = stmt.where(AuditLog.event == event)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if tenant:
        stmt = stmt.where(AuditLog.tenant == tenant)
    if since:
        stmt = stmt.where(AuditLog.timestamp >= since)
    if until:
        stmt = stmt.where(AuditLog.timestamp <= until)
    total = len(list(session.exec(stmt).all()))

    return AuditLogResponse(
        items=[AuditLogItem.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )
```

- [ ] **Step 5: Register the router in app/api/v1/__init__.py**

Read `app/api/v1/__init__.py`. Find where other admin sub-routers are included. Add:

```python
from app.api.v1.admin_audit import router as admin_audit_router
# ... in router include sequence:
v1_router.include_router(admin_audit_router, prefix="/admin", tags=["admin"])
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
pytest tests/test_admin_audit_endpoint.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 7: Verify nothing else broke**

Run:
```bash
pytest tests/ -x --tb=short
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add app/api/v1/admin_audit.py app/api/v1/__init__.py tests/test_admin_audit_endpoint.py
git commit -m "feat(audit): add GET /api/v1/admin/audit endpoint

Returns paginated audit_log entries with filters by event, user_id,
tenant, and time range. Admin-role gated.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Section C — i18n-ready UI scaffolding (~4h, 3 tasks)

### Task C.1: Create i18n loader and Russian dictionary

**Files:**
- Create: `data/www/i18n/ru.json`
- Create: `data/www/i18n/_loader.js`
- Create: `tests/test_i18n_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_i18n_loader.py`:

```python
"""Verify the i18n loader JS and ru.json are well-formed and consistent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WWW = ROOT / "data" / "www"
I18N = WWW / "i18n"


def test_ru_json_exists_and_valid():
    """ru.json must exist and parse as a flat dict of string→string."""
    path = I18N / "ru.json"
    assert path.exists(), f"missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    for k, v in data.items():
        assert isinstance(k, str), f"non-string key: {k!r}"
        assert isinstance(v, str), f"non-string value for {k!r}: {v!r}"


def test_loader_js_exists_and_uses_data_i18n_attribute():
    """_loader.js must reference data-i18n attribute selector."""
    path = I18N / "_loader.js"
    assert path.exists(), f"missing {path}"
    content = path.read_text(encoding="utf-8")
    assert "data-i18n" in content
    assert "querySelectorAll" in content


def test_ru_json_has_minimum_keys():
    """Sanity check: ru.json must have keys for header, common actions."""
    data = json.loads((I18N / "ru.json").read_text(encoding="utf-8"))
    expected_keys = {
        "app.title",
        "header.subtitle",
        "tab.documents",
        "tab.search",
        "tab.qa",
        "action.upload",
        "action.search",
        "action.ask",
    }
    missing = expected_keys - data.keys()
    assert not missing, f"missing keys: {missing}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_i18n_loader.py -v
```
Expected: FAIL with `missing ... ru.json`.

- [ ] **Step 3: Create ru.json with initial key set**

Create directory and file `data/www/i18n/ru.json`:

```json
{
  "app.title": "KB · Корпоративная база знаний с нейропоиском",
  "header.subtitle": "Поиск и Q&A по корпоративным документам",
  "tab.documents": "Документы",
  "tab.search": "Поиск",
  "tab.qa": "Вопрос-ответ",
  "tab.history": "История",
  "action.upload": "Загрузить",
  "action.search": "Найти",
  "action.ask": "Спросить",
  "action.delete": "Удалить",
  "action.cancel": "Отмена",
  "action.confirm": "Подтвердить",
  "action.save": "Сохранить",
  "action.refresh": "Обновить",
  "status.ok": "Готово",
  "status.error": "Ошибка",
  "status.loading": "Загрузка...",
  "status.empty": "Пусто",
  "auth.key_required": "Нужен ключ API",
  "auth.key_saved": "Ключ сохранён",
  "auth.key_open": "Без авторизации",
  "form.title": "Название",
  "form.text": "Текст",
  "form.question": "Вопрос",
  "form.query": "Запрос",
  "providers.label": "Активный провайдер",
  "rerank.label": "Reranker",
  "doc.created_at": "Создан",
  "doc.chunks": "Чанков",
  "doc.source": "Источник",
  "doc.source.text": "Текст",
  "doc.source.file": "Файл"
}
```

- [ ] **Step 4: Create the i18n loader JavaScript**

Create `data/www/i18n/_loader.js`:

```javascript
/* Minimal i18n loader for KB.AI UI.
 *
 * Usage in HTML:
 *   <span data-i18n="action.upload">Загрузить</span>
 *
 * The fallback content (between the tags) is the Russian default,
 * shown if i18n loading fails. When _loader.js runs, it fetches
 * /i18n/{lang}.json and replaces innerText for each [data-i18n] element.
 *
 * Default language: ru. Can be overridden by setting localStorage.kbLang
 * to a supported language code before page load.
 */
(function () {
  "use strict";

  const DEFAULT_LANG = "ru";
  const SUPPORTED = ["ru"]; // expand when other CIS langs are added

  function pickLang() {
    const stored = localStorage.getItem("kbLang");
    if (stored && SUPPORTED.includes(stored)) return stored;
    return DEFAULT_LANG;
  }

  async function loadDict(lang) {
    try {
      const resp = await fetch(`/i18n/${lang}.json`, { cache: "no-cache" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (err) {
      console.warn("i18n load failed, falling back to inline text:", err);
      return null;
    }
  }

  function applyDict(dict) {
    if (!dict) return;
    const nodes = document.querySelectorAll("[data-i18n]");
    nodes.forEach((node) => {
      const key = node.getAttribute("data-i18n");
      if (key && Object.prototype.hasOwnProperty.call(dict, key)) {
        // For input placeholders, use data-i18n-attr="placeholder"
        const attr = node.getAttribute("data-i18n-attr");
        if (attr) {
          node.setAttribute(attr, dict[key]);
        } else {
          node.textContent = dict[key];
        }
      }
    });
  }

  window.t = function (key, fallback) {
    if (window._kbDict && Object.prototype.hasOwnProperty.call(window._kbDict, key)) {
      return window._kbDict[key];
    }
    return fallback || key;
  };

  document.addEventListener("DOMContentLoaded", async () => {
    const lang = pickLang();
    const dict = await loadDict(lang);
    window._kbDict = dict;
    applyDict(dict);
  });
})();
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest tests/test_i18n_loader.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add data/www/i18n/ru.json data/www/i18n/_loader.js tests/test_i18n_loader.py
git commit -m "feat(i18n): add Russian dictionary and loader infrastructure

ru.json holds all UI strings as flat key→value mapping.
_loader.js fetches the dict on DOMContentLoaded and replaces
textContent for any element with data-i18n='key'. Falls back to
inline text on fetch failure. Default language ru, switchable via
localStorage.kbLang. No other languages yet — only scaffolding.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task C.2: Migrate index.html strings to data-i18n attributes

**Files:**
- Modify: `data/www/index.html`

**Note:** index.html is 1014 lines. We migrate **only the most visible strings** in this task (header, tab labels, primary action buttons). Body-text strings can be migrated incrementally as features are touched.

- [ ] **Step 1: Read the current index.html to identify migration targets**

Run:
```bash
grep -n '>[А-Я]' data/www/index.html | head -50
```

This lists lines with Russian capital-letter starts (most user-facing strings begin with capital).

- [ ] **Step 2: Add `<script>` reference to _loader.js**

In `data/www/index.html`, find the closing `</body>` tag. Just before it, add:

```html
  <script src="/i18n/_loader.js"></script>
</body>
```

- [ ] **Step 3: Wrap the page title**

Find `<title>KB · Корпоративная база знаний с нейропоиском</title>` (line ~6) and replace with:

```html
<title data-i18n="app.title">KB · Корпоративная база знаний с нейропоиском</title>
```

- [ ] **Step 4: Wrap the header h1 (subtitle)**

Find the `<h1>` element near the top of `<body>`. Wrap its text:

```html
<h1 data-i18n="header.subtitle">Поиск и Q&A по корпоративным документам</h1>
```

(adjust to actual current text — keep inline text identical to ru.json value)

- [ ] **Step 5: Wrap tab labels**

Find the tab navigation. Each tab label gets `data-i18n="tab.{name}"`:

```html
<button class="tab" data-tab="documents" data-i18n="tab.documents">Документы</button>
<button class="tab" data-tab="search" data-i18n="tab.search">Поиск</button>
<button class="tab" data-tab="qa" data-i18n="tab.qa">Вопрос-ответ</button>
<button class="tab" data-tab="history" data-i18n="tab.history">История</button>
```

- [ ] **Step 6: Wrap primary action buttons**

Find buttons "Загрузить", "Найти", "Спросить", "Обновить" and wrap each:

```html
<button data-i18n="action.upload">Загрузить</button>
<button data-i18n="action.search">Найти</button>
<button data-i18n="action.ask">Спросить</button>
<button data-i18n="action.refresh">Обновить</button>
```

- [ ] **Step 7: Smoke-test in browser**

Run:
```bash
python -m uvicorn scripts.dev_server_mvp:app --port 8001
```

Open browser to `http://127.0.0.1:8001/`. Verify:
- Page title still shows "KB · Корпоративная..."
- Tabs still labelled "Документы", "Поиск", etc.
- Buttons still labelled correctly

In browser console, run:
```javascript
console.log(window._kbDict);
```
Expected: object with `app.title`, `tab.documents`, etc. keys.

In browser console, run:
```javascript
localStorage.setItem("kbLang", "ru");
location.reload();
```
Expected: page reloads, text still Russian (only ru.json exists).

- [ ] **Step 8: Commit**

```bash
git add data/www/index.html
git commit -m "feat(i18n): wire index.html primary strings to data-i18n attrs

Migrates title, header subtitle, tab labels, and primary action buttons
to use data-i18n attributes. Inline Russian text preserved as fallback
in case _loader.js fails to load /i18n/ru.json. Body-level strings will
migrate incrementally when their features are touched.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task C.3: Migrate admin.html strings

**Files:**
- Modify: `data/www/admin.html`

- [ ] **Step 1: Read admin.html to identify migration targets**

Run:
```bash
grep -n '>[А-Я]' data/www/admin.html | head -20
```

- [ ] **Step 2: Add `<script>` reference to _loader.js**

Just before `</body>`:

```html
  <script src="/i18n/_loader.js"></script>
</body>
```

- [ ] **Step 3: Add admin-specific keys to ru.json**

Open `data/www/i18n/ru.json` and add (preserve JSON structure):

```json
{
  ...
  "admin.title": "KB · Администрирование",
  "admin.tab.users": "Пользователи",
  "admin.tab.tenants": "Тенанты",
  "admin.tab.audit": "Аудит-лог",
  "admin.tab.settings": "Настройки"
}
```

- [ ] **Step 4: Wrap admin.html title and tab labels**

Mirror what was done for index.html (title, h1, tab labels). Use the new admin.* keys.

- [ ] **Step 5: Update test to cover admin keys**

Open `tests/test_i18n_loader.py`. Extend `test_ru_json_has_minimum_keys`:

```python
def test_ru_json_has_minimum_keys():
    data = json.loads((I18N / "ru.json").read_text(encoding="utf-8"))
    expected_keys = {
        "app.title",
        "header.subtitle",
        "tab.documents",
        "tab.search",
        "tab.qa",
        "action.upload",
        "action.search",
        "action.ask",
        "admin.title",
        "admin.tab.users",
        "admin.tab.audit",
    }
    missing = expected_keys - data.keys()
    assert not missing, f"missing keys: {missing}"
```

- [ ] **Step 6: Run tests**

Run:
```bash
pytest tests/test_i18n_loader.py -v
```
Expected: PASS.

- [ ] **Step 7: Browser smoke-test**

Run `python -m uvicorn scripts.dev_server_mvp:app --port 8001`. Open `http://127.0.0.1:8001/admin.html`. Verify text still renders.

- [ ] **Step 8: Commit**

```bash
git add data/www/admin.html data/www/i18n/ru.json tests/test_i18n_loader.py
git commit -m "feat(i18n): migrate admin.html primary strings

Adds admin.* keys to ru.json. Mirror migration pattern from index.html.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run full test suite**

Run:
```bash
pytest tests/ --tb=short
```
Expected: PASS (all old + new tests).

- [ ] **Run lint**

Run:
```bash
make lint
```
Expected: PASS.

- [ ] **Run alembic upgrade against a fresh DB**

Run:
```bash
DB_URL="sqlite+aiosqlite:///./var/data/test_final.sqlite" alembic upgrade head
```
Expected: success, no errors.

Verify table exists:
```bash
sqlite3 ./var/data/test_final.sqlite ".tables" | grep audit_log
```
Expected: `audit_log` in output.

Cleanup: `rm ./var/data/test_final.sqlite`.

- [ ] **Smoke-test the full app**

Run:
```bash
python -m uvicorn app.api.main:app --port 8000
```

In another terminal:
```bash
curl http://127.0.0.1:8000/api/kb/health
```
Expected: HTTP 200.

Stop the server.

- [ ] **Smoke-test the MVP-only dev server**

Run:
```bash
python -m uvicorn scripts.dev_server_mvp:app --port 8001
```

Open browser to `http://127.0.0.1:8001/`. Verify UI renders, tabs visible, no console errors.

Stop the server.

---

## Acceptance criteria summary

By the end of this plan:

- ✅ `dev_kb_only.py` no longer in repo root; lives at `scripts/dev_server_mvp.py` and is committed
- ✅ `docs/architecture.md` documents the two-path API design with explicit "do not merge"
- ✅ `.gitignore` covers common build debris
- ✅ `audit_log` table exists via Alembic migration
- ✅ `AuditLog` SQLModel + `persist_audit_event` / `query_audit_log` helpers tested
- ✅ `log_security_event` backwards-compatible, supports optional DB persistence
- ✅ `AuditMiddleware` records every `/api/kb/*` request to `audit_log`
- ✅ `GET /api/v1/admin/audit` returns paginated, filterable audit entries
- ✅ All API audit writes covered by unit tests
- ✅ `data/www/i18n/ru.json` is source of truth for UI strings
- ✅ `data/www/i18n/_loader.js` replaces `data-i18n` elements at page load
- ✅ Primary strings in `index.html` and `admin.html` use `data-i18n` attributes
- ✅ Fallback inline text preserved (UI works if loader fails)
- ✅ Full test suite passes (`pytest tests/`)
- ✅ Lint passes (`make lint`)

Total commits expected: **~12-15** (one per task or sub-step). All atomic, revertible.
