# PDF Citation Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Превратить плоские text snippets в `/api/kb/ask` ответах в кликабельные цитаты `[файл.pdf, стр. 12]`, открывающие modal PDF.js viewer с подсветкой текста чанка через find API.

**Architecture:**
- **Backend:** новая колонка `kb_chunks.page_number`, новые колонки `kb_documents.has_original_file` + `file_relpath`, per-page chunking при ingestion, blob storage в `var/data/kb_files/<doc_id>.pdf`, новый эндпоинт `GET /api/kb/documents/{id}/file` с path-traversal guard.
- **Frontend:** vendored PDF.js (lazy import), извлечение auth-helpers в `kb-auth.js`, native `<dialog>` modal, `pdf-viewer.js` контроллер, find API для подсветки.
- **Retrofit:** clean break — старые документы остаются без viewer (graceful fallback на text snippet).

**Tech Stack:**
- Backend: FastAPI, sqlite3 (raw), Alembic, pytest, httpx TestClient
- Frontend: vanilla JS (ESM), PDF.js v4.x legacy build, native `<dialog>` element
- Testing: pytest + manual browser smoke

**Spec:** [`docs/superpowers/specs/2026-05-22-pdf-citation-viewer-design.md`](../specs/2026-05-22-pdf-citation-viewer-design.md)

---

## File Structure

**Files created:**
- `alembic/versions/20260522_02_pdf_citation.py` — migration adding `page_number`, `has_original_file`, `file_relpath`
- `data/www/vendor/pdfjs/build/pdf.mjs` — vendored PDF.js core (downloaded from npmjs/CDN)
- `data/www/vendor/pdfjs/build/pdf.worker.mjs` — vendored PDF.js worker
- `data/www/vendor/pdfjs/LICENSE` — Apache-2.0 license
- `data/www/js/kb-auth.js` — extracted auth-helpers (`window.kbAuth.{getApiKey, withAuthHeaders, fetch}`)
- `data/www/js/pdf-viewer.js` — `window.kbPdfViewer.openCitation()` controller + render + find dispatch
- `tests/test_kb_store_pages.py` — page-per-chunk store tests
- `tests/test_kb_mvp_upload_blob.py` — PDF blob persistence tests
- `tests/test_kb_mvp_file_endpoint.py` — `GET /documents/{id}/file` tests
- `tests/test_kb_mvp_search_response.py` — HitOut.page + has_original propagation
- `tests/test_migration_pdf_citation.py` — alembic upgrade test
- `tests/test_pdf_viewer_js.py` — JS smoke (Python-side regex/file checks)
- `tests/test_citation_rendering_html.py` — index.html structure verification

**Files modified:**
- `app/services/kb_store.py` — Document/SearchHit dataclasses, `add_document` signature, `search` JOIN, schema init, `update_file_metadata`
- `app/api/kb_mvp.py` — `_parse_file_bytes_with_pages`, blob save in `upload_document`, new file endpoint, `HitOut` model, `_hit_to_out`
- `data/www/index.html` — sources rendering to citation buttons, dialog markup + CSS, scripts load kb-auth.js + pdf-viewer.js
- `data/www/i18n/ru.json` — new i18n keys for citations/viewer
- `data/www/i18n/_loader.js` — `{var}` interpolation in textContent
- `tests/test_kb_mvp.py` — keep existing tests green after API changes (no new tests added here, but verify)

**Test conventions used in this repo:**
- `pytest tests/<file>.py -v` — basic invocation
- `tests/conftest.py` — existing fixtures
- Windows PowerShell shell: use `py` launcher or direct `python` if available
- No venv — deps already on user's site-packages (see `MEMORY.md`)

---

## Section A — Schema migration + SearchHit propagation (~4h, 3 tasks)

### Task A.1: Alembic migration `20260522_02_pdf_citation`

**Files:**
- Create: `alembic/versions/20260522_02_pdf_citation.py`
- Create: `tests/test_migration_pdf_citation.py`

**Background:** Adds `kb_chunks.page_number INT NULL`, `kb_documents.has_original_file BOOL NOT NULL DEFAULT 0`, `kb_documents.file_relpath TEXT NULL`, plus composite index `idx_kb_chunks_doc_page`. The previous migration is `20260522_01_audit_log` — verified at plan-write time via `ls alembic/versions/`.

- [ ] **Step 1: Verify the current head of the alembic chain**

Run:
```powershell
py -m alembic heads
```
Expected output: single revision `20260522_01_audit_log`. If multiple heads or different revision, **stop and ask**; chain mismatch needs human intervention.

- [ ] **Step 2: Write the failing migration test**

Create `tests/test_migration_pdf_citation.py`:

```python
"""Verify the PDF-citation migration adds expected columns and index."""
from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path


def test_pdf_citation_migration_adds_columns(tmp_path: Path) -> None:
    """Run `alembic upgrade head` against a fresh SQLite, check schema."""
    db_path = tmp_path / "test.sqlite"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    env = {**os.environ, "DB_URL": db_url}
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic failed: {result.stderr}"

    conn = sqlite3.connect(str(db_path))

    # kb_chunks.page_number
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
    assert "page_number" in cols, f"page_number not in {cols}"

    # kb_documents new columns
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_documents)")}
    assert "has_original_file" in cols, f"has_original_file not in {cols}"
    assert "file_relpath" in cols, f"file_relpath not in {cols}"

    # Composite index
    idx_names = {
        row[1] for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type='index' AND tbl_name='kb_chunks'"
        )
    }
    assert "idx_kb_chunks_doc_page" in idx_names, f"index missing in {idx_names}"

    conn.close()


def test_pdf_citation_migration_default_values(tmp_path: Path) -> None:
    """After upgrade, existing kb_documents rows should default has_original_file=0."""
    db_path = tmp_path / "test.sqlite"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    env = {**os.environ, "DB_URL": db_url}

    # Apply only the previous migration first
    subprocess.run(
        ["alembic", "upgrade", "20260522_01_audit_log"],
        env=env, capture_output=True, text=True, check=True,
    )

    # Insert a document the old way
    conn = sqlite3.connect(str(db_path))
    # kb_documents may not exist yet if it's created on store init — skip if so.
    has_kb = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kb_documents'"
    ).fetchone()
    if has_kb is None:
        conn.close()
        return  # store-managed table — covered by other tests
    conn.execute(
        "INSERT INTO kb_documents(title, text, created_at) VALUES (?, ?, ?)",
        ("Old doc", "body", "2026-05-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    # Apply our migration
    subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env, capture_output=True, text=True, check=True,
    )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT has_original_file, file_relpath FROM kb_documents WHERE title='Old doc'"
    ).fetchone()
    conn.close()
    if row is not None:
        assert row[0] == 0, f"expected has_original_file=0 for legacy row, got {row[0]}"
        assert row[1] is None, f"expected file_relpath=NULL, got {row[1]}"
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_migration_pdf_citation.py -v
```
Expected: FAIL with column missing.

- [ ] **Step 4: Write the migration**

Create `alembic/versions/20260522_02_pdf_citation.py`:

```python
"""Add PDF citation columns: page_number, has_original_file, file_relpath.

Revision ID: 20260522_02_pdf_citation
Revises: 20260522_01_audit_log
Create Date: 2026-05-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260522_02_pdf_citation"
down_revision = "20260522_01_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # kb_chunks and kb_documents are created lazily by KnowledgeBaseStore on
    # first use (see app/services/kb_store.py:_init_schema). When alembic
    # runs against a fresh DB they may not exist yet. We use the
    # SQLite-friendly idempotent guard via op.execute with IF NOT EXISTS for
    # the index, and rely on store._init_schema for the base tables. To
    # cover both flows, we create tables here only if they don't already
    # exist, then add columns conditionally.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "kb_documents" not in existing_tables:
        op.create_table(
            "kb_documents",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("title", sa.Text, nullable=False),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("created_at", sa.Text, nullable=False),
            sa.Column("source", sa.Text, nullable=False, server_default="text"),
            sa.Column("filename", sa.Text, nullable=True),
            sa.Column("mime_type", sa.Text, nullable=True),
        )

    if "kb_chunks" not in existing_tables:
        op.create_table(
            "kb_chunks",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("document_id", sa.Integer, sa.ForeignKey("kb_documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("chunk_index", sa.Integer, nullable=False),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("embedding", sa.LargeBinary, nullable=False),
            sa.Column("embedder", sa.Text, nullable=False, server_default="hash"),
            sa.Column("dim", sa.Integer, nullable=False, server_default="256"),
        )

    # Refresh inspector after potential table creations
    inspector = sa.inspect(bind)

    chunk_cols = {col["name"] for col in inspector.get_columns("kb_chunks")}
    if "page_number" not in chunk_cols:
        op.add_column("kb_chunks", sa.Column("page_number", sa.Integer, nullable=True))

    doc_cols = {col["name"] for col in inspector.get_columns("kb_documents")}
    if "has_original_file" not in doc_cols:
        op.add_column(
            "kb_documents",
            sa.Column("has_original_file", sa.Boolean, nullable=False, server_default=sa.text("0")),
        )
    if "file_relpath" not in doc_cols:
        op.add_column("kb_documents", sa.Column("file_relpath", sa.String(512), nullable=True))

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("kb_chunks")}
    if "idx_kb_chunks_doc_page" not in existing_indexes:
        op.create_index("idx_kb_chunks_doc_page", "kb_chunks", ["document_id", "page_number"])


def downgrade() -> None:
    op.drop_index("idx_kb_chunks_doc_page", table_name="kb_chunks")
    op.drop_column("kb_documents", "file_relpath")
    op.drop_column("kb_documents", "has_original_file")
    op.drop_column("kb_chunks", "page_number")
```

**Why the `if not exists` guards:** the `kb_*` tables are created at runtime by `KnowledgeBaseStore._init_schema()` (see `app/services/kb_store.py:232`), not by alembic. A fresh DB never sees them via migrations until first store use. The guards make the migration safe in both orders.

- [ ] **Step 5: Run the test to verify it passes**

Run:
```powershell
py -m pytest tests/test_migration_pdf_citation.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 6: Verify existing migrations still apply against a clean DB**

Run:
```powershell
$env:DB_URL = "sqlite+aiosqlite:///./var/data/test_chain.sqlite"
py -m alembic upgrade head
```
Expected: success. Then check the chain:
```powershell
py -m alembic current
```
Expected: `20260522_02_pdf_citation (head)`.

Cleanup:
```powershell
Remove-Item .\var\data\test_chain.sqlite
$env:DB_URL = $null
```

- [ ] **Step 7: Commit**

```powershell
git add alembic/versions/20260522_02_pdf_citation.py tests/test_migration_pdf_citation.py
git commit -m @'
feat(kb-mvp): add migration for PDF citation columns

Adds page_number to kb_chunks; has_original_file + file_relpath to
kb_documents. Composite index idx_kb_chunks_doc_page covers the
(document_id, page_number) lookup pattern used by future citation
queries. server_default ensures existing rows get safe defaults.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task A.2: Extend `Document` and `SearchHit` dataclasses; update `add_document` to accept `pages`

**Files:**
- Modify: `app/services/kb_store.py` (Document, SearchHit, _init_schema, _row_to_document, add_document, search, get_document, list_documents)
- Create: `tests/test_kb_store_pages.py`

**Background:** `Document` needs `has_original_file` and `file_relpath`. `SearchHit` needs `page` and `has_original`. `add_document` accepts an optional `pages: Sequence[tuple[int, str]]` and uses per-page chunking when provided; falls back to old behaviour (single virtual page) when only `text` is given. All SELECTs widen to include the new columns.

- [ ] **Step 1: Write the failing test for per-page chunking**

Create `tests/test_kb_store_pages.py`:

```python
"""Test per-page chunking and Document/SearchHit page_number propagation."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeBaseStore:
    return KnowledgeBaseStore(tmp_path / "test.sqlite")


def test_legacy_text_path_still_works(store: KnowledgeBaseStore) -> None:
    """add_document(title, text=...) keeps old behaviour — page_number is NULL."""
    doc = store.add_document("Title", text="a" * 1500)
    assert doc.id > 0
    assert doc.has_original_file is False
    assert doc.file_relpath is None
    # search → SearchHit.page is None for legacy chunks
    hits = store.search("a")
    assert hits
    assert all(hit.page is None for hit in hits)
    assert all(hit.has_original is False for hit in hits)


def test_pages_path_assigns_page_per_chunk(store: KnowledgeBaseStore) -> None:
    """add_document(pages=[...]) chunks per page and preserves page_number."""
    pages = [
        (1, "page one " * 200),  # large enough to produce 1+ chunks
        (2, "page two " * 200),
        (3, "page three " * 200),
    ]
    doc = store.add_document("Pages", pages=pages, source="file", filename="x.pdf")
    assert doc.id > 0
    # Force search returns hits per page
    hits = store.search("page one", top_k=10)
    assert hits
    pages_found = {hit.page for hit in hits if "page one" in hit.text}
    assert pages_found == {1}, f"expected page 1 only, got {pages_found}"


def test_update_file_metadata_marks_has_original(store: KnowledgeBaseStore) -> None:
    """update_file_metadata flips has_original_file and stores relpath."""
    doc = store.add_document("doc", text="hi")
    store.update_file_metadata(doc.id, file_relpath="kb_files/1.pdf")
    refreshed = store.get_document(doc.id)
    assert refreshed is not None
    assert refreshed.has_original_file is True
    assert refreshed.file_relpath == "kb_files/1.pdf"


def test_search_includes_has_original_flag(store: KnowledgeBaseStore) -> None:
    """Searched chunks of a doc with original file get has_original=True."""
    doc = store.add_document(
        "doc",
        pages=[(1, "alpha beta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(doc.id, file_relpath="kb_files/1.pdf")
    hits = store.search("alpha", top_k=5)
    assert hits
    assert all(hit.has_original for hit in hits)
    assert all(hit.page == 1 for hit in hits)


def test_add_document_pages_and_text_raise(store: KnowledgeBaseStore) -> None:
    """Passing both text and pages is ambiguous → raise."""
    with pytest.raises(ValueError):
        store.add_document("x", text="foo", pages=[(1, "bar")])


def test_add_document_empty_pages_raises(store: KnowledgeBaseStore) -> None:
    """All-empty pages → ValueError(Text is empty)."""
    with pytest.raises(ValueError):
        store.add_document("x", pages=[(1, ""), (2, "  ")])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```powershell
py -m pytest tests/test_kb_store_pages.py -v
```
Expected: FAIL (TypeError on `pages=` kwarg, no `has_original_file` field on Document, no `page` field on SearchHit).

- [ ] **Step 3: Add new fields to `Document` and `SearchHit` dataclasses**

Open `app/services/kb_store.py`. Find `class Document` (line ~68-79) and replace with:

```python
@dataclass(frozen=True)
class Document:
    """A stored document with chunk-count and origin metadata."""

    id: int
    title: str
    text: str
    created_at: str
    chunks: int
    source: str = "text"
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    has_original_file: bool = False
    file_relpath: Optional[str] = None
```

Find `class SearchHit` (line ~82-92) and replace with:

```python
@dataclass(frozen=True)
class SearchHit:
    """One ranked chunk returned by similarity search."""

    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    source: str = "text"
    filename: Optional[str] = None
    page: Optional[int] = None
    has_original: bool = False
```

- [ ] **Step 4: Extend `_init_schema` to include new columns**

Find `_init_schema` (line ~232-274). Update the `kb_documents` and `kb_chunks` `CREATE TABLE IF NOT EXISTS` statements to include the new columns. The current statement is:

```python
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'text',
                    filename TEXT,
                    mime_type TEXT
                );
```

Replace with:

```python
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'text',
                    filename TEXT,
                    mime_type TEXT,
                    has_original_file INTEGER NOT NULL DEFAULT 0,
                    file_relpath TEXT
                );
```

For `kb_chunks`, change:

```python
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedder TEXT NOT NULL DEFAULT 'hash',
                    dim INTEGER NOT NULL DEFAULT 256
                );
```

To:

```python
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedder TEXT NOT NULL DEFAULT 'hash',
                    dim INTEGER NOT NULL DEFAULT 256,
                    page_number INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_page ON kb_chunks(document_id, page_number);
```

**Why DEFAULT 0 not FALSE:** SQLite has no boolean type — it stores 0/1 in INTEGER. SQLAlchemy via the alembic migration uses `sa.Boolean`, which maps to INTEGER on SQLite. Hand-written SQL must match.

**Why duplicate the column in `_init_schema` AND in the migration:** the store is used both with and without alembic. When you instantiate `KnowledgeBaseStore` directly (e.g. in tests, in `scripts/dev_server_mvp.py`), `_init_schema` runs and is the source of truth. The migration only matters when DB was set up via alembic. Both paths must agree.

- [ ] **Step 5: Update `add_document` signature and `_row_to_document`**

Find `add_document` (line ~295-346). Replace with:

```python
    def add_document(
        self,
        title: str,
        text: Optional[str] = None,
        *,
        pages: Optional[Sequence[tuple[int, str]]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        source: str = "text",
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Document:
        if text is not None and pages is not None:
            raise ValueError("Pass either text= or pages=, not both")

        cleaned_title = (title or "").strip() or "Untitled"
        if len(cleaned_title) > 300:
            cleaned_title = cleaned_title[:300]

        # Normalise input to per-page form. Legacy text= goes in as a single
        # virtual page; pages= is used verbatim. Both go through split_text.
        if pages is not None:
            normalised: list[tuple[int, str]] = []
            for page_no, page_text in pages:
                cleaned_page = (page_text or "").strip()
                if cleaned_page:
                    normalised.append((int(page_no), cleaned_page))
            if not normalised:
                raise ValueError("Text is empty")
            full_text = "\n\n".join(t for _, t in normalised)
            if len(full_text) > MAX_TEXT_LEN:
                raise ValueError(f"Text exceeds {MAX_TEXT_LEN} characters")
        else:
            cleaned_text = (text or "").strip()
            if not cleaned_text:
                raise ValueError("Text is empty")
            if len(cleaned_text) > MAX_TEXT_LEN:
                raise ValueError(f"Text exceeds {MAX_TEXT_LEN} characters")
            normalised = [(None, cleaned_text)]  # None — no page info
            full_text = cleaned_text

        # Per-page chunking — each chunk remembers its source page number
        chunks_with_pages: list[tuple[Optional[int], str]] = []
        for page_no, page_text in normalised:
            page_chunks = split_text(page_text, chunk_size=chunk_size, overlap=overlap) or [page_text]
            for chunk in page_chunks:
                chunks_with_pages.append((page_no, chunk))

        chunk_texts = [t for _, t in chunks_with_pages]
        created_at = datetime.now(timezone.utc).isoformat()
        embedded_blobs, embedder_name, dim = self._embed_chunks(chunk_texts)

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO kb_documents(title, text, created_at, source, filename, mime_type,
                                          has_original_file, file_relpath)
                VALUES(?, ?, ?, ?, ?, ?, 0, NULL)
                """,
                (cleaned_title, full_text, created_at, source, filename, mime_type),
            )
            doc_id = int(cur.lastrowid)
            for idx, ((page_no, chunk), blob) in enumerate(zip(chunks_with_pages, embedded_blobs)):
                conn.execute(
                    """
                    INSERT INTO kb_chunks(document_id, chunk_index, text, embedding,
                                           embedder, dim, page_number)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, idx, chunk, blob, embedder_name, dim, page_no),
                )

        return Document(
            id=doc_id,
            title=cleaned_title,
            text=full_text,
            created_at=created_at,
            chunks=len(chunks_with_pages),
            source=source,
            filename=filename,
            mime_type=mime_type,
            has_original_file=False,
            file_relpath=None,
        )
```

Find `_row_to_document` (line ~348-359). Replace with:

```python
    @staticmethod
    def _row_to_document(row: tuple) -> Document:
        # Row order matches the SELECTs in list_documents/get_document.
        return Document(
            id=row[0],
            title=row[1],
            text=row[2],
            created_at=row[3],
            chunks=int(row[4] or 0),
            source=row[5] or "text",
            filename=row[6],
            mime_type=row[7],
            has_original_file=bool(row[8]),
            file_relpath=row[9],
        )
```

- [ ] **Step 6: Update `list_documents` and `get_document` SELECTs**

Find `list_documents` (line ~361-372). Replace SQL with:

```python
        rows = conn.execute(
            """
            SELECT d.id, d.title, d.text, d.created_at,
                (SELECT COUNT(*) FROM kb_chunks c WHERE c.document_id = d.id) AS chunks,
                d.source, d.filename, d.mime_type,
                d.has_original_file, d.file_relpath
            FROM kb_documents d
            ORDER BY d.id DESC
            """
        ).fetchall()
```

Find `get_document` (line ~374-385). Replace SQL with:

```python
            row = conn.execute(
                """
                SELECT d.id, d.title, d.text, d.created_at,
                    (SELECT COUNT(*) FROM kb_chunks c WHERE c.document_id = d.id) AS chunks,
                    d.source, d.filename, d.mime_type,
                    d.has_original_file, d.file_relpath
                FROM kb_documents d WHERE d.id = ?
                """,
                (int(doc_id),),
            ).fetchone()
```

- [ ] **Step 7: Update `search` SELECT to include page_number and has_original_file**

Find `search` (line ~392-443). Replace the SQL block (lines ~407-416) with:

```python
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.document_id, d.title, c.chunk_index, c.text, c.embedding, c.dim,
                       d.source, d.filename, c.page_number, d.has_original_file
                FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id
                WHERE c.dim = ?
                LIMIT ?
                """,
                (q_dim, hard_limit),
            ).fetchall()
```

And replace the row-unpack block (lines ~423-441) with:

```python
        scored: List[Tuple[float, SearchHit]] = []
        for doc_id, title, idx, text, blob, _dim, source, filename, page_number, has_original in rows:
            score = _cosine(q_vec, self._unpack(blob))
            if score <= 0.0:
                continue
            scored.append(
                (
                    score,
                    SearchHit(
                        document_id=int(doc_id),
                        document_title=title,
                        chunk_index=int(idx),
                        text=text,
                        score=score,
                        source=source or "text",
                        filename=filename,
                        page=int(page_number) if page_number is not None else None,
                        has_original=bool(has_original),
                    ),
                )
            )
```

- [ ] **Step 8: Add `update_file_metadata` method**

Add this method to `KnowledgeBaseStore` right after `delete_document` (around line 391):

```python
    def update_file_metadata(self, doc_id: int, *, file_relpath: str) -> bool:
        """Flip has_original_file=1 and store the relative blob path.

        Returns True if a row was updated. Caller should ensure the file
        actually exists at ``<settings.data_dir>/<file_relpath>`` first;
        this method does not verify the filesystem.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE kb_documents SET has_original_file = 1, file_relpath = ? WHERE id = ?",
                (file_relpath, int(doc_id)),
            )
            return cur.rowcount > 0
```

- [ ] **Step 9: Run the tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_store_pages.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 10: Run existing kb_mvp tests to ensure no regression**

Run:
```powershell
py -m pytest tests/test_kb_mvp.py -v
```
Expected: PASS. Any failure here likely means `_row_to_document` broke a caller — check the failure and fix.

- [ ] **Step 11: Commit**

```powershell
git add app/services/kb_store.py tests/test_kb_store_pages.py
git commit -m @'
feat(kb-mvp): per-page chunking and page_number in SearchHit

Adds pages= parameter to add_document for per-page chunking. Legacy
text= path is preserved (chunks get page_number=NULL). Document and
SearchHit dataclasses extended with has_original_file/file_relpath
and page/has_original respectively. update_file_metadata enables
post-INSERT file association.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task A.3: Extend `HitOut` (Pydantic) + `_hit_to_out` + verify propagation through `/search`, `/ask`, `/ask/stream`

**Files:**
- Modify: `app/api/kb_mvp.py` (`HitOut`, `_hit_to_out`, `_sources_payload_to_hit_out`)
- Create: `tests/test_kb_mvp_search_response.py`

- [ ] **Step 1: Write the failing test for HitOut propagation**

Create `tests/test_kb_mvp_search_response.py`:

```python
"""Test that page and has_original propagate through search/ask responses."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services import kb_llm
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeBaseStore:
    return KnowledgeBaseStore(tmp_path / "test.sqlite")


@pytest.fixture
def app_with_store(store: KnowledgeBaseStore, monkeypatch):
    """Build a minimal FastAPI app with the MVP router and pinned store."""
    from fastapi import FastAPI
    from app.api.kb_mvp import router

    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    # Force extractive answer to avoid hitting any real LLM
    monkeypatch.setattr(kb_llm, "select_provider", lambda: None)
    return fastapi_app


def test_search_response_has_page_and_has_original(app_with_store, store):
    store.add_document(
        "doc1",
        pages=[(1, "alpha beta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(1, file_relpath="kb_files/1.pdf")

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/search", json={"query": "alpha"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["hits"], "expected hits"
    for hit in data["hits"]:
        assert hit["page"] == 1
        assert hit["has_original"] is True


def test_ask_response_has_page_and_has_original(app_with_store, store):
    store.add_document(
        "doc1",
        pages=[(1, "alpha beta " * 50), (2, "gamma delta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(1, file_relpath="kb_files/1.pdf")

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/ask", json={"question": "alpha"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"], "expected sources"
    # At least one source should match page 1
    pages = {s["page"] for s in data["sources"]}
    assert 1 in pages or 2 in pages, f"no page info in sources: {data['sources']}"
    assert all(s["has_original"] is True for s in data["sources"])


def test_legacy_text_document_has_null_page(app_with_store, store):
    """Documents added via legacy text= path should have page=null."""
    store.add_document("legacy", text="alpha beta " * 50)

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/search", json={"query": "alpha"})
    data = resp.json()
    for hit in data["hits"]:
        assert hit["page"] is None
        assert hit["has_original"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_mvp_search_response.py -v
```
Expected: FAIL — `KeyError: 'page'` or `'has_original'` in the response JSON.

- [ ] **Step 3: Extend the `HitOut` model**

Open `app/api/kb_mvp.py`. Find `class HitOut` (line ~115-124). Replace with:

```python
class HitOut(BaseModel):
    """A single ranked chunk — used by both ``/search`` and ``/ask``."""

    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    source: str = "text"
    filename: Optional[str] = None
    page: Optional[int] = None
    has_original: bool = False
```

- [ ] **Step 4: Update `_hit_to_out`**

Find `_hit_to_out` (line ~246-255). Replace with:

```python
def _hit_to_out(hit: SearchHit) -> HitOut:
    return HitOut(
        document_id=hit.document_id,
        document_title=hit.document_title,
        chunk_index=hit.chunk_index,
        text=hit.text,
        score=round(hit.score, 6),
        source=hit.source,
        filename=hit.filename,
        page=hit.page,
        has_original=hit.has_original,
    )
```

- [ ] **Step 5: Update `_sources_payload_to_hit_out` for restoring persisted message sources**

Find `_sources_payload_to_hit_out` (line ~268-287). Replace the `HitOut(...)` constructor block to include the new fields:

```python
def _sources_payload_to_hit_out(items: List[Any]) -> List[HitOut]:
    out: List[HitOut] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        try:
            page_val = raw.get("page")
            page_int: Optional[int]
            if page_val is None:
                page_int = None
            else:
                try:
                    page_int = int(page_val)
                except (TypeError, ValueError):
                    page_int = None
            out.append(
                HitOut(
                    document_id=int(raw.get("document_id") or 0),
                    document_title=str(raw.get("document_title") or ""),
                    chunk_index=int(raw.get("chunk_index") or 0),
                    text=str(raw.get("text") or ""),
                    score=float(raw.get("score") or 0.0),
                    source=str(raw.get("source") or "text"),
                    filename=raw.get("filename") if isinstance(raw.get("filename"), str) else None,
                    page=page_int,
                    has_original=bool(raw.get("has_original") or False),
                )
            )
        except (TypeError, ValueError):
            continue
    return out
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_mvp_search_response.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 7: Run the broader MVP suite to confirm nothing else broke**

Run:
```powershell
py -m pytest tests/test_kb_mvp.py tests/test_kb_store_pages.py tests/test_kb_mvp_search_response.py -v
```
Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add app/api/kb_mvp.py tests/test_kb_mvp_search_response.py
git commit -m @'
feat(kb-mvp): expose page and has_original in citation sources

HitOut Pydantic model gains page (Optional[int]) and has_original (bool).
_hit_to_out maps from SearchHit. _sources_payload_to_hit_out restores
fields from persisted message JSON, tolerating missing keys for backward
compatibility with rows written before this migration.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

## Section B — Blob upload + file endpoint (~6h, 4 tasks)

### Task B.1: Add `_parse_file_bytes_with_pages` helper

**Files:**
- Modify: `app/api/kb_mvp.py` (new helper alongside `_parse_file_bytes`)
- Create: `tests/test_parse_file_bytes_with_pages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parse_file_bytes_with_pages.py`:

```python
"""Test _parse_file_bytes_with_pages helper."""
from __future__ import annotations

import pytest

from app.api.kb_mvp import _parse_file_bytes_with_pages


def test_parse_txt_returns_single_page():
    pages, mime = _parse_file_bytes_with_pages("notes.txt", b"hello world")
    assert pages == [(1, "hello world")]
    assert mime == "text/plain"


def test_parse_md_returns_single_page():
    pages, mime = _parse_file_bytes_with_pages("notes.md", b"# Title\n\nbody")
    assert pages == [(1, "# Title\n\nbody")]
    assert mime == "text/markdown"


def test_parse_empty_extension_falls_back_to_text():
    pages, mime = _parse_file_bytes_with_pages("noext", b"raw bytes")
    assert pages == [(1, "raw bytes")]
    assert mime == "text/plain"


def test_parse_rich_format_returns_multiple_pages(monkeypatch):
    """When parse_document yields pages, helper preserves them."""
    class FakeResult:
        pages = [(1, "page 1 text"), (2, "page 2 text")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    def fake_parse_document(filename, data):
        return FakeResult()

    import app.api.kb_mvp as kb_mvp
    monkeypatch.setattr("app.ingest.chunking.parse_document", fake_parse_document)

    pages, mime = _parse_file_bytes_with_pages("doc.pdf", b"%PDF-1.4")
    assert pages == [(1, "page 1 text"), (2, "page 2 text")]
    assert mime == "application/pdf"


def test_parse_rich_format_drops_empty_pages(monkeypatch):
    class FakeResult:
        pages = [(1, ""), (2, "real"), (3, "  ")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())
    pages, mime = _parse_file_bytes_with_pages("doc.pdf", b"%PDF-1.4")
    assert pages == [(2, "real")]
    assert mime == "application/pdf"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_parse_file_bytes_with_pages.py -v
```
Expected: FAIL with `ImportError: cannot import name '_parse_file_bytes_with_pages'`.

- [ ] **Step 3: Write the helper**

Open `app/api/kb_mvp.py`. Find `_parse_file_bytes` (line ~458-496). Add a new function **immediately after** it:

```python
def _parse_file_bytes_with_pages(
    filename: str, data: bytes
) -> tuple[list[tuple[int, str]], str]:
    """Like :func:`_parse_file_bytes` but preserves per-page structure.

    Returns ``(pages, mime_type)`` where ``pages`` is a list of
    ``(page_number, text)`` tuples (page numbers 1-indexed). Empty pages
    are dropped. Plain-text formats produce a single virtual page.

    Raises ``HTTPException`` on parse failure (same contract as
    ``_parse_file_bytes``).
    """

    ext = _extension_for(filename)
    if not ext:
        text = _decode_text(data).strip()
        return ([(1, text)] if text else []), "text/plain"

    if ext in {"txt", "md", "markdown"}:
        text = _decode_text(data).strip()
        mime = "text/markdown" if ext != "txt" else "text/plain"
        return ([(1, text)] if text else []), mime

    try:
        from app.ingest.chunking import parse_document
    except Exception as exc:  # pragma: no cover - optional dep missing
        LOGGER.warning("parse_document unavailable (%s); decoding as text", exc)
        text = _decode_text(data).strip()
        return ([(1, text)] if text else []), "application/octet-stream"

    try:
        result = parse_document(filename, data)
    except Exception as exc:
        LOGGER.exception("Failed to parse %s: %s", filename, exc)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"FAILED_TO_PARSE: {exc}",
        ) from exc

    raw_pages = getattr(result, "pages", []) or []
    pages: list[tuple[int, str]] = []
    for page_number, page_text in raw_pages:
        text = (str(page_text) if page_text is not None else "").strip()
        if text:
            pages.append((int(page_number), text))

    mime = (result.metadata.get("document", {}) or {}).get("mime_type") or "application/octet-stream"
    return pages, mime
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```powershell
py -m pytest tests/test_parse_file_bytes_with_pages.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```powershell
git add app/api/kb_mvp.py tests/test_parse_file_bytes_with_pages.py
git commit -m @'
feat(kb-mvp): add _parse_file_bytes_with_pages helper

Parallel helper to _parse_file_bytes that preserves Docling per-page
structure. Empty pages dropped. Plain-text formats produce a single
virtual page. Parse errors raise the same 422 HTTPException as the
legacy helper.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task B.2: Modify `upload_document` to save PDF blob and use pages

**Files:**
- Modify: `app/api/kb_mvp.py` (`upload_document`)
- Create: `tests/test_kb_mvp_upload_blob.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_mvp_upload_blob.py`:

```python
"""Test PDF blob persistence in upload_document."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_and_store(tmp_path: Path, monkeypatch):
    # Pin data_dir to tmp_path so the test owns var/data/kb_files/
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Force settings reload — pattern depends on app.core.config caching
    from app.core import config as _cfg
    if hasattr(_cfg, "get_settings"):
        _cfg.get_settings.cache_clear()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")

    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app, store, tmp_path


def _minimal_pdf_bytes() -> bytes:
    """Return a tiny valid PDF that Docling can parse, or a stub that
    `parse_document` recognises. Use a real header + one page."""
    # PDF.js will parse this; for upload-side tests we patch the parser
    # instead of relying on real Docling parsing.
    return b"%PDF-1.4\n%minimal\n"


def test_upload_pdf_persists_blob(app_and_store, monkeypatch):
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha beta " * 30), (2, "gamma delta " * 30)]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    pdf_bytes = _minimal_pdf_bytes()
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"title": "Doc"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    doc_id = body["id"]

    # Blob saved
    blob_path = tmp_path / "kb_files" / f"{doc_id}.pdf"
    assert blob_path.exists(), f"blob missing at {blob_path}"
    assert blob_path.read_bytes() == pdf_bytes

    # has_original_file should now be true
    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc.has_original_file is True
    assert doc.file_relpath == f"kb_files/{doc_id}.pdf"


def test_upload_non_pdf_does_not_save_blob(app_and_store, monkeypatch):
    app, store, tmp_path = app_and_store

    client = TestClient(app)
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    kb_files = tmp_path / "kb_files"
    if kb_files.exists():
        assert not any(kb_files.iterdir()), "non-PDF should not produce a blob"

    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc.has_original_file is False
    assert doc.file_relpath is None


def test_upload_pdf_orphan_tmp_cleaned_on_parse_error(app_and_store, monkeypatch):
    """If parse_document raises, no tmp-* blob should remain."""
    app, store, tmp_path = app_and_store

    def broken_parse(*_):
        raise RuntimeError("synthetic parse failure")

    monkeypatch.setattr("app.ingest.chunking.parse_document", broken_parse)

    client = TestClient(app)
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4\n"), "application/pdf")},
    )
    assert resp.status_code == 422, resp.text

    kb_files = tmp_path / "kb_files"
    if kb_files.exists():
        leftovers = [p for p in kb_files.iterdir() if p.name.startswith(".tmp-")]
        assert leftovers == [], f"orphan tmp blobs found: {leftovers}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_mvp_upload_blob.py -v
```
Expected: FAIL — `blob_path` not created; `has_original_file` is False after PDF upload.

- [ ] **Step 3: Modify `upload_document` to save the blob**

Open `app/api/kb_mvp.py`. Add to the imports at the top:

```python
import uuid
from pathlib import Path
```

Find `upload_document` (line ~540-587). Replace the **body** of the function (everything after the docstring) with:

```python
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="FILENAME_REQUIRED")

    ext = _extension_for(filename)
    if ext not in SUPPORTED_UPLOAD_EXT:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"UNSUPPORTED_EXT: .{ext}. Allowed: {', '.join(sorted(SUPPORTED_UPLOAD_EXT))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="EMPTY_FILE")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"FILE_TOO_LARGE: max {MAX_UPLOAD_BYTES} bytes",
        )

    # For PDFs we keep the raw blob so the viewer can show the original.
    # Write to a tmp name first; rename to <doc_id>.pdf AFTER the DB INSERT
    # succeeds, so we never leave a blob without a matching row.
    tmp_blob: Optional[Path] = None
    kb_files_dir: Optional[Path] = None
    if ext == "pdf":
        from app.core.config import get_settings  # local import to avoid cycles
        settings = get_settings()
        kb_files_dir = Path(settings.data_dir) / "kb_files"
        kb_files_dir.mkdir(parents=True, exist_ok=True)
        tmp_blob = kb_files_dir / f".tmp-{uuid.uuid4().hex}.pdf"
        tmp_blob.write_bytes(data)

    try:
        pages, mime_type = _parse_file_bytes_with_pages(filename, data)
        if not pages:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail="NO_EXTRACTABLE_TEXT"
            )

        effective_title = (title or "").strip() or filename
        store = _store_for(request)
        try:
            doc = store.add_document(
                effective_title,
                pages=pages,
                source="file",
                filename=filename,
                mime_type=mime_type,
            )
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        # Promote tmp blob to final name and mark the doc.
        if tmp_blob is not None and kb_files_dir is not None:
            final_blob = kb_files_dir / f"{doc.id}.pdf"
            tmp_blob.rename(final_blob)
            tmp_blob = None  # ownership transferred
            store.update_file_metadata(doc.id, file_relpath=f"kb_files/{doc.id}.pdf")
            # Refresh the in-memory Document so we return up-to-date flags
            refreshed = store.get_document(doc.id)
            if refreshed is not None:
                doc = refreshed

        return _doc_to_out(doc)
    finally:
        # Clean up orphan tmp on any error path
        if tmp_blob is not None:
            try:
                tmp_blob.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("failed to remove tmp blob %s", tmp_blob)
```

- [ ] **Step 4: Run the upload test to verify all paths**

Run:
```powershell
py -m pytest tests/test_kb_mvp_upload_blob.py -v
```
Expected: PASS (3 tests). If the third (orphan cleanup) fails because the `Path.rename` happens before the parse failure, re-check the order: the parse call is **after** the `tmp_blob.write_bytes`, and the `finally` covers any failure between. The cleanup test patches `parse_document` to raise, which is hit inside `_parse_file_bytes_with_pages`, after the tmp write.

- [ ] **Step 5: Run the full MVP suite to confirm no regression**

Run:
```powershell
py -m pytest tests/test_kb_mvp.py tests/test_kb_mvp_upload_blob.py tests/test_kb_store_pages.py tests/test_kb_mvp_search_response.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/api/kb_mvp.py tests/test_kb_mvp_upload_blob.py
git commit -m @'
feat(kb-mvp): persist PDF blob in var/data/kb_files/ on upload

PDF uploads now save the raw bytes to a tmp file before parsing.
After a successful INSERT the tmp is renamed to <doc_id>.pdf and
has_original_file is flipped via update_file_metadata. Any failure
between tmp-write and final-rename cleans up the orphan tmp.
Non-PDF uploads behave as before (no blob).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task B.3: Cascade deletion — remove blob on `DELETE /api/kb/documents/{id}`

**Files:**
- Modify: `app/api/kb_mvp.py` (`delete_document` endpoint)
- Modify: `tests/test_kb_mvp_upload_blob.py` (add cascade test)

- [ ] **Step 1: Add the cascade test**

Append to `tests/test_kb_mvp_upload_blob.py`:

```python
def test_delete_document_removes_blob(app_and_store, monkeypatch):
    """DELETE on a doc with original_file removes both DB row and blob."""
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4\n"), "application/pdf")},
    )
    doc_id = resp.json()["id"]
    blob_path = tmp_path / "kb_files" / f"{doc_id}.pdf"
    assert blob_path.exists()

    resp = client.delete(f"/api/kb/documents/{doc_id}")
    assert resp.status_code == 200, resp.text

    assert not blob_path.exists(), "blob should be removed after delete"
    assert store.get_document(doc_id) is None


def test_delete_document_without_blob_no_error(app_and_store):
    """DELETE on a non-PDF doc completes even though no blob exists."""
    app, store, tmp_path = app_and_store

    client = TestClient(app)
    client.post(
        "/api/kb/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")},
    )
    doc_id = 1  # first doc

    resp = client.delete(f"/api/kb/documents/{doc_id}")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_mvp_upload_blob.py::test_delete_document_removes_blob -v
```
Expected: FAIL — blob still exists after delete.

- [ ] **Step 3: Update `delete_document` to remove the blob**

Open `app/api/kb_mvp.py`. Find `delete_document` (line ~619-626). Replace with:

```python
@protected.delete("/documents/{doc_id}")
def delete_document(doc_id: int, request: Request) -> dict[str, Any]:
    """Delete a document, its chunks, and the original blob (if any).

    The blob is removed BEFORE the DB row so an orphaned filesystem entry
    is never possible. If the file vanished (race / manual cleanup), we
    log a warning but still drop the DB row — the goal is to satisfy
    DELETE, not to fail because of dangling state.
    """

    store = _store_for(request)
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")

    if doc.has_original_file and doc.file_relpath:
        from app.core.config import get_settings
        settings = get_settings()
        blob_path = Path(settings.data_dir) / doc.file_relpath
        try:
            blob_path.unlink(missing_ok=True)
        except OSError as exc:
            LOGGER.warning("failed to remove blob for doc %d: %s", doc_id, exc)

    if not store.delete_document(doc_id):
        # Race: someone else deleted it between get_document and delete.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")
    return {"ok": True, "id": doc_id}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_mvp_upload_blob.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```powershell
git add app/api/kb_mvp.py tests/test_kb_mvp_upload_blob.py
git commit -m @'
feat(kb-mvp): cascade-delete original blob with document

DELETE /api/kb/documents/{id} now removes <data_dir>/<file_relpath>
before dropping the DB row. Missing blob is logged but not fatal.
Non-PDF documents (no blob) continue to delete cleanly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task B.4: New `GET /api/kb/documents/{doc_id}/file` endpoint

**Files:**
- Modify: `app/api/kb_mvp.py` (new endpoint + import `FileResponse`)
- Create: `tests/test_kb_mvp_file_endpoint.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kb_mvp_file_endpoint.py`:

```python
"""Test GET /api/kb/documents/{id}/file endpoint."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_and_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import config as _cfg
    if hasattr(_cfg, "get_settings"):
        _cfg.get_settings.cache_clear()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app, store, tmp_path


@pytest.fixture
def uploaded_pdf(app_and_store, monkeypatch):
    """Upload one PDF and return (client, doc_id, tmp_path)."""
    app, store, tmp_path = app_and_store

    class FakeResult:
        pages = [(1, "alpha"), (2, "beta")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    pdf_bytes = b"%PDF-1.4\nhello\n"
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    return client, resp.json()["id"], tmp_path, pdf_bytes


def test_file_endpoint_returns_pdf(uploaded_pdf):
    client, doc_id, _, pdf_bytes = uploaded_pdf
    resp = client.get(f"/api/kb/documents/{doc_id}/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "inline" in resp.headers["content-disposition"]
    assert resp.content == pdf_bytes


def test_file_endpoint_404_for_unknown_doc(app_and_store):
    app, _, _ = app_and_store
    client = TestClient(app)
    resp = client.get("/api/kb/documents/99999/file")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "DOCUMENT_NOT_FOUND"


def test_file_endpoint_404_for_doc_without_original(app_and_store):
    app, store, _ = app_and_store
    store.add_document("txt", text="hello")
    client = TestClient(app)
    resp = client.get("/api/kb/documents/1/file")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "NO_ORIGINAL_FILE"


def test_file_endpoint_410_when_blob_missing(uploaded_pdf):
    client, doc_id, tmp_path, _ = uploaded_pdf
    # Remove blob from disk while DB still references it
    blob_path = tmp_path / "kb_files" / f"{doc_id}.pdf"
    blob_path.unlink()
    resp = client.get(f"/api/kb/documents/{doc_id}/file")
    assert resp.status_code == 410
    assert resp.json()["detail"] == "FILE_DELETED"


def test_file_endpoint_path_traversal_returns_500(app_and_store):
    """If DB has been tampered to contain ../ in file_relpath, refuse."""
    app, store, tmp_path = app_and_store
    store.add_document("pdf", pages=[(1, "x")], source="file", filename="x.pdf")

    # Inject a malicious relpath directly via sqlite
    import sqlite3
    conn = sqlite3.connect(store.db_path)
    conn.execute(
        "UPDATE kb_documents SET has_original_file=1, file_relpath=? WHERE id=1",
        ("kb_files/../../../etc/passwd",),
    )
    conn.commit()
    conn.close()

    client = TestClient(app)
    resp = client.get("/api/kb/documents/1/file")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "STORAGE_ERROR"


def test_file_endpoint_requires_auth_when_key_set(app_and_store, monkeypatch):
    """When KB_API_KEY is set, the endpoint demands X-API-Key header."""
    app, store, tmp_path = app_and_store
    monkeypatch.setenv("KB_API_KEY", "secret-key-xxx")
    # Force kb_auth to re-read env
    from app.api import kb_auth as _ka
    if hasattr(_ka, "_load_api_key"):
        _ka._load_api_key.cache_clear()

    class FakeResult:
        pages = [(1, "alpha")]
        metadata = {"document": {"mime_type": "application/pdf"}}

    monkeypatch.setattr("app.ingest.chunking.parse_document", lambda *_: FakeResult())

    client = TestClient(app)
    # Upload requires auth too
    resp = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4\n"), "application/pdf")},
        headers={"X-API-Key": "secret-key-xxx"},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["id"]

    # Without header → 401
    resp = client.get(f"/api/kb/documents/{doc_id}/file")
    assert resp.status_code == 401

    # With header → 200
    resp = client.get(
        f"/api/kb/documents/{doc_id}/file",
        headers={"X-API-Key": "secret-key-xxx"},
    )
    assert resp.status_code == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```powershell
py -m pytest tests/test_kb_mvp_file_endpoint.py -v
```
Expected: all 6 FAIL (endpoint not implemented).

- [ ] **Step 3: Implement the endpoint**

Open `app/api/kb_mvp.py`. Update the imports at the top to include `FileResponse`:

```python
from fastapi.responses import FileResponse, StreamingResponse
```

Add the endpoint right after `get_document` (around line ~617):

```python
@protected.get("/documents/{doc_id}/file")
def get_document_file(doc_id: int, request: Request) -> FileResponse:
    """Stream the original blob for documents with has_original_file=true.

    Returns ``application/pdf`` with ``inline`` disposition so the
    browser/PDF.js can render it. Auth-gated by the ``protected``
    router — when ``KB_API_KEY`` is set, requires the standard
    ``X-API-Key`` header.
    """

    store = _store_for(request)
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DOCUMENT_NOT_FOUND")

    if not doc.has_original_file or not doc.file_relpath:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="NO_ORIGINAL_FILE")

    from app.core.config import get_settings
    settings = get_settings()
    data_dir = Path(settings.data_dir).resolve()
    expected_root = (data_dir / "kb_files").resolve()
    absolute = (data_dir / doc.file_relpath).resolve()

    # Path-traversal guard: resolved path must live under <data_dir>/kb_files/
    try:
        absolute.relative_to(expected_root)
    except ValueError:
        LOGGER.error(
            "Path traversal attempted for doc %d: %s", doc_id, doc.file_relpath
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STORAGE_ERROR"
        )

    if not absolute.is_file():
        LOGGER.warning(
            "Original file missing for doc %d: %s", doc_id, absolute
        )
        raise HTTPException(status.HTTP_410_GONE, detail="FILE_DELETED")

    return FileResponse(
        absolute,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{doc.filename or doc_id}.pdf"',
        },
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_mvp_file_endpoint.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 5: Full MVP suite check**

Run:
```powershell
py -m pytest tests/test_kb_mvp.py tests/test_kb_mvp_upload_blob.py tests/test_kb_mvp_file_endpoint.py tests/test_kb_store_pages.py tests/test_kb_mvp_search_response.py tests/test_parse_file_bytes_with_pages.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/api/kb_mvp.py tests/test_kb_mvp_file_endpoint.py
git commit -m @'
feat(kb-mvp): add GET /documents/{id}/file endpoint

Streams the original PDF blob with inline Content-Disposition and
application/pdf media type. Returns 404 (no_original_file or doc),
410 (file deleted from FS but DB still references), 500 (path
traversal in file_relpath) — covered by unit tests using
Path.resolve().relative_to() for the guard. Auth-gated by the
existing protected router via X-API-Key when KB_API_KEY is set.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

## Section C — Frontend (~8h, 7 tasks)

### Task C.1: Vendor PDF.js into `data/www/vendor/pdfjs/`

**Files:**
- Create: `data/www/vendor/pdfjs/build/pdf.mjs`
- Create: `data/www/vendor/pdfjs/build/pdf.worker.mjs`
- Create: `data/www/vendor/pdfjs/LICENSE`
- Modify: `.gitattributes` (optional — mark vendor files as binary so they don't trigger line-ending warnings)

**Background:** PDF.js v4.10.38 legacy build from Mozilla. Apache-2.0. ~2.5MB gzipped. Lazy-imported by `pdf-viewer.js`.

- [ ] **Step 1: Download the legacy build**

Run (PowerShell):
```powershell
$ver = "4.10.38"
$base = "https://github.com/mozilla/pdf.js/releases/download/v$ver"
$dest = "data\www\vendor\pdfjs"
New-Item -ItemType Directory -Force "$dest\build" | Out-Null
Invoke-WebRequest -OutFile "$dest\pdfjs-$ver-legacy-dist.zip" -Uri "$base/pdfjs-$ver-legacy-dist.zip"
Expand-Archive -Path "$dest\pdfjs-$ver-legacy-dist.zip" -DestinationPath "$dest\_unpacked"
Copy-Item "$dest\_unpacked\build\pdf.mjs" "$dest\build\pdf.mjs"
Copy-Item "$dest\_unpacked\build\pdf.worker.mjs" "$dest\build\pdf.worker.mjs"
Copy-Item "$dest\_unpacked\LICENSE" "$dest\LICENSE"
Remove-Item -Recurse -Force "$dest\_unpacked"
Remove-Item "$dest\pdfjs-$ver-legacy-dist.zip"
```

If the download URL changes between plan-write and plan-execute, fall back to:
```powershell
npm pack pdfjs-dist@$ver
# then untar pdfjs-dist-*.tgz and copy build/pdf.mjs, build/pdf.worker.mjs, LICENSE
```

- [ ] **Step 2: Verify files exist and are non-empty**

Run:
```powershell
Get-ChildItem "data\www\vendor\pdfjs\build" | Select-Object Name, Length
Get-Item "data\www\vendor\pdfjs\LICENSE"
```
Expected: `pdf.mjs` (~1MB+), `pdf.worker.mjs` (~1MB+), `LICENSE` (~10KB).

- [ ] **Step 3: Verify .gitignore doesn't exclude vendor**

Run:
```powershell
git check-ignore data\www\vendor\pdfjs\build\pdf.mjs
```
Expected: empty output (path not ignored). If ignored, inspect `.gitignore` and adjust — usually `node_modules/` is the culprit, but `vendor/` should be safe.

- [ ] **Step 4: Add `.gitattributes` entry to suppress LF/CRLF noise**

Append (or create) `.gitattributes`:

```
data/www/vendor/** -text
```

- [ ] **Step 5: Commit**

```powershell
git add data/www/vendor/pdfjs .gitattributes
git commit -m @'
chore(deps): vendor PDF.js v4.10.38 legacy build (Apache-2.0)

Adds Mozilla PDF.js core + worker to data/www/vendor/pdfjs/. Used by
the upcoming pdf-viewer.js for client-side PDF rendering with
text-search highlight via PDFFindController. Legacy build chosen for
compatibility with older corporate browsers (Edge, Safari).

.gitattributes marks the vendored files as binary so git does not
attempt LF/CRLF normalisation on each pull.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task C.2: Extract auth helpers into `data/www/js/kb-auth.js`

**Files:**
- Create: `data/www/js/kb-auth.js`
- Modify: `data/www/index.html` (replace inline `getApiKey`/`withAuthHeaders`/`rawApi` with calls to `window.kbAuth`, add `<script src="/js/kb-auth.js">`)
- Create: `tests/test_kb_auth_js.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_auth_js.py`:

```python
"""Verify kb-auth.js exports the expected API and index.html uses it."""
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WWW = ROOT / "data" / "www"
JS = WWW / "js"


def test_kb_auth_js_exists():
    assert (JS / "kb-auth.js").exists(), "kb-auth.js missing"


def test_kb_auth_js_exports_window_namespace():
    content = (JS / "kb-auth.js").read_text(encoding="utf-8")
    assert "window.kbAuth" in content, "kb-auth.js must define window.kbAuth"
    for fn in ("getApiKey", "withAuthHeaders", "fetch"):
        assert fn in content, f"kb-auth.js missing {fn}"


def test_kb_auth_js_uses_correct_storage_key():
    content = (JS / "kb-auth.js").read_text(encoding="utf-8")
    # Must match the key used by the existing inline UI
    assert '"kb_mvp_api_key"' in content


def test_index_html_loads_kb_auth_js():
    html = (WWW / "index.html").read_text(encoding="utf-8")
    assert "/js/kb-auth.js" in html, "index.html must script-include kb-auth.js"
    # And must reference window.kbAuth somewhere
    assert "kbAuth" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_auth_js.py -v
```
Expected: FAIL (file missing).

- [ ] **Step 3: Create `data/www/js/kb-auth.js`**

Create `data/www/js/kb-auth.js`:

```javascript
/* Shared auth helpers for the MVP UI.
 *
 * Exposes:
 *   window.kbAuth.getApiKey()       — read API key from localStorage
 *   window.kbAuth.withAuthHeaders(h) — clone h and add X-API-Key when set
 *   window.kbAuth.fetch(path, opts) — fetch `/api/kb${path}` with auth headers
 *
 * The previous implementation was inline in index.html. This file is
 * also consumed by pdf-viewer.js for the original-blob endpoint.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "kb_mvp_api_key";
  const API_BASE = "/api/kb";

  function getApiKey() {
    try {
      return localStorage.getItem(STORAGE_KEY) || "";
    } catch (_) {
      return "";
    }
  }

  function setApiKey(value) {
    try {
      if (value) localStorage.setItem(STORAGE_KEY, value);
      else localStorage.removeItem(STORAGE_KEY);
    } catch (_) {
      /* private mode — silently degrade */
    }
  }

  function withAuthHeaders(headers) {
    const out = Object.assign({}, headers || {});
    const key = getApiKey();
    if (key) out["X-API-Key"] = key;
    return out;
  }

  function authFetch(path, opts) {
    const options = Object.assign({}, opts || {});
    options.headers = withAuthHeaders(options.headers);
    return fetch(API_BASE + path, options);
  }

  window.kbAuth = {
    storageKey: STORAGE_KEY,
    apiBase: API_BASE,
    getApiKey: getApiKey,
    setApiKey: setApiKey,
    withAuthHeaders: withAuthHeaders,
    fetch: authFetch,
  };
})();
```

- [ ] **Step 4: Update `index.html` to load and use `kb-auth.js`**

Open `data/www/index.html`. Find the line `const AUTH_STORAGE_KEY = "kb_mvp_api_key";` (around line 396). Replace the auth-helpers block (lines ~394-416 — `apiBase` through `rawApi`) with:

```html
    const $ = (id) => document.getElementById(id);
    const apiBase = window.kbAuth.apiBase;  // "/api/kb"
    const AUTH_STORAGE_KEY = window.kbAuth.storageKey;

    const getApiKey = window.kbAuth.getApiKey;
    const withAuthHeaders = window.kbAuth.withAuthHeaders;

    const json = (path, opts = {}) => {
      const headers = withAuthHeaders({ "Content-Type": "application/json", ...(opts.headers || {}) });
      return fetch(`${apiBase}${path}`, { ...opts, headers });
    };
    const rawApi = (path, opts = {}) => window.kbAuth.fetch(path, opts);
```

Then add a `<script src="/js/kb-auth.js"></script>` **before** the existing big `<script>` block:

Find:
```html
  </footer>
  </div>

  <script>
```

Replace with:
```html
  </footer>
  </div>

  <script src="/js/kb-auth.js"></script>
  <script>
```

Also find existing call sites that did `localStorage.setItem(AUTH_STORAGE_KEY, value)` (around line 516) and replace with `window.kbAuth.setApiKey(value)` for consistency. Lines around 516-528 currently:

```javascript
        if (value) localStorage.setItem(AUTH_STORAGE_KEY, value);
        else localStorage.removeItem(AUTH_STORAGE_KEY);
```

Replace with:
```javascript
        window.kbAuth.setApiKey(value);
```

And:
```javascript
        localStorage.removeItem(AUTH_STORAGE_KEY);
```
Replace with:
```javascript
        window.kbAuth.setApiKey("");
```

- [ ] **Step 5: Confirm the dev server still mounts /js/ static files**

Check `scripts/dev_server_mvp.py` and `app/core/app.py` for the static-files mount. The current setup probably mounts the entire `data/www/` directory at `/`, which would serve `/js/kb-auth.js` automatically. Verify:

```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001 --reload
```

In another shell:
```powershell
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8001/js/kb-auth.js" | Select-Object -ExpandProperty StatusCode
```
Expected: `200`. If `404`, the static mount needs adjusting — inspect `app/core/app.py` for `StaticFiles` calls and ensure the path matches `data/www`.

Stop the server with Ctrl-C.

- [ ] **Step 6: Run the tests**

Run:
```powershell
py -m pytest tests/test_kb_auth_js.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```powershell
git add data/www/js/kb-auth.js data/www/index.html tests/test_kb_auth_js.py
git commit -m @'
refactor(kb-mvp): extract auth helpers into kb-auth.js

The X-API-Key handling that lived inline in index.html now lives in
data/www/js/kb-auth.js as window.kbAuth.{getApiKey, withAuthHeaders,
fetch, setApiKey}. index.html now sources it via <script src> and
delegates. This makes the helpers available to pdf-viewer.js (added
in a later task) without duplicating localStorage key constants.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task C.3: Add `{var}` interpolation to `_loader.js`

**Files:**
- Modify: `data/www/i18n/_loader.js`
- Modify: `tests/test_i18n_loader.py` (extend)

- [ ] **Step 1: Add interpolation test**

Open `tests/test_i18n_loader.py`. Append:

```python
def test_loader_supports_interpolation():
    """_loader.js must expose a t() helper that substitutes {var} tokens."""
    content = (I18N / "_loader.js").read_text(encoding="utf-8")
    assert "t(" in content or "window.t" in content
    # The substitution pattern uses simple {key} braces — verify it's wired
    assert "{" in content and "replace" in content, (
        "_loader.js should support {var}-style interpolation in t()"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_i18n_loader.py::test_loader_supports_interpolation -v
```
Expected: FAIL.

- [ ] **Step 3: Extend `_loader.js`**

Open `data/www/i18n/_loader.js`. Find the `window.t = function (key, fallback)` block (around line 60-65 of the existing file). Replace with:

```javascript
  window.t = function (key, fallback, vars) {
    let raw;
    if (window._kbDict && Object.prototype.hasOwnProperty.call(window._kbDict, key)) {
      raw = window._kbDict[key];
    } else {
      raw = fallback || key;
    }
    if (!vars || typeof vars !== "object") return raw;
    return String(raw).replace(/\{(\w+)\}/g, (m, name) => {
      return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : m;
    });
  };
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```powershell
py -m pytest tests/test_i18n_loader.py -v
```
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```powershell
git add data/www/i18n/_loader.js tests/test_i18n_loader.py
git commit -m @'
feat(i18n): add {var} interpolation to t() helper

t(key, fallback, vars) now substitutes {name} tokens with vars[name].
Used by citation rendering for "{filename}, стр. {page}" format
strings. Backwards compatible — t(key) and t(key, fallback) keep
their previous semantics.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task C.4: Add citation/viewer i18n keys to `ru.json`

**Files:**
- Modify: `data/www/i18n/ru.json`
- Modify: `tests/test_i18n_loader.py` (verify new keys present)

- [ ] **Step 1: Extend the minimum-keys test**

Open `tests/test_i18n_loader.py`. Find `test_ru_json_has_minimum_keys` and add to `expected_keys`:

```python
        "citation.with_page",
        "citation.no_page",
        "citation.text_doc",
        "modal.viewer_title",
        "action.close",
        "viewer.page",
        "viewer.prev",
        "viewer.next",
        "viewer.error.not_available",
        "viewer.error.file_deleted",
        "viewer.error.load_failed",
        "viewer.fallback.text_only",
        "viewer.fallback.scan_no_text",
        "viewer.fallback.page_out_of_range",
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_i18n_loader.py::test_ru_json_has_minimum_keys -v
```
Expected: FAIL with "missing keys: {...}".

- [ ] **Step 3: Add keys to `ru.json`**

Open `data/www/i18n/ru.json`. Add the new keys (keep alphabetical-ish order grouped by prefix):

```json
{
  "citation.with_page": "{filename}, стр. {page}",
  "citation.no_page": "{filename}",
  "citation.text_doc": "{title}",
  "modal.viewer_title": "Просмотр документа",
  "action.close": "Закрыть",
  "viewer.page": "Стр.",
  "viewer.prev": "Предыдущая",
  "viewer.next": "Следующая",
  "viewer.error.not_available": "Оригинал документа недоступен",
  "viewer.error.file_deleted": "Файл удалён на сервере",
  "viewer.error.load_failed": "Не удалось открыть PDF",
  "viewer.fallback.text_only": "Показан только текст фрагмента — оригинальный документ недоступен",
  "viewer.fallback.scan_no_text": "В PDF нет текстового слоя — подсветка фрагмента невозможна, прокрутите вручную",
  "viewer.fallback.page_out_of_range": "Страница {page} вне диапазона документа (всего {total}). Открыта последняя страница."
}
```

**Important:** merge into the existing JSON object, do not replace the file. Preserve all existing keys.

- [ ] **Step 4: Verify JSON is still valid**

Run:
```powershell
py -c "import json; json.load(open('data/www/i18n/ru.json', encoding='utf-8')); print('OK')"
```
Expected: `OK`.

- [ ] **Step 5: Run the test**

Run:
```powershell
py -m pytest tests/test_i18n_loader.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add data/www/i18n/ru.json tests/test_i18n_loader.py
git commit -m @'
feat(i18n): add citation and viewer keys to ru.json

Adds 14 new keys for: citation rendering (with/without page),
modal title and close action, viewer toolbar (page/prev/next),
error states (not available, deleted, load failed), and fallback
banners (text-only, no text layer in scan, page out of range).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task C.5: Add `<dialog>` modal markup + CSS to `index.html`

**Files:**
- Modify: `data/www/index.html`

- [ ] **Step 1: Add CSS for the modal**

Open `data/www/index.html`. Find the closing `</style>` tag in the head. Add these rules **immediately before** `</style>`:

```css
    /* PDF citation viewer modal */
    .kb-modal {
      width: min(95vw, 1100px);
      max-height: 90vh;
      padding: 0;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      background: var(--card);
      color: var(--text);
    }
    .kb-modal::backdrop {
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(2px);
    }
    .kb-modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
    }
    .kb-modal-header h2 {
      margin: 0;
      font-size: 1rem;
    }
    .kb-modal-close {
      background: transparent;
      border: none;
      font-size: 1.25rem;
      cursor: pointer;
      color: var(--muted);
      padding: 0.25rem 0.5rem;
    }
    .kb-modal-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
      color: var(--muted);
    }
    .kb-modal-filename {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text);
    }
    .kb-modal-page-input {
      width: 4rem;
      padding: 0.2rem 0.4rem;
      text-align: center;
    }
    .kb-modal-body {
      position: relative;
      overflow: auto;
      max-height: calc(90vh - 7rem);
      background: var(--bg);
      padding: 0.5rem;
    }
    .kb-modal-loading,
    .kb-modal-error {
      padding: 1.5rem;
      text-align: center;
      color: var(--muted);
    }
    .kb-modal-error { color: var(--danger); }
    .kb-modal-text-fallback {
      padding: 1rem;
      background: var(--card);
      border-radius: 0.5rem;
      white-space: pre-wrap;
      line-height: 1.5;
    }
    .kb-modal-text-fallback .kb-fallback-reason {
      display: block;
      margin-bottom: 0.75rem;
      color: var(--warn);
      font-style: italic;
    }
    .kb-canvas-wrap {
      position: relative;
      margin: 0 auto;
      max-width: max-content;
    }
    .kb-canvas-wrap canvas { display: block; }
    /* PDF.js text layer styles (minimal — find-bar highlights) */
    .kb-text-layer {
      position: absolute;
      inset: 0;
      overflow: hidden;
      opacity: 0.2;
      line-height: 1.0;
      user-select: text;
    }
    .kb-text-layer ::selection { background: rgba(0, 100, 255, 0.4); }
    .kb-text-layer .highlight {
      background: rgba(255, 255, 0, 0.55);
      border-radius: 2px;
    }
    .kb-text-layer .highlight.selected {
      background: rgba(255, 140, 0, 0.7);
    }

    /* Citation button in chat */
    .kb-citation {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.3rem 0.6rem;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid transparent;
      cursor: pointer;
      font-size: 0.85rem;
    }
    .kb-citation:hover { border-color: var(--accent); }
    .kb-citation[data-has-original="false"] { background: var(--bg); color: var(--muted); }
    .kb-citation-icon {
      font-weight: 600;
      font-size: 0.7rem;
      letter-spacing: 0.05em;
    }
    .kb-citation-chevron { font-size: 0.85rem; }
```

- [ ] **Step 2: Add the `<dialog>` markup**

Open `data/www/index.html`. Find the closing `</body>` tag near the end. **Immediately before** `<script src="/i18n/_loader.js"></script>`, add:

```html
  <dialog id="kb-pdf-modal" class="kb-modal" aria-labelledby="kb-modal-title">
    <header class="kb-modal-header">
      <h2 id="kb-modal-title" data-i18n="modal.viewer_title">Просмотр документа</h2>
      <button type="button" class="kb-modal-close"
              data-i18n-attr="aria-label" data-i18n="action.close"
              aria-label="Закрыть">&times;</button>
    </header>
    <div class="kb-modal-toolbar">
      <span class="kb-modal-filename"></span>
      <span>
        <span data-i18n="viewer.page">Стр.</span>
        <input type="number" class="kb-modal-page-input" min="1" value="1" />
        / <span class="kb-modal-page-total">&mdash;</span>
      </span>
      <button type="button" class="kb-modal-prev"
              data-i18n-attr="aria-label" data-i18n="viewer.prev"
              aria-label="Предыдущая">&larr;</button>
      <button type="button" class="kb-modal-next"
              data-i18n-attr="aria-label" data-i18n="viewer.next"
              aria-label="Следующая">&rarr;</button>
    </div>
    <div class="kb-modal-body">
      <div class="kb-modal-loading" data-i18n="status.loading">Загрузка...</div>
      <div class="kb-modal-error" hidden></div>
      <div class="kb-modal-canvas-host"></div>
      <div class="kb-modal-text-fallback" hidden></div>
    </div>
  </dialog>
```

- [ ] **Step 3: Verify HTML still validates and has no obvious errors**

Run:
```powershell
py -c "from html.parser import HTMLParser; HTMLParser().feed(open('data/www/index.html', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`. Crude check but catches gross malformedness.

- [ ] **Step 4: Smoke-test in browser**

Run:
```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001 --reload
```

Open `http://127.0.0.1:8001/` in a browser. The page should look identical to before (the dialog is hidden by default).

In DevTools Console:
```javascript
document.getElementById("kb-pdf-modal").showModal();
```
Expected: modal appears with header "Просмотр документа", toolbar, loading state.

Close with Esc.

Stop the server.

- [ ] **Step 5: Commit**

```powershell
git add data/www/index.html
git commit -m @'
feat(kb-mvp): add PDF viewer modal markup and CSS to index.html

Adds <dialog id="kb-pdf-modal"> with native HTML5 semantics: focus
trap, Esc-close, and backdrop click handled by the browser. Toolbar
has page input, prev/next, and filename slot. Body has loading,
error, canvas host, and text-fallback slots. CSS supports both light
and dark colour schemes via existing var(--*) tokens. Citation
button (.kb-citation) styling added for upcoming chat rendering.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task C.6: Create `data/www/js/pdf-viewer.js` controller

**Files:**
- Create: `data/www/js/pdf-viewer.js`
- Modify: `data/www/index.html` (add `<script src="/js/pdf-viewer.js">`)
- Create: `tests/test_pdf_viewer_js.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_pdf_viewer_js.py`:

```python
"""Smoke-test pdf-viewer.js structure and integration with kb-auth."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "data" / "www" / "js"


def test_pdf_viewer_js_exists():
    assert (JS / "pdf-viewer.js").exists(), "pdf-viewer.js missing"


def test_pdf_viewer_exports_window_namespace():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "window.kbPdfViewer" in content
    assert "openCitation" in content


def test_pdf_viewer_uses_kb_auth():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "window.kbAuth.fetch" in content or "kbAuth.fetch" in content


def test_pdf_viewer_handles_404_410():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "404" in content and "410" in content


def test_pdf_viewer_lazy_imports_pdfjs():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "import(" in content
    assert "/vendor/pdfjs/build/pdf.mjs" in content


def test_pdf_viewer_uses_find_phrase_search():
    content = (JS / "pdf-viewer.js").read_text(encoding="utf-8")
    assert "phraseSearch" in content
    assert "highlightAll" in content


def test_index_html_loads_pdf_viewer_js():
    html = (ROOT / "data" / "www" / "index.html").read_text(encoding="utf-8")
    assert "/js/pdf-viewer.js" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_pdf_viewer_js.py -v
```
Expected: FAIL (file missing).

- [ ] **Step 3: Write `pdf-viewer.js`**

Create `data/www/js/pdf-viewer.js`:

```javascript
/* PDF citation viewer controller.
 *
 * Exposes window.kbPdfViewer.openCitation({docId, page, snippet,
 * hasOriginal, filename, fallbackTitle}). Lazy-imports PDF.js from
 * /vendor/pdfjs on first use. Uses window.kbAuth.fetch for the
 * /api/kb/documents/{id}/file blob request.
 *
 * Highlight strategy: PDF.js text layer + find API with phraseSearch
 * on the first ~30 words of the chunk snippet. Falls back to a plain
 * text view in the modal when the document has no original or the PDF
 * has no text layer.
 */
(function () {
  "use strict";

  const PDFJS_URL = "/vendor/pdfjs/build/pdf.mjs";
  const WORKER_URL = "/vendor/pdfjs/build/pdf.worker.mjs";
  let _pdfjsLib = null;
  let _state = null; // current open viewer state

  async function loadPdfJs() {
    if (_pdfjsLib) return _pdfjsLib;
    const lib = await import(PDFJS_URL);
    lib.GlobalWorkerOptions.workerSrc = WORKER_URL;
    _pdfjsLib = lib;
    return lib;
  }

  function modalEls() {
    const modal = document.getElementById("kb-pdf-modal");
    return {
      modal,
      filename: modal.querySelector(".kb-modal-filename"),
      pageInput: modal.querySelector(".kb-modal-page-input"),
      pageTotal: modal.querySelector(".kb-modal-page-total"),
      prevBtn: modal.querySelector(".kb-modal-prev"),
      nextBtn: modal.querySelector(".kb-modal-next"),
      closeBtn: modal.querySelector(".kb-modal-close"),
      loading: modal.querySelector(".kb-modal-loading"),
      error: modal.querySelector(".kb-modal-error"),
      canvasHost: modal.querySelector(".kb-modal-canvas-host"),
      textFallback: modal.querySelector(".kb-modal-text-fallback"),
    };
  }

  function tr(key, fallback, vars) {
    if (typeof window.t === "function") return window.t(key, fallback, vars);
    return fallback || key;
  }

  function reset(els) {
    els.loading.hidden = false;
    els.error.hidden = true;
    els.error.textContent = "";
    els.canvasHost.innerHTML = "";
    els.textFallback.hidden = true;
    els.textFallback.innerHTML = "";
    els.pageInput.value = "1";
    els.pageTotal.textContent = "—";
  }

  function showError(els, message) {
    els.loading.hidden = true;
    els.error.hidden = false;
    els.error.textContent = message;
  }

  function showTextFallback(els, snippet, reason) {
    els.loading.hidden = true;
    els.canvasHost.innerHTML = "";
    els.textFallback.hidden = false;
    const reasonEl = document.createElement("span");
    reasonEl.className = "kb-fallback-reason";
    reasonEl.textContent = reason;
    const snippetEl = document.createElement("div");
    snippetEl.textContent = snippet || "";
    els.textFallback.replaceChildren(reasonEl, snippetEl);
  }

  async function fetchPdfBlob(docId) {
    const resp = await window.kbAuth.fetch(`/documents/${docId}/file`);
    if (resp.status === 410) {
      throw Object.assign(new Error(tr("viewer.error.file_deleted")), { code: 410 });
    }
    if (resp.status === 404) {
      throw Object.assign(new Error(tr("viewer.error.not_available")), { code: 404 });
    }
    if (resp.status === 401) {
      throw Object.assign(new Error("API key required"), { code: 401 });
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return await resp.arrayBuffer();
  }

  async function renderPage(state, pageNum) {
    const { pdfDoc, els, scale } = state;
    const clamped = Math.max(1, Math.min(pdfDoc.numPages, pageNum | 0));
    state.currentPage = clamped;
    els.pageInput.value = String(clamped);
    els.pageTotal.textContent = String(pdfDoc.numPages);

    const page = await pdfDoc.getPage(clamped);
    const viewport = page.getViewport({ scale });

    const wrap = document.createElement("div");
    wrap.className = "kb-canvas-wrap";

    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.style.width = viewport.width + "px";
    canvas.style.height = viewport.height + "px";
    wrap.appendChild(canvas);

    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "kb-text-layer";
    textLayerDiv.style.width = viewport.width + "px";
    textLayerDiv.style.height = viewport.height + "px";
    wrap.appendChild(textLayerDiv);

    els.canvasHost.innerHTML = "";
    els.canvasHost.appendChild(wrap);
    els.loading.hidden = true;

    const ctx = canvas.getContext("2d");
    await page.render({ canvasContext: ctx, viewport: viewport }).promise;

    // Build text layer for find-highlight to work
    const textContent = await page.getTextContent();
    const pdfjs = state.pdfjsLib;
    if (pdfjs && pdfjs.renderTextLayer) {
      await pdfjs.renderTextLayer({
        textContentSource: textContent,
        container: textLayerDiv,
        viewport: viewport,
      }).promise;
    }
  }

  function buildSearchQuery(snippet) {
    if (!snippet) return "";
    // First 30 whitespace-separated tokens — typically uniquely matches
    // the chunk on the page and survives wrap-around to next page.
    return String(snippet).split(/\s+/).slice(0, 30).join(" ").trim();
  }

  function triggerFind(state) {
    const { pdfjsLib, pdfDoc, snippet } = state;
    const query = buildSearchQuery(snippet);
    if (!query) return;

    // PDFFindController integration — minimal version that highlights
    // matches in the rendered text layer.
    try {
      const eventBus = new pdfjsLib.EventBus();
      const linkService = new pdfjsLib.PDFLinkService({ eventBus: eventBus });
      linkService.setDocument(pdfDoc, null);
      const findController = new pdfjsLib.PDFFindController({
        eventBus: eventBus,
        linkService: linkService,
      });
      findController.setDocument(pdfDoc);
      eventBus.dispatch("find", {
        source: window,
        type: "",
        query: query,
        caseSensitive: false,
        entireWord: false,
        phraseSearch: true,
        highlightAll: true,
        findPrevious: false,
      });
      state.findController = findController;
    } catch (err) {
      console.warn("PDF.js find dispatch failed:", err);
      // Non-fatal — the page is rendered, just no highlight
    }
  }

  function wireToolbar(state) {
    const { els, pdfDoc } = state;
    els.prevBtn.onclick = () => renderPage(state, state.currentPage - 1).then(() => triggerFind(state));
    els.nextBtn.onclick = () => renderPage(state, state.currentPage + 1).then(() => triggerFind(state));
    els.pageInput.onchange = () => {
      const n = parseInt(els.pageInput.value, 10);
      if (Number.isFinite(n)) renderPage(state, n).then(() => triggerFind(state));
    };
    els.closeBtn.onclick = () => els.modal.close();
  }

  async function openCitation(opts) {
    const { docId, page, snippet, hasOriginal, filename, fallbackTitle } = opts || {};
    const els = modalEls();
    reset(els);
    els.filename.textContent = filename || fallbackTitle || `Документ #${docId}`;
    els.modal.showModal();

    if (!hasOriginal) {
      showTextFallback(els, snippet, tr("viewer.fallback.text_only"));
      return;
    }

    try {
      const pdfBytes = await fetchPdfBlob(docId);
      const pdfjsLib = await loadPdfJs();
      const pdfDoc = await pdfjsLib.getDocument({ data: pdfBytes }).promise;

      let initialPage = page || 1;
      if (initialPage > pdfDoc.numPages) {
        showError(els, tr(
          "viewer.fallback.page_out_of_range",
          "Страница вне диапазона",
          { page: initialPage, total: pdfDoc.numPages },
        ));
        initialPage = pdfDoc.numPages;
      }

      _state = {
        pdfDoc: pdfDoc,
        pdfjsLib: pdfjsLib,
        els: els,
        scale: 1.2,
        snippet: snippet,
        currentPage: initialPage,
        findController: null,
      };
      wireToolbar(_state);
      await renderPage(_state, initialPage);
      triggerFind(_state);
    } catch (err) {
      console.error("PDF viewer error:", err);
      showTextFallback(els, snippet, tr("viewer.error.load_failed") + ": " + (err.message || err));
    }
  }

  // Tear-down handler — release blob memory when modal closes
  document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("kb-pdf-modal");
    if (!modal) return;
    modal.addEventListener("close", () => {
      if (_state && _state.pdfDoc && typeof _state.pdfDoc.destroy === "function") {
        try { _state.pdfDoc.destroy(); } catch (_) { /* ignore */ }
      }
      _state = null;
    });
  });

  window.kbPdfViewer = { openCitation: openCitation };
})();
```

- [ ] **Step 4: Add `<script src="/js/pdf-viewer.js">` to `index.html`**

Open `data/www/index.html`. Find:
```html
  <script src="/js/kb-auth.js"></script>
```
Insert **after** it:
```html
  <script src="/js/pdf-viewer.js" defer></script>
```

(`defer` ensures DOMContentLoaded handler inside the file runs once DOM is parsed.)

- [ ] **Step 5: Run the tests**

Run:
```powershell
py -m pytest tests/test_pdf_viewer_js.py -v
```
Expected: PASS (7 tests).

- [ ] **Step 6: Smoke-test in browser**

This step requires a PDF in the DB and is best done with full integration. Skip the live PDF render for now (covered in Task C.7) — just verify:
```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001 --reload
```
Open `http://127.0.0.1:8001/`, open DevTools Console, run:
```javascript
window.kbPdfViewer.openCitation({
  docId: 0,
  page: 1,
  snippet: "test snippet",
  hasOriginal: false,
  filename: "test.pdf"
});
```
Expected: modal opens, shows text-fallback "Показан только текст фрагмента — оригинальный документ недоступен" with `test snippet` below.

Close modal with Esc.

Stop the server.

- [ ] **Step 7: Commit**

```powershell
git add data/www/js/pdf-viewer.js data/www/index.html tests/test_pdf_viewer_js.py
git commit -m @'
feat(kb-mvp): add PDF viewer controller with text-search highlight

data/www/js/pdf-viewer.js exposes window.kbPdfViewer.openCitation.
Lazy-imports PDF.js from /vendor/pdfjs on first use. Uses
window.kbAuth.fetch for the blob endpoint. Renders the requested
page on a canvas plus a text layer so PDFFindController can
highlight matches of the chunk snippet (first 30 words, phraseSearch
on, highlightAll on). Falls back to plain-text view on 404/410 or
when has_original=false.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task C.7: Replace sources rendering in `index.html` with citation buttons

**Files:**
- Modify: `data/www/index.html` (locate and rewrite the chat-message-render block)
- Create: `tests/test_citation_rendering_html.py`

**Background:** The existing chat code renders sources as a list of plain text snippets (the exact location must be located by grep — see Step 1). The replacement renders them as `<button class="kb-citation">` elements with `data-doc-id`, `data-page`, `data-snippet`, `data-has-original`, and click-binding to `window.kbPdfViewer.openCitation`.

- [ ] **Step 1: Locate the existing sources-rendering block**

Run:
```powershell
py -c "import re,sys; t=open('data/www/index.html',encoding='utf-8').read(); [print(i+1,l) for i,l in enumerate(t.splitlines()) if 'source' in l.lower() or 'sources' in l.lower()][:40]"
```

Note line numbers around the `sources` rendering — typically a `forEach` or template-literal that builds list items. Look for `hit.text` or `source.text` usage and adjacent `filename`, `document_title`.

- [ ] **Step 2: Write the structure test**

Create `tests/test_citation_rendering_html.py`:

```python
"""Verify citation buttons are present in chat rendering of index.html."""
from __future__ import annotations

import re
from pathlib import Path


HTML = Path(__file__).resolve().parents[1] / "data" / "www" / "index.html"


def test_citation_template_exists():
    text = HTML.read_text(encoding="utf-8")
    # The render code must produce kb-citation buttons with the expected
    # data attributes
    assert "kb-citation" in text, "no .kb-citation class in index.html"
    assert "data-doc-id" in text, "data-doc-id attribute missing"
    assert "data-page" in text, "data-page attribute missing"
    assert "data-has-original" in text, "data-has-original attribute missing"
    assert "data-snippet" in text, "data-snippet attribute missing"


def test_citation_click_wires_to_pdf_viewer():
    text = HTML.read_text(encoding="utf-8")
    assert "kbPdfViewer.openCitation" in text, "click handler must call openCitation"


def test_citation_uses_i18n_keys_for_label():
    """The label for citation buttons should be generated via t() with
    citation.with_page/citation.no_page/citation.text_doc keys."""
    text = HTML.read_text(encoding="utf-8")
    # Loose check — the keys appear somewhere in the JS
    keys_found = sum(
        1 for k in ("citation.with_page", "citation.no_page", "citation.text_doc")
        if k in text
    )
    assert keys_found >= 1, "citation.* i18n keys not used in index.html"
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_citation_rendering_html.py -v
```
Expected: FAIL (no kb-citation in HTML).

- [ ] **Step 4: Inspect the current source-rendering JS and rewrite it**

Open `data/www/index.html`. Search (Ctrl-F in the editor) for the block that renders `source.text` / `source.filename` / `source.document_title`. Pattern likely looks like:

```javascript
        const sourcesHtml = (sources || []).map((s, i) => `
          <li>
            <strong>[${i+1}] ${s.filename || s.document_title}</strong>
            <div>${s.text}</div>
          </li>
        `).join("");
```

Replace with a function `renderCitations(sources)` and call site:

```javascript
        function renderCitations(sources) {
          return (sources || []).map((s, i) => {
            const hasOriginal = !!s.has_original;
            const page = s.page;
            const filename = s.filename || s.document_title || "";
            let label;
            if (filename && page) {
              label = t("citation.with_page", filename + ", стр. " + page,
                        { filename: filename, page: page });
            } else if (filename) {
              label = t("citation.no_page", filename, { filename: filename });
            } else {
              label = t("citation.text_doc", s.document_title,
                        { title: s.document_title });
            }
            const snippetAttr = String(s.text || "").replace(/"/g, "&quot;");
            const chevron = hasOriginal ? '<span class="kb-citation-chevron" aria-hidden="true">›</span>' : '';
            return `
              <button type="button" class="kb-citation"
                      data-doc-id="${s.document_id}"
                      data-page="${page == null ? "" : page}"
                      data-snippet="${snippetAttr}"
                      data-has-original="${hasOriginal ? "true" : "false"}"
                      data-filename="${(filename || "").replace(/"/g, "&quot;")}"
                      data-title="${(s.document_title || "").replace(/"/g, "&quot;")}">
                <span class="kb-citation-icon" aria-hidden="true">[${i+1}]</span>
                <span class="kb-citation-text">${label}</span>
                ${chevron}
              </button>`;
          }).join("");
        }
```

Then change the call site (originally producing `<li>` list) to use:

```javascript
        const sourcesHtml = renderCitations(sources);
```

And the surrounding markup should be `<div class="kb-citations">${sourcesHtml}</div>` (or whatever container is appropriate — adapt to existing layout).

- [ ] **Step 5: Add delegated click-handler for citation buttons**

In the same `<script>` block, after the existing initialisation (look for `document.addEventListener("DOMContentLoaded", ...)`), append:

```javascript
        document.body.addEventListener("click", (ev) => {
          const btn = ev.target.closest(".kb-citation");
          if (!btn) return;
          ev.preventDefault();
          const docId = parseInt(btn.dataset.docId, 10);
          const pageRaw = btn.dataset.page;
          const page = pageRaw ? parseInt(pageRaw, 10) : null;
          const snippet = btn.dataset.snippet || "";
          const hasOriginal = btn.dataset.hasOriginal === "true";
          const filename = btn.dataset.filename || "";
          const fallbackTitle = btn.dataset.title || "";
          if (window.kbPdfViewer && typeof window.kbPdfViewer.openCitation === "function") {
            window.kbPdfViewer.openCitation({
              docId, page, snippet, hasOriginal, filename, fallbackTitle,
            });
          }
        });
```

- [ ] **Step 6: Smoke-test the citation in a live browser**

Run:
```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001 --reload
```

Open `http://127.0.0.1:8001/`. Manually:

1. Upload a small test PDF via the "Документы" tab (use any 3-5 page PDF)
2. Switch to "Вопрос-ответ" tab and ask a question that should hit the PDF
3. In the response, find the source/citation block — confirm each is now a button with shape `[1] filename.pdf, стр. N ›`
4. Click the button — modal opens, PDF renders on the correct page, snippet is highlighted in yellow

If the modal opens but no highlight appears, check DevTools Network — ensure `pdf.worker.mjs` loaded. If find returned no matches, check that the PDF has a text layer (try a different PDF generated from Word, not a scan).

Stop the server.

- [ ] **Step 7: Run the structure tests**

Run:
```powershell
py -m pytest tests/test_citation_rendering_html.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```powershell
git add data/www/index.html tests/test_citation_rendering_html.py
git commit -m @'
feat(kb-mvp): render chat citations as clickable PDF-viewer buttons

Sources in /ask and /ask/stream responses now render as
<button class="kb-citation"> with data-doc-id / data-page /
data-snippet / data-has-original attributes. A delegated body
click-handler bridges to window.kbPdfViewer.openCitation. Citation
label uses citation.with_page / citation.no_page / citation.text_doc
i18n keys with {filename}/{page} interpolation. Documents without
an original blob (legacy, non-PDF) still render as buttons but
without the chevron — clicking shows the text snippet in the modal
fallback view.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

## Section D — Polish (~2h, 3 tasks)

### Task D.1: Error states + scan-no-text-layer detection

**Files:**
- Modify: `data/www/js/pdf-viewer.js` (extend find-result handler)

- [ ] **Step 1: Add no-text-layer detection**

Open `data/www/js/pdf-viewer.js`. After `renderPage`, check if the text layer is empty:

In `renderPage`, after `await pdfjs.renderTextLayer({...}).promise`, add:

```javascript
    // Detect "scan PDF" — text layer with no extractable spans means
    // find-API has nothing to highlight.
    const spanCount = textLayerDiv.querySelectorAll("span").length;
    state.hasTextLayer = spanCount > 0;
```

In `triggerFind`, before the dispatch, add:

```javascript
    if (state.hasTextLayer === false) {
      const banner = document.createElement("div");
      banner.className = "kb-modal-error";
      banner.style.position = "absolute";
      banner.style.top = "0";
      banner.style.left = "0";
      banner.style.right = "0";
      banner.style.padding = "0.4rem";
      banner.textContent = tr("viewer.fallback.scan_no_text",
        "В PDF нет текстового слоя — подсветка фрагмента невозможна");
      state.els.canvasHost.style.position = "relative";
      // Avoid duplicate banners on repeated page renders
      const existing = state.els.canvasHost.querySelector(".kb-modal-error");
      if (existing) existing.remove();
      state.els.canvasHost.prepend(banner);
      return;
    }
```

- [ ] **Step 2: Manual verification**

Run:
```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001 --reload
```

Upload a scanned PDF (one without text layer) and ask a question on it. Confirm:
- Modal opens
- PDF renders
- Yellow banner near top says "В PDF нет текстового слоя…"

Stop the server. (If you don't have a scan handy, this task can be validated against a text PDF where the banner does **not** appear — that's the negative test.)

- [ ] **Step 3: Commit**

```powershell
git add data/www/js/pdf-viewer.js
git commit -m @'
feat(kb-mvp): show banner when PDF has no text layer

After rendering the page, count text-layer span elements. If zero
(typical for scanned PDFs without OCR), show an inline banner using
viewer.fallback.scan_no_text instead of silently failing the find
dispatch. The PDF page itself remains visible so the user can scroll
to the right place manually.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task D.2: Lazy-load smoke test — verify initial paint stays light

**Files:**
- Create: `tests/test_lazy_pdfjs_load.py`

- [ ] **Step 1: Write the test**

Create `tests/test_lazy_pdfjs_load.py`:

```python
"""Verify the initial HTML payload does NOT include PDF.js inline."""
from __future__ import annotations

from pathlib import Path


HTML = Path(__file__).resolve().parents[1] / "data" / "www" / "index.html"


def test_index_html_does_not_inline_pdfjs():
    text = HTML.read_text(encoding="utf-8")
    # PDF.js is vendored — it should be loaded via dynamic import from
    # pdf-viewer.js, not script-tagged inline.
    assert "/vendor/pdfjs/build/pdf.mjs" not in text, (
        "index.html must not eagerly load PDF.js — must be lazy via "
        "pdf-viewer.js dynamic import"
    )


def test_pdf_viewer_uses_dynamic_import():
    js = Path(__file__).resolve().parents[1] / "data" / "www" / "js" / "pdf-viewer.js"
    text = js.read_text(encoding="utf-8")
    # await import("/vendor/pdfjs/build/pdf.mjs") pattern
    assert 'import("/vendor/pdfjs/build/pdf.mjs")' in text or \
        "import('/vendor/pdfjs/build/pdf.mjs')" in text
```

- [ ] **Step 2: Run the test**

Run:
```powershell
py -m pytest tests/test_lazy_pdfjs_load.py -v
```
Expected: PASS (2 tests). If the first one fails, it means a previous task put a `<script src="/vendor/pdfjs/...">` somewhere it shouldn't be — fix by removing it.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_lazy_pdfjs_load.py
git commit -m @'
test(kb-mvp): assert PDF.js is loaded lazily, not on initial paint

Two checks: index.html must NOT include /vendor/pdfjs in any
<script src> tag; and pdf-viewer.js must use dynamic import() for
the library. Together this enforces the design decision to keep
the main bundle ~80KB until the first citation click triggers
the ~2.5MB PDF.js load.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@
```

---

### Task D.3: Final integration smoke + full test sweep

- [ ] **Step 1: Run the full test suite**

Run:
```powershell
py -m pytest tests/ -v
```
Expected: all PASS (or at least no new failures introduced by this plan; if some pre-existing tests fail for unrelated reasons, document them).

- [ ] **Step 2: Run linter**

Run:
```powershell
py -m ruff check app/services/kb_store.py app/api/kb_mvp.py
py -m black --check app/services/kb_store.py app/api/kb_mvp.py
```
Expected: no errors. Fix any reported issues with `py -m ruff check --fix` and `py -m black`.

- [ ] **Step 3: End-to-end manual smoke checklist**

Start the full stack:
```powershell
py -m uvicorn app.api.main:app --port 8000 --reload
```

Open `http://127.0.0.1:8000/` and run through this checklist:

1. **Upload a multi-page PDF** (5-10 pages, with text content — generated from Word/Pages, not a scan). Use the "Документы" tab.
2. **Verify** the document appears in the list with the filename.
3. **Switch to "Вопрос-ответ"** and ask a question that should match content on page 3 or 5.
4. **Click the citation button** in the response.
5. **Verify modal opens** — title shows filename, page number matches.
6. **Verify highlight** — yellow boxes appear over the chunk text on the page.
7. **Use prev/next page** buttons in the toolbar — navigation works.
8. **Type a page number** in the input — page changes.
9. **Press Esc** — modal closes, focus returns to the citation button.
10. **Delete the blob via FS**:
    ```powershell
    Remove-Item .\var\data\kb_files\*.pdf
    ```
    Reload page, click citation again. Modal opens, shows error "Файл удалён на сервере", text snippet displayed.
11. **Enable auth**:
    ```powershell
    $env:KB_API_KEY = "test-key-12345"
    ```
    Restart server. Reload page. Save key in UI. Repeat steps 3-9 — everything should work because UI adds `X-API-Key` header automatically.

12. **Restart with auth still on, no key in UI** — confirm citation click shows an auth-error in the modal (the fetch returns 401).

13. **DevTools Network** — first page load: pdf.mjs/pdf.worker.mjs NOT in network log. First citation click: pdf.mjs and pdf.worker.mjs appear, marked as `(disk cache)` on second click.

If any step fails, file an issue and address before declaring complete. Otherwise:

- [ ] **Step 4: Cleanup test data**

```powershell
$env:KB_API_KEY = $null
Remove-Item -Force .\var\data\kb_files\* -ErrorAction SilentlyContinue
Remove-Item .\var\data\kb_mvp.sqlite -ErrorAction SilentlyContinue
```

- [ ] **Step 5: No commit needed** — verification only.

---

## Acceptance criteria summary

By the end of this plan:

- ✅ `kb_chunks.page_number`, `kb_documents.has_original_file`, `kb_documents.file_relpath` exist in schema via alembic + store auto-init
- ✅ `add_document(pages=[(page, text)])` chunks per page and stores `page_number`
- ✅ `SearchHit.page` and `.has_original` are populated; `HitOut` exposes them
- ✅ PDF uploads save the raw blob to `var/data/kb_files/<id>.pdf`; non-PDFs do not
- ✅ Orphan tmp blobs are cleaned on parse failure
- ✅ `GET /api/kb/documents/{id}/file` streams the blob with proper auth, path-traversal guard, and 404/410/500 mappings
- ✅ `DELETE /api/kb/documents/{id}` cascade-removes the blob
- ✅ PDF.js v4.x legacy build vendored under `data/www/vendor/pdfjs/`
- ✅ Auth helpers extracted to `data/www/js/kb-auth.js`, consumed by both `index.html` inline JS and `pdf-viewer.js`
- ✅ `_loader.js` supports `{var}` interpolation in `t(key, fallback, vars)`
- ✅ `ru.json` has 14 new keys for citations and viewer
- ✅ `<dialog id="kb-pdf-modal">` added to `index.html` with native focus-trap, Esc-close, backdrop
- ✅ `data/www/js/pdf-viewer.js` exposes `window.kbPdfViewer.openCitation`, lazy-imports PDF.js, renders page on canvas, builds text layer, dispatches find with phraseSearch
- ✅ Chat sources render as `<button class="kb-citation">` with click → `kbPdfViewer.openCitation`
- ✅ Scan PDFs without text layer show a fallback banner instead of silent no-highlight
- ✅ PDF.js NOT inlined on initial paint (verified by `test_lazy_pdfjs_load.py`)
- ✅ Full backend test suite green (`pytest tests/`)
- ✅ Linters green (`ruff check`, `black --check`)
- ✅ End-to-end manual smoke 1-13 pass

Total commits expected: **~16-18** (one per task or sub-step). All atomic, revertible.

Cumulative effort: backend ~10h, frontend ~8h, polish ~2h = **~20h target**, matches `2026-05-22-project-vision-design.md` Phase 1.2 estimate.
