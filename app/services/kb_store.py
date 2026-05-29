"""SQLite-backed document store for the MVP knowledge base.

Provides :class:`KnowledgeBaseStore` with CRUD over ``kb_documents``,
``kb_chunks``, ``kb_conversations`` and ``kb_messages`` plus semantic
search via a pluggable embedder (see :mod:`app.services.kb_embeddings`).
Embedder name and vector dimension are persisted per chunk so vectors
from different models can coexist during a reindex transition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from app.observability import retrieval_health

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100
EMBEDDING_DIM = 256
MAX_TEXT_LEN = 1_000_000
MAX_QUERY_LEN = 2_000
MAX_CONVERSATION_TITLE = 200
MAX_MESSAGE_CONTENT = 50_000
DEFAULT_HISTORY_LIMIT = 10
DEFAULT_SEARCH_HARD_LIMIT = 10_000
HASHING_EMBEDDER_NAME = "hash"


def _search_hard_limit() -> int:
    """Read ``KB_SEARCH_HARD_LIMIT`` with a safe fallback.

    Cap on how many chunks we pull into Python for cosine scoring per
    query. Without this, a corpus with 100K+ chunks would allocate
    hundreds of MB per request — a trivial DoS vector for the MVP.
    """

    raw = os.environ.get("KB_SEARCH_HARD_LIMIT")
    if raw is None:
        return DEFAULT_SEARCH_HARD_LIMIT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SEARCH_HARD_LIMIT
    return max(100, min(value, 1_000_000))


VALID_MESSAGE_ROLES = {"user", "assistant", "system"}


class _EmbedderLike(Protocol):
    name: str
    dimension: int

    def embed(self, text: str) -> List[float]: ...


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


@dataclass(frozen=True)
class Conversation:
    """A logical chat thread with N messages."""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(frozen=True)
class Message:
    """One message inside a conversation."""

    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    sources: List[Mapping[str, Any]] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def embed(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Return a deterministic hashing-based embedding for ``text``.

    Dependency-free fallback used when no remote embedder is configured.
    Real embedders live in :mod:`app.services.kb_embeddings`.
    """

    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dim

    vec = [0.0] * dim
    for token in tokens:
        padded = f"#{token}#"
        if len(padded) < 3:
            buckets: Iterable[str] = (padded,)
        else:
            buckets = (padded[i : i + 3] for i in range(len(padded) - 2))
        for trigram in buckets:
            digest = hashlib.blake2s(trigram.encode("utf-8"), digest_size=4).digest()
            slot = int.from_bytes(digest, "big") % dim
            sign = 1.0 if (digest[-1] & 1) else -1.0
            vec[slot] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split ``text`` into overlapping windows preserving word boundaries."""

    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        segment = cleaned[start:end]
        if end < len(cleaned):
            last_space = segment.rfind(" ")
            if last_space > chunk_size // 2:
                end = start + last_space
                segment = cleaned[start:end]
        trimmed = segment.strip()
        if trimmed:
            chunks.append(trimmed)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _resolve_embedder(explicit: _EmbedderLike | None) -> _EmbedderLike:
    if explicit is not None:
        return explicit
    # Lazy import — kb_embeddings depends on kb_store.embed, so we resolve
    # this here to avoid the circular import at module load.
    from app.services.kb_embeddings import get_embedder

    return get_embedder()


class KnowledgeBaseStore:
    """SQLite-backed store for MVP documents, chunks and embeddings."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        embedder: _EmbedderLike | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embedder = _resolve_embedder(embedder)
        self._init_schema()

    @property
    def embedder(self) -> _EmbedderLike:
        return self._embedder

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # Columns appended to the schema after a DB may already have been created
    # by an older build. ``CREATE TABLE IF NOT EXISTS`` never alters an
    # existing table, so these are reconciled in place by ``_reconcile_columns``
    # (below) before the indexes are built — one index (idx_kb_chunks_doc_page)
    # references ``page_number`` and would raise "no such column" on an old DB.
    # Keep in sync with the CREATE TABLE statements in ``_init_schema`` AND with
    # alembic/versions/20260522_02_pdf_citation.py (the full-stack/Postgres path
    # that the lightweight MVP path deliberately does not run).
    # Each entry is (table, column, column_definition).
    _COLUMN_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
        ("kb_chunks", "page_number", "INTEGER"),
        ("kb_documents", "has_original_file", "INTEGER NOT NULL DEFAULT 0"),
        ("kb_documents", "file_relpath", "TEXT"),
    )

    def _reconcile_columns(self, conn: sqlite3.Connection) -> None:
        """Add columns that post-date a DB created by an older build.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op for a table that already
        exists, so it never backfills columns added to the schema later. For
        each known migration, add the column with ``ALTER TABLE`` when its
        table exists but lacks it. Tables that don't exist yet are skipped —
        ``_init_schema`` creates them fresh with the full column set. Idempotent
        (guarded by a column-presence check), so it is safe on every open.

        Table/column names come from the hardcoded ``_COLUMN_MIGRATIONS``
        constant, never user input; SQLite cannot bind identifiers in DDL, so
        interpolation here is both necessary and safe.
        """
        for table, column, definition in self._COLUMN_MIGRATIONS:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue  # fresh DB — CREATE TABLE below includes the column
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            # Upgrade any pre-existing tables before running the schema script:
            # an index below references page_number, which an old DB lacks until
            # reconciled. On a fresh DB this is a no-op (no tables yet).
            self._reconcile_columns(conn)
            conn.executescript(
                """
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
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_dim ON kb_chunks(dim);
                CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_page ON kb_chunks(document_id, page_number);
                CREATE TABLE IF NOT EXISTS kb_conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES kb_conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    provider TEXT,
                    model TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kb_messages_conv ON kb_messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS kb_feedback (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL
                        REFERENCES kb_conversations(id) ON DELETE CASCADE,
                    message_id INTEGER NOT NULL
                        REFERENCES kb_messages(id) ON DELETE CASCADE,
                    user_id TEXT,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    comment TEXT,
                    alternative_answer TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kb_feedback_message
                    ON kb_feedback(message_id);
                CREATE INDEX IF NOT EXISTS idx_kb_feedback_rating_created
                    ON kb_feedback(rating, created_at);
                """
            )

    @staticmethod
    def _pack(vec: Sequence[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack(blob: bytes) -> List[float]:
        count = len(blob) // 4
        if count == 0:
            return []
        return list(struct.unpack(f"{count}f", blob))

    def _embed_chunks(self, chunks: Sequence[str]) -> tuple[list[bytes], str, int]:
        embedder = self._embedder
        vectors = [embedder.embed(chunk) for chunk in chunks]
        dim = getattr(embedder, "dimension", None) or (
            len(vectors[0]) if vectors else EMBEDDING_DIM
        )
        return [self._pack(vec) for vec in vectors], embedder.name, int(dim)

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
        if text is None and pages is None:
            raise ValueError("Pass either text= or pages=")

        cleaned_title = (title or "").strip() or "Untitled"
        if len(cleaned_title) > 300:
            cleaned_title = cleaned_title[:300]

        # Normalise input to per-page form. Legacy text= goes in as a single
        # virtual page; pages= is used verbatim. Both go through split_text.
        if pages is not None:
            normalised: list[tuple[Optional[int], str]] = []
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
            page_chunks = split_text(page_text, chunk_size=chunk_size, overlap=overlap) or [
                page_text
            ]
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

    def list_documents(self) -> List[Document]:
        with self._connect() as conn:
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
        return [self._row_to_document(row) for row in rows]

    def get_document(self, doc_id: int) -> Optional[Document]:
        with self._connect() as conn:
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
        return self._row_to_document(row) if row is not None else None

    def delete_document(self, doc_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM kb_documents WHERE id = ?", (int(doc_id),))
            return cur.rowcount > 0

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

    def search(self, query: str, *, top_k: int = 5) -> List[SearchHit]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        if len(cleaned) > MAX_QUERY_LEN:
            cleaned = cleaned[:MAX_QUERY_LEN]
        if top_k <= 0:
            return []
        top_k = min(top_k, 50)

        q_vec = self._embedder.embed(cleaned)
        q_dim = len(q_vec)

        hard_limit = _search_hard_limit()
        reasons: list[retrieval_health.RetrievalReason] = []
        detail = ""
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
            if not rows:
                has_chunks = conn.execute("SELECT EXISTS(SELECT 1 FROM kb_chunks)").fetchone()[0]
                if has_chunks:
                    dims = [
                        str(r[0])
                        for r in conn.execute(
                            "SELECT DISTINCT dim FROM kb_chunks LIMIT 3"
                        ).fetchall()
                    ]
                    reasons.append(retrieval_health.RetrievalReason.EMBEDDING_DIM_MISMATCH)
                    detail = f"query dim {q_dim} not in stored dims {{{', '.join(dims)}}}"
        if len(rows) >= hard_limit:
            LOGGER.warning(
                "kb_store.search hit hard limit (%d chunks). Consider Qdrant for large corpora.",
                hard_limit,
            )
            reasons.append(retrieval_health.RetrievalReason.SEARCH_TRUNCATED)
            detail = detail or f"scan capped at {hard_limit} chunks"
        if getattr(self._embedder, "name", None) == HASHING_EMBEDDER_NAME:
            reasons.append(retrieval_health.RetrievalReason.HASHING_EMBEDDER)
            detail = detail or "embedder=hash (near-random semantic matches)"
        retrieval_health.report(
            retrieval_health.RetrievalReport(source="sqlite", reasons=tuple(reasons), detail=detail)
        )

        scored: List[Tuple[float, SearchHit]] = []
        for (
            doc_id,
            title,
            idx,
            text,
            blob,
            _dim,
            source,
            filename,
            page_number,
            has_original,
        ) in rows:
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
        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in scored[:top_k]]

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _new_conversation_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _coerce_title(raw: Optional[str], fallback_seed: str) -> str:
        title = (raw or "").strip()
        if not title:
            seed = (fallback_seed or "").strip().splitlines()
            title = seed[0] if seed else "Новый диалог"
        if len(title) > MAX_CONVERSATION_TITLE:
            title = title[: MAX_CONVERSATION_TITLE - 1] + "…"
        return title or "Новый диалог"

    def create_conversation(
        self,
        title: Optional[str] = None,
        *,
        seed_text: str = "",
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation row and return its dataclass."""

        conv_id = (conversation_id or self._new_conversation_id()).strip()
        if not conv_id:
            conv_id = self._new_conversation_id()
        effective_title = self._coerce_title(title, seed_text)
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO kb_conversations(id, title, created_at, updated_at) VALUES(?, ?, ?, ?)",
                (conv_id, effective_title, now, now),
            )
        return Conversation(id=conv_id, title=effective_title, created_at=now, updated_at=now)

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                    (SELECT COUNT(*) FROM kb_messages m WHERE m.conversation_id = c.id) AS msgs
                FROM kb_conversations c WHERE c.id = ?
                """,
                (str(conv_id),),
            ).fetchone()
        if row is None:
            return None
        return Conversation(
            id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
            message_count=int(row[4] or 0),
        )

    def list_conversations(self, *, limit: int = 100) -> List[Conversation]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                    (SELECT COUNT(*) FROM kb_messages m WHERE m.conversation_id = c.id) AS msgs
                FROM kb_conversations c
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            Conversation(
                id=row[0],
                title=row[1],
                created_at=row[2],
                updated_at=row[3],
                message_count=int(row[4] or 0),
            )
            for row in rows
        ]

    def delete_conversation(self, conv_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM kb_conversations WHERE id = ?", (str(conv_id),))
            return cur.rowcount > 0

    def rename_conversation(self, conv_id: str, title: str) -> Optional[Conversation]:
        cleaned = self._coerce_title(title, fallback_seed="")
        now = self._now()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE kb_conversations SET title = ?, updated_at = ? WHERE id = ?",
                (cleaned, now, str(conv_id)),
            )
            if cur.rowcount == 0:
                return None
        return self.get_conversation(conv_id)

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_sources(sources: Optional[Sequence[Mapping[str, Any]]]) -> Optional[str]:
        if not sources:
            return None
        try:
            return json.dumps(list(sources), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            LOGGER.warning("Dropping non-JSON-serialisable message sources")
            return None

    @staticmethod
    def _load_sources(raw: Optional[str]) -> List[Mapping[str, Any]]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return list(value) if isinstance(value, list) else []

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        sources: Optional[Sequence[Mapping[str, Any]]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Message:
        """Append a message to a conversation, bumping its ``updated_at``."""

        role_clean = (role or "").strip().lower()
        if role_clean not in VALID_MESSAGE_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_MESSAGE_ROLES)}")
        content_clean = (content or "").strip()
        if not content_clean:
            raise ValueError("message content is empty")
        if len(content_clean) > MAX_MESSAGE_CONTENT:
            content_clean = content_clean[:MAX_MESSAGE_CONTENT]

        sources_blob = self._normalise_sources(sources)
        now = self._now()

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM kb_conversations WHERE id = ?",
                (str(conversation_id),),
            ).fetchone()
            if existing is None:
                raise LookupError(f"conversation {conversation_id!r} not found")
            cur = conn.execute(
                """
                INSERT INTO kb_messages(conversation_id, role, content, sources_json, provider, model, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(conversation_id),
                    role_clean,
                    content_clean,
                    sources_blob,
                    provider,
                    model,
                    now,
                ),
            )
            msg_id = int(cur.lastrowid)
            conn.execute(
                "UPDATE kb_conversations SET updated_at = ? WHERE id = ?",
                (now, str(conversation_id)),
            )

        return Message(
            id=msg_id,
            conversation_id=str(conversation_id),
            role=role_clean,
            content=content_clean,
            created_at=now,
            sources=self._load_sources(sources_blob),
            provider=provider,
            model=model,
        )

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Return messages in chronological order (oldest first).

        When ``limit`` is given, returns the most recent ``limit``
        messages but still ordered oldest→newest (so LLM prompts read
        naturally).
        """

        if limit is not None:
            limit = max(1, min(int(limit), 500))

        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT id, conversation_id, role, content, sources_json, provider, model, created_at
                    FROM kb_messages
                    WHERE conversation_id = ?
                    ORDER BY id ASC
                    """,
                    (str(conversation_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, conversation_id, role, content, sources_json, provider, model, created_at
                    FROM kb_messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (str(conversation_id), limit),
                ).fetchall()
                rows = list(reversed(rows))
        return [
            Message(
                id=row[0],
                conversation_id=row[1],
                role=row[2],
                content=row[3],
                sources=self._load_sources(row[4]),
                provider=row[5],
                model=row[6],
                created_at=row[7],
            )
            for row in rows
        ]

    def recent_messages(
        self, conversation_id: str, *, limit: int = DEFAULT_HISTORY_LIMIT
    ) -> List[Message]:
        """Return the most recent ``limit`` messages in chronological order."""

        return self.list_messages(conversation_id, limit=limit)

    # ------------------------------------------------------------------
    # Feedback (W4 — DPO post-training)
    # ------------------------------------------------------------------

    def store_feedback(
        self,
        *,
        conversation_id: str,
        message_id: int,
        user_id: Optional[str],
        rating: int,
        comment: Optional[str],
        alternative_answer: Optional[str],
    ) -> str:
        """Persist one feedback row; returns a new UUID id.

        Raises :class:`ValueError` for out-of-range ``rating`` before the
        DB CHECK constraint catches it, so the API layer can map it to
        HTTP 400 without parsing sqlite3 error messages.
        """

        import uuid

        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or 1, got {rating}")

        fid = uuid.uuid4().hex
        now = self._now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kb_feedback (
                    id, conversation_id, message_id, user_id,
                    rating, comment, alternative_answer, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fid,
                    conversation_id,
                    int(message_id),
                    user_id,
                    int(rating),
                    comment,
                    alternative_answer,
                    now,
                ),
            )
        return fid

    def iter_feedback_pairs(self):
        """Yield :class:`app.services.dpo_dataset.DPOPair` from live feedback.

        Pairing rules (W4 spec § 6.3):
          * Most-recent rating per (message_id, user_id) wins.
          * thumbs-down + alternative_answer → emit (alt as chosen, assistant as rejected).
          * thumbs-up + alternative_answer → same shape (alt is the user-provided gold).
          * thumbs-up alone → look back for a same-message thumbs-down with alt.
          * Anything else → skip silently.
          * Orphaned assistant messages (no preceding user) → skip with debug log.
        """

        from app.services.dpo_dataset import DPOPair, RejectStrategy

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.conversation_id, f.message_id, f.user_id,
                       f.rating, f.alternative_answer, f.created_at,
                       m.content AS assistant_content
                FROM kb_feedback f
                JOIN kb_messages m ON m.id = f.message_id
                ORDER BY f.message_id, f.user_id, f.created_at DESC
                """
            ).fetchall()

            groups: dict[tuple[int, Optional[str]], list] = {}
            for row in rows:
                key = (int(row[2]), row[3])
                groups.setdefault(key, []).append(row)

            for (msg_id, _user_id), group in groups.items():
                latest = group[0]
                # latest indices: 0=id, 1=conversation_id, 2=message_id, 3=user_id,
                # 4=rating, 5=alternative_answer, 6=created_at, 7=assistant_content
                preceding = conn.execute(
                    """
                    SELECT content FROM kb_messages
                    WHERE conversation_id = ? AND role = 'user' AND id < ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (latest[1], msg_id),
                ).fetchone()
                if preceding is None:
                    LOGGER.debug(
                        "Skipping feedback %s — no preceding user message",
                        latest[0],
                    )
                    continue
                user_prompt = preceding[0]
                assistant_text = latest[7]
                rating = latest[4]
                alt = latest[5]

                if rating == -1 and alt:
                    yield DPOPair(
                        prompt=user_prompt,
                        chosen=alt,
                        rejected=assistant_text,
                        strategy=RejectStrategy.LIVE_ALT,
                        source="live",
                        source_chunk_id=None,
                        feedback_ids=(latest[0],),
                    )
                elif rating == 1:
                    if alt:
                        yield DPOPair(
                            prompt=user_prompt,
                            chosen=alt,
                            rejected=assistant_text,
                            strategy=RejectStrategy.LIVE_ALT,
                            source="live",
                            source_chunk_id=None,
                            feedback_ids=(latest[0],),
                        )
                        continue
                    downvote = next(
                        (r for r in group[1:] if r[4] == -1 and r[5]),
                        None,
                    )
                    if downvote:
                        yield DPOPair(
                            prompt=user_prompt,
                            chosen=assistant_text,
                            rejected=downvote[5],
                            strategy=RejectStrategy.LIVE_PAIRED,
                            source="live",
                            source_chunk_id=None,
                            feedback_ids=(latest[0], downvote[0]),
                        )


_DEFAULT_STORE: Optional[KnowledgeBaseStore] = None
_STORE_LOCK = threading.Lock()


def _default_db_path() -> str:
    explicit = os.environ.get("KB_MVP_DB_PATH")
    if explicit:
        return explicit
    base = os.environ.get("DATA_DIR", "./var/data")
    return str(Path(base) / "kb_mvp.sqlite")


def get_store() -> KnowledgeBaseStore:
    """Return the cached default store, creating it on first call."""

    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        return _DEFAULT_STORE
    with _STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = KnowledgeBaseStore(_default_db_path())
        return _DEFAULT_STORE


def reset_default_store() -> None:
    """Drop the cached default store (used in tests)."""

    global _DEFAULT_STORE
    with _STORE_LOCK:
        _DEFAULT_STORE = None


__all__ = [
    "Conversation",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_OVERLAP",
    "Document",
    "EMBEDDING_DIM",
    "HASHING_EMBEDDER_NAME",
    "KnowledgeBaseStore",
    "MAX_CONVERSATION_TITLE",
    "MAX_MESSAGE_CONTENT",
    "MAX_QUERY_LEN",
    "MAX_TEXT_LEN",
    "Message",
    "SearchHit",
    "VALID_MESSAGE_ROLES",
    "embed",
    "get_store",
    "reset_default_store",
    "split_text",
]
