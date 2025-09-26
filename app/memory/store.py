from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import psycopg
except ImportError:  # pragma: no cover - tests run without postgres
    psycopg = None


def _pg_conninfo() -> Optional[str]:
    host = os.getenv("PGHOST")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    if not host or not database or not user:
        return None
    port = os.getenv("PGPORT", "5432")
    password = os.getenv("PGPASSWORD")
    parts = [f"host={host}", f"port={port}", f"dbname={database}", f"user={user}"]
    if password:
        parts.append(f"password={password}")
    return " ".join(parts)


class MemoryStore:
    """Persist chat conversations either in SQLite or PostgreSQL."""

    def __init__(self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int) -> None:
        self.db_path = db_path
        self.ttl = ttl_days * 86400
        self.trigger = summary_trigger
        self.max_tokens = max_tokens
        self._pg_conninfo = _pg_conninfo()

        if self._pg_conninfo:
            if psycopg is None:
                raise RuntimeError("psycopg is required when PostgreSQL is configured")
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    conv_id TEXT,
                    role TEXT,
                    content TEXT,
                    ts INTEGER
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_messages_lookup ON messages(user_id, conv_id, ts)"
            )
            connection.commit()

    def _init_postgres(self) -> None:
        assert self._pg_conninfo and psycopg is not None
        with psycopg.connect(self._pg_conninfo, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT,
                        conv_id TEXT,
                        role TEXT,
                        content TEXT,
                        ts BIGINT
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS ix_messages_lookup ON messages(user_id, conv_id, ts)"
                )

    @contextmanager
    def _sqlite_conn(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def record(self, user_id: str, conv_id: str | None, msg: str, ans: str) -> None:
        conv = conv_id or "default"
        timestamp = int(time.time())
        if self._pg_conninfo and psycopg is not None:
            with psycopg.connect(self._pg_conninfo, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO messages(user_id, conv_id, role, content, ts) VALUES(%s, %s, %s, %s, %s)",
                        (user_id, conv, "user", msg, timestamp),
                    )
                    cursor.execute(
                        "INSERT INTO messages(user_id, conv_id, role, content, ts) VALUES(%s, %s, %s, %s, %s)",
                        (user_id, conv, "assistant", ans, timestamp),
                    )
            return

        with self._sqlite_conn() as connection:
            connection.execute(
                "INSERT INTO messages(user_id, conv_id, role, content, ts) VALUES(?,?,?,?,?)",
                (user_id, conv, "user", msg, timestamp),
            )
            connection.execute(
                "INSERT INTO messages(user_id, conv_id, role, content, ts) VALUES(?,?,?,?,?)",
                (user_id, conv, "assistant", ans, timestamp),
            )
            connection.commit()

    def load_context(self, user_id: str, conv_id: str | None) -> str:
        conv = conv_id or "default"
        cutoff = int(time.time()) - self.ttl

        rows: Sequence[tuple[str, str]]
        if self._pg_conninfo and psycopg is not None:
            with psycopg.connect(self._pg_conninfo) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT role, content FROM messages
                        WHERE user_id=%s AND conv_id=%s AND ts>=%s
                        ORDER BY id DESC LIMIT 10
                        """,
                        (user_id, conv, cutoff),
                    )
                    rows = cursor.fetchall()
        else:
            with sqlite3.connect(self.db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT role, content FROM messages
                    WHERE user_id=? AND conv_id=? AND ts>=?
                    ORDER BY id DESC LIMIT 10
                    """,
                    (user_id, conv, cutoff),
                ).fetchall()

        ordered = list(rows)[::-1]
        transcript = "\n".join(f"{role}: {content}" for role, content in ordered)
        return transcript[: self.max_tokens * 2]


__all__ = ["MemoryStore"]
