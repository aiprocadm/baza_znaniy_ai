"""Persistence layer for chat conversation memory using SQLite only."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional


class MemoryStore:
    """Persist chat conversations in a lightweight SQLite database."""

    def __init__(self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int) -> None:
        self.db_path = db_path
        self.ttl = ttl_days * 86400
        self.trigger = summary_trigger
        self.max_tokens = max_tokens
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        if self.db_path != ":memory:":
            directory = Path(self.db_path).parent
            if not directory.parts:
                directory = Path(".")
            directory.mkdir(parents=True, exist_ok=True)
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

    @contextmanager
    def _sqlite_connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        try:
            yield connection
        finally:
            connection.close()

    def record(self, user_id: str, conv_id: Optional[str], message: str, answer: str) -> None:
        conv = conv_id or "default"
        timestamp = int(time.time())
        with self._sqlite_connection() as connection:
            connection.execute(
                "INSERT INTO messages(user_id, conv_id, role, content, ts) VALUES(?,?,?,?,?)",
                (user_id, conv, "user", message, timestamp),
            )
            connection.execute(
                "INSERT INTO messages(user_id, conv_id, role, content, ts) VALUES(?,?,?,?,?)",
                (user_id, conv, "assistant", answer, timestamp),
            )
            connection.commit()

    def load_context(self, user_id: str, conv_id: Optional[str]) -> str:
        conv = conv_id or "default"
        cutoff = int(time.time()) - self.ttl

        with self._sqlite_connection() as connection:
            rows: Iterable[tuple[str, str]] = connection.execute(
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
