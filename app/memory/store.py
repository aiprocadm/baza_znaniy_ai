import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterable

try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None

class MemoryStore:
    def __init__(self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int):
        self.ttl = ttl_days * 86400
        self.trigger = summary_trigger
        self.max_tokens = max_tokens
        self.db_url = os.getenv("DATABASE_URL")
        self.db_path = db_path

        if self.db_url:
            if psycopg is None:
                raise RuntimeError("psycopg is required when DATABASE_URL is set")
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY,
                user_id TEXT, conv_id TEXT, role TEXT, content TEXT, ts INTEGER
            )"""
            )
            c.commit()

    def _init_postgres(self) -> None:
        with psycopg.connect(self.db_url, autocommit=True) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS messages(
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                conv_id TEXT,
                role TEXT,
                content TEXT,
                ts BIGINT
            )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_lookup ON messages(user_id, conv_id, ts)"
            )

    @contextmanager
    def _sqlite_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _postgres_cursor(self):
        assert self.db_url and psycopg is not None
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                yield cur

    def record(self, user_id: str, conv_id: str|None, msg: str, ans: str):
        conv_id = conv_id or "default"
        now = int(time.time())
        if self.db_url:
            with psycopg.connect(self.db_url, autocommit=True) as conn:
                conn.execute(
                    "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(%s,%s,%s,%s,%s)",
                    (user_id, conv_id, "user", msg, now),
                )
                conn.execute(
                    "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(%s,%s,%s,%s,%s)",
                    (user_id, conv_id, "assistant", ans, now),
                )
            return

        with self._sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(?,?,?,?,?)",
                (user_id, conv_id, "user", msg, now),
            )
            conn.execute(
                "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(?,?,?,?,?)",
                (user_id, conv_id, "assistant", ans, now),
            )
            conn.commit()

    def load_context(self, user_id: str, conv_id: str|None):
        conv_id = conv_id or "default"
        cutoff = int(time.time()) - self.ttl
        rows: Iterable[tuple[str, str]]
        if self.db_url:
            with self._postgres_cursor() as cur:
                cur.execute(
                    """SELECT role, content FROM messages
                            WHERE user_id=%s AND conv_id=%s AND ts>=%s
                            ORDER BY id DESC LIMIT 10""",
                    (user_id, conv_id, cutoff),
                )
                rows = cur.fetchall()
        else:
            with self._sqlite_conn() as conn:
                rows = conn.execute(
                    """SELECT role,content FROM messages
                                    WHERE user_id=? AND conv_id=? AND ts>=?
                                    ORDER BY id DESC LIMIT 10""",
                    (user_id, conv_id, cutoff),
                ).fetchall()

        rows = list(rows)[::-1]
        text = "\n".join(f"{r}: {t}" for r, t in rows)
        return text[:self.max_tokens*2]
