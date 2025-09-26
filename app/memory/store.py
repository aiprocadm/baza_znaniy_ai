import os
import sqlite3
import time
from typing import Optional

import psycopg


def _pg_conninfo() -> Optional[str]:
    host = os.getenv("PGHOST")
    db = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    if not host or not db or not user:
        return None
    port = os.getenv("PGPORT", "5432")
    password = os.getenv("PGPASSWORD")
    params = [f"host={host}", f"port={port}", f"dbname={db}", f"user={user}"]
    if password:
        params.append(f"password={password}")
    return " ".join(params)


class MemoryStore:
    def __init__(self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int):
        self.db_path = db_path
        self.ttl = ttl_days * 86400
        self.trigger = summary_trigger
        self.max_tokens = max_tokens
        self._pg_conninfo = _pg_conninfo()
        if self._pg_conninfo:
            self._init_postgres()
        else:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as c:
                c.execute(
                    """CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY,
                user_id TEXT, conv_id TEXT, role TEXT, content TEXT, ts INTEGER
            )"""
                )
                c.execute("CREATE INDEX IF NOT EXISTS ix_messages_user_conv ON messages(user_id, conv_id)")
                c.commit()

    def _init_postgres(self) -> None:
        with psycopg.connect(self._pg_conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
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
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_messages_user_conv ON messages(user_id, conv_id)"
                )
            conn.commit()

    def record(self, user_id: str, conv_id: str | None, msg: str, ans: str):
        conv_id = conv_id or "default"
        now = int(time.time())
        if self._pg_conninfo:
            with psycopg.connect(self._pg_conninfo, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(%s,%s,%s,%s,%s)",
                        (user_id, conv_id, "user", msg, now),
                    )
                    cur.execute(
                        "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(%s,%s,%s,%s,%s)",
                        (user_id, conv_id, "assistant", ans, now),
                    )
            return

        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(?,?,?,?,?)",
                (user_id, conv_id, "user", msg, now),
            )
            c.execute(
                "INSERT INTO messages(user_id,conv_id,role,content,ts) VALUES(?,?,?,?,?)",
                (user_id, conv_id, "assistant", ans, now),
            )
            c.commit()

    def load_context(self, user_id: str, conv_id: str | None):
        conv_id = conv_id or "default"
        cutoff = int(time.time()) - self.ttl
        if self._pg_conninfo:
            with psycopg.connect(self._pg_conninfo) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, content FROM messages
                        WHERE user_id=%s AND conv_id=%s AND ts>=%s
                        ORDER BY id DESC LIMIT 10
                        """,
                        (user_id, conv_id, cutoff),
                    )
                    rows = cur.fetchall()
        else:
            with sqlite3.connect(self.db_path) as c:
                rows = c.execute(
                    """SELECT role,content FROM messages
                                WHERE user_id=? AND conv_id=? AND ts>=?
                                ORDER BY id DESC LIMIT 10""",
                    (user_id, conv_id, cutoff),
                ).fetchall()
        rows = rows[::-1]
        text = "\n".join(f"{r[0]}: {r[1]}" for r in rows)
        return text[: self.max_tokens * 2]
