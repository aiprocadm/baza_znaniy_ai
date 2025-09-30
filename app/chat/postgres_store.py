"""PostgreSQL-backed persistence for chat conversations."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .store import ChatStoreProtocol, ConversationAccessError

__all__ = ["PostgresChatStore"]


class PostgresChatStore(ChatStoreProtocol):
    """Persist chat conversations and summaries using PostgreSQL."""

    def __init__(self, dsn: str, *, schema: str | None = None) -> None:
        self.dsn = dsn
        self.schema = (schema or "").strip() or None
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        connection = psycopg.connect(self.dsn, row_factory=dict_row)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _qualified(self, name: str) -> sql.Identifier:
        if self.schema:
            return sql.Identifier(self.schema, name)
        return sql.Identifier(name)

    def _index_identifier(self, name: str) -> sql.Identifier:
        if self.schema:
            return sql.Identifier(self.schema, name)
        return sql.Identifier(name)

    def _init_schema(self) -> None:
        # Ensure schema exists before creating tables.
        if self.schema:
            with psycopg.connect(self.dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}")
                        .format(sql.Identifier(self.schema))
                    )
                connection.commit()

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            summary TEXT,
                            messages_since_summary INTEGER NOT NULL DEFAULT 0,
                            created_at BIGINT NOT NULL
                        )
                        """
                    ).format(self._qualified("conversations"))
                )
                cursor.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id BIGSERIAL PRIMARY KEY,
                            conversation_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            created_at BIGINT NOT NULL
                        )
                        """
                    ).format(
                        self._qualified("messages"),
                        self._qualified("conversations"),
                    )
                )
                cursor.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {}(conversation_id, created_at)"
                    ).format(
                        self._index_identifier("ix_messages_conversation"),
                        self._qualified("messages"),
                    )
                )

    def ensure_conversation(self, user_id: str, conversation_id: Optional[str]) -> str:
        conv_id = conversation_id or uuid.uuid4().hex
        timestamp = int(time.time())
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SELECT user_id FROM {} WHERE id = %s").format(
                        self._qualified("conversations")
                    ),
                    (conv_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}(id, user_id, summary, messages_since_summary, created_at)
                            VALUES (%s, %s, NULL, 0, %s)
                            """
                        ).format(self._qualified("conversations")),
                        (conv_id, user_id, timestamp),
                    )
                elif row["user_id"] != user_id:
                    raise ConversationAccessError("conversation belongs to a different user")
        return conv_id

    def get_summary(self, conversation_id: str) -> Optional[str]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SELECT summary FROM {} WHERE id = %s").format(
                        self._qualified("conversations")
                    ),
                    (conversation_id,),
                )
                row = cursor.fetchone()
        return row["summary"] if row else None

    def get_recent_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        base_query = sql.SQL(
            "SELECT role, content FROM {} WHERE conversation_id = %s "
            "ORDER BY created_at DESC, id DESC"
        ).format(self._qualified("messages"))
        params: List[object] = [conversation_id]
        if limit is not None:
            query = base_query + sql.SQL(" LIMIT %s")
            params.append(limit)
        else:
            query = base_query

        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

        ordered = list(rows)[::-1]
        return [(row["role"], row["content"]) for row in ordered]

    def record_exchange(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        timestamp = int(time.time())
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}(conversation_id, role, content, created_at)
                        VALUES (%s, 'user', %s, %s)
                        """
                    ).format(self._qualified("messages")),
                    (conversation_id, user_message, timestamp),
                )
                cursor.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}(conversation_id, role, content, created_at)
                        VALUES (%s, 'assistant', %s, %s)
                        """
                    ).format(self._qualified("messages")),
                    (conversation_id, assistant_message, timestamp),
                )
                cursor.execute(
                    sql.SQL(
                        """
                        UPDATE {}
                        SET messages_since_summary = messages_since_summary + 2
                        WHERE id = %s
                        """
                    ).format(self._qualified("conversations")),
                    (conversation_id,),
                )

    def messages_since_summary(self, conversation_id: str) -> int:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT messages_since_summary FROM {} WHERE id = %s"
                    ).format(self._qualified("conversations")),
                    (conversation_id,),
                )
                row = cursor.fetchone()
        return int(row["messages_since_summary"]) if row else 0

    def save_summary(self, conversation_id: str, summary: str) -> None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        UPDATE {}
                        SET summary = %s, messages_since_summary = 0
                        WHERE id = %s
                        """
                    ).format(self._qualified("conversations")),
                    (summary, conversation_id),
                )
