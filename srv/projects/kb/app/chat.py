"""Chat conversation storage and summarisation utilities."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import time
import uuid
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = ["ChatStore", "ConversationAccessError", "ConversationSummarizer"]


class ConversationAccessError(RuntimeError):
    """Raised when an operation targets a conversation owned by another user."""


class ChatStore:
    """Persist chat conversations and summaries in SQLite."""

    def __init__(self, db_path: str, *, secret: str | None = None) -> None:
        self.db_path = db_path
        self._secret = secret.encode("utf-8") if secret else None
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT,
                    messages_since_summary INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, created_at)"
            )

    def _generate_conversation_id(self, user_id: str) -> str:
        seed = f"{user_id}:{time.time_ns()}:{uuid.uuid4().hex}".encode("utf-8")
        if self._secret:
            digest = hmac.new(self._secret, seed, hashlib.sha256).hexdigest()
            return digest
        return uuid.uuid4().hex

    def ensure_conversation(self, user_id: str, conversation_id: Optional[str]) -> str:
        """Return *conversation_id* for the given user, creating a new one when needed."""

        conv_id = conversation_id or self._generate_conversation_id(user_id)
        timestamp = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO conversations(id, user_id, summary, messages_since_summary, created_at)
                    VALUES (?, ?, NULL, 0, ?)
                    """,
                    (conv_id, user_id, timestamp),
                )
            elif row["user_id"] != user_id:
                raise ConversationAccessError("conversation belongs to a different user")
        return conv_id

    def get_summary(self, conversation_id: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return row["summary"] if row else None

    def get_recent_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        query = (
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at DESC, id DESC"
        )
        params: Tuple[object, ...]
        params = (conversation_id,)
        with self._connect() as connection:
            cursor = connection.execute(query, params)
            rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
        ordered = list(rows)[::-1]
        return [(row["role"], row["content"]) for row in ordered]

    def record_exchange(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        timestamp = int(time.time())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(conversation_id, role, content, created_at)
                VALUES (?, 'user', ?, ?)
                """,
                (conversation_id, user_message, timestamp),
            )
            connection.execute(
                """
                INSERT INTO messages(conversation_id, role, content, created_at)
                VALUES (?, 'assistant', ?, ?)
                """,
                (conversation_id, assistant_message, timestamp),
            )
            connection.execute(
                """
                UPDATE conversations
                SET messages_since_summary = messages_since_summary + 2
                WHERE id = ?
                """,
                (conversation_id,),
            )

    def messages_since_summary(self, conversation_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT messages_since_summary FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return int(row["messages_since_summary"]) if row else 0

    def save_summary(self, conversation_id: str, summary: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET summary = ?, messages_since_summary = 0
                WHERE id = ?
                """,
                (summary, conversation_id),
            )


class ConversationSummarizer:
    """Summarise chat conversations using a language model."""

    def __init__(
        self,
        store: ChatStore,
        llm_generate: Callable[[str], str],
        max_history: int = 50,
    ) -> None:
        self.store = store
        self._llm_generate = llm_generate
        self.max_history = max_history

    def _format_history(self, history: Sequence[Tuple[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{role}: {content}" for role, content in history]
        return "\n".join(lines)

    def summarize(self, conversation_id: str) -> str | None:
        """Generate and persist a short summary for *conversation_id*."""

        history = self.store.get_recent_messages(conversation_id, limit=self.max_history)
        if not history:
            return None
        current_summary = self.store.get_summary(conversation_id) or ""

        prompt_parts = [
            "Суммаризируй приведённый ниже диалог на русском языке.",
            "Выдели ключевые решения и контекст для следующих сообщений.",
        ]
        if current_summary:
            prompt_parts.append(f"Текущее саммари: {current_summary}")
        prompt_parts.append("Диалог:")
        prompt_parts.append(self._format_history(history))
        prompt_parts.append("Новая краткая выжимка:")
        prompt = "\n\n".join(part for part in prompt_parts if part)

        try:
            summary = self._llm_generate(prompt).strip()
        except Exception:  # pragma: no cover - defensive logging
            return None

        if not summary:
            return None

        self.store.save_summary(conversation_id, summary)
        return summary
