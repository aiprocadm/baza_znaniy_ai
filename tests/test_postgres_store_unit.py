"""Unit tests for PostgresChatStore that avoid real database connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import pytest
from psycopg import sql

from app.chat import postgres_store
from app.chat.postgres_store import PostgresChatStore


class FakeCursor:
    """A lightweight cursor that records executed SQL statements."""

    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.fetchone_result: Optional[Dict[str, Any]] = None
        self.fetchall_result: Iterable[Dict[str, Any]] = ()

    def execute(
        self, query: Any, params: Any = None
    ) -> None:  # noqa: ANN401 - matches psycopg signature
        self.connection.log.append(("execute", query, params))

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self.fetchone_result

    def fetchall(self) -> List[Dict[str, Any]]:
        return list(self.fetchall_result)

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN201 - test double
        return False


class FakeConnection:
    """A connection object that mimics psycopg behaviour for tests."""

    def __init__(
        self, cursor_factory: Optional[Callable[["FakeConnection"], FakeCursor]] = None
    ) -> None:
        self.cursor_factory = cursor_factory or (lambda conn: FakeCursor(conn))
        self.log: List[tuple[Any, Any, Any]] = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0
        self.cursors: List[FakeCursor] = []
        self.dsn: Optional[str] = None
        self.row_factory: Any = None

    def cursor(self) -> FakeCursor:
        cursor = self.cursor_factory(self)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN201 - test double
        self.close()
        return False


@dataclass
class ConnectCall:
    dsn: str
    row_factory: Any
    connection: FakeConnection


class FakeConnect:
    """Callable that returns fake connections in sequence."""

    def __init__(self, connections: Iterable[FakeConnection]) -> None:
        self._queue = list(connections)
        self.calls: List[ConnectCall] = []

    def __call__(self, dsn: str, row_factory: Any = None) -> FakeConnection:
        if not self._queue:
            raise AssertionError("No fake connections left to return")
        connection = self._queue.pop(0)
        connection.dsn = dsn
        connection.row_factory = row_factory
        self.calls.append(ConnectCall(dsn, row_factory, connection))
        return connection


def test_init_calls_init_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[PostgresChatStore] = []

    def fake_init_schema(self: PostgresChatStore) -> None:
        calls.append(self)

    monkeypatch.setattr(PostgresChatStore, "_init_schema", fake_init_schema)
    store = PostgresChatStore("postgres://test")
    assert calls == [store]


def test_init_with_schema_creates_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    schema_conn = FakeConnection()
    tables_conn = FakeConnection()
    fake_connect = FakeConnect([schema_conn, tables_conn])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test", schema="chat_schema")
    assert store.schema == "chat_schema"

    assert [call.row_factory for call in fake_connect.calls] == [None, postgres_store.dict_row]

    assert schema_conn.commit_calls == 1
    assert schema_conn.close_calls == 1
    assert len(schema_conn.log) == 1
    schema_query = schema_conn.log[0][1]
    assert isinstance(schema_query, sql.Composed)
    schema_identifier = schema_query._obj[1]
    assert isinstance(schema_identifier, sql.Identifier)
    assert schema_identifier._obj == ("chat_schema",)

    assert tables_conn.commit_calls == 1
    assert tables_conn.close_calls == 1
    executed = [entry for entry in tables_conn.log if entry[0] == "execute"]
    assert len(executed) == 3

    conversations_identifier = executed[0][1]._obj[1]
    assert isinstance(conversations_identifier, sql.Identifier)
    assert conversations_identifier._obj == ("chat_schema", "conversations")

    messages_query = executed[1][1]
    message_table_identifier = messages_query._obj[1]
    referenced_table_identifier = messages_query._obj[3]
    assert isinstance(message_table_identifier, sql.Identifier)
    assert message_table_identifier._obj == ("chat_schema", "messages")
    assert isinstance(referenced_table_identifier, sql.Identifier)
    assert referenced_table_identifier._obj == ("chat_schema", "conversations")

    index_query = executed[2][1]
    index_name_identifier = index_query._obj[1]
    indexed_table_identifier = index_query._obj[3]
    assert isinstance(index_name_identifier, sql.Identifier)
    assert index_name_identifier._obj == ("chat_schema", "ix_messages_conversation")
    assert isinstance(indexed_table_identifier, sql.Identifier)
    assert indexed_table_identifier._obj == ("chat_schema", "messages")


def test_connection_commits_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)
    connection = FakeConnection()
    fake_connect = FakeConnect([connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test")

    with store._connection() as _conn:
        assert _conn is connection

    assert connection.commit_calls == 1
    assert connection.rollback_calls == 0
    assert connection.close_calls == 1


def test_connection_rolls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)
    connection = FakeConnection()
    fake_connect = FakeConnect([connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test")

    with pytest.raises(RuntimeError):
        with store._connection() as _conn:
            assert _conn is connection
            raise RuntimeError("boom")

    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1
    assert connection.close_calls == 1


def test_ensure_conversation_inserts_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)
    connection = FakeConnection()
    fake_connect = FakeConnect([connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)
    monkeypatch.setattr(postgres_store.time, "time", lambda: 42)

    store = PostgresChatStore("postgres://test", schema="chat")

    conv_id = store.ensure_conversation("user-1", "conv-1")
    assert conv_id == "conv-1"

    executed = [entry for entry in connection.log if entry[0] == "execute"]
    assert len(executed) == 2

    select_query, insert_query = executed
    assert select_query[2] == ("conv-1",)
    select_identifier = select_query[1]._obj[1]
    assert isinstance(select_identifier, sql.Identifier)
    assert select_identifier._obj == ("chat", "conversations")

    params = insert_query[2]
    assert params[0] == "conv-1"
    assert params[1] == "user-1"
    assert params[2] == 42

    insert_identifier = insert_query[1]._obj[1]
    assert isinstance(insert_identifier, sql.Identifier)
    assert insert_identifier._obj == ("chat", "conversations")

    assert connection.commit_calls == 1
    assert connection.close_calls == 1


def test_get_summary_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)

    def cursor_factory(connection: FakeConnection) -> FakeCursor:
        cursor = FakeCursor(connection)
        cursor.fetchone_result = {"summary": "cached summary"}
        return cursor

    connection = FakeConnection(cursor_factory)
    fake_connect = FakeConnect([connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test")
    result = store.get_summary("conv-123")

    assert result == "cached summary"

    executed = [entry for entry in connection.log if entry[0] == "execute"]
    assert len(executed) == 1
    assert executed[0][2] == ("conv-123",)


def test_get_recent_messages_orders_oldest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)

    def cursor_factory(connection: FakeConnection) -> FakeCursor:
        cursor = FakeCursor(connection)
        cursor.fetchall_result = [
            {"role": "assistant", "content": "third"},
            {"role": "user", "content": "second"},
            {"role": "system", "content": "first"},
        ]
        return cursor

    connection = FakeConnection(cursor_factory)
    fake_connect = FakeConnect([connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test")
    messages = store.get_recent_messages("conv-xyz")

    assert messages == [
        ("system", "first"),
        ("user", "second"),
        ("assistant", "third"),
    ]

    executed = [entry for entry in connection.log if entry[0] == "execute"]
    assert len(executed) == 1
    assert executed[0][2] == ["conv-xyz"]


def test_record_exchange_updates_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)
    monkeypatch.setattr(postgres_store.time, "time", lambda: 123)

    message_connection = FakeConnection()

    def summary_cursor_factory(connection: FakeConnection) -> FakeCursor:
        cursor = FakeCursor(connection)
        cursor.fetchone_result = {"messages_since_summary": 2}
        return cursor

    summary_connection = FakeConnection(summary_cursor_factory)

    fake_connect = FakeConnect([message_connection, summary_connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test")

    store.record_exchange("conv-1", "hello", "hi there")

    executed = [entry for entry in message_connection.log if entry[0] == "execute"]
    assert len(executed) == 3

    user_insert, assistant_insert, update_stmt = executed
    assert user_insert[2] == ("conv-1", "hello", 123)
    assert assistant_insert[2] == ("conv-1", "hi there", 123)
    assert update_stmt[2] == ("conv-1",)
    assert "messages_since_summary = messages_since_summary + 2" in str(update_stmt[1])

    assert store.messages_since_summary("conv-1") == 2


def test_save_summary_resets_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PostgresChatStore, "_init_schema", lambda self: None)

    connection = FakeConnection()
    fake_connect = FakeConnect([connection])
    monkeypatch.setattr(postgres_store.psycopg, "connect", fake_connect)

    store = PostgresChatStore("postgres://test")
    store.save_summary("conv-1", "new summary text")

    executed = [entry for entry in connection.log if entry[0] == "execute"]
    assert len(executed) == 1

    query, params = executed[0][1], executed[0][2]
    assert params == ("new summary text", "conv-1")
    assert "messages_since_summary = 0" in str(query)
