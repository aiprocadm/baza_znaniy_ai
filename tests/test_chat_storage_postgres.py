"""Integration tests for the PostgreSQL chat store backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import uuid

import pytest

pytest.importorskip("psycopg")
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

if getattr(psycopg, "IS_STUB", False):
    pytest.skip(
        "psycopg stub detected; skipping PostgreSQL integration tests", allow_module_level=True
    )


def _pg_binaries_available() -> bool:
    pg_config_path = shutil.which("pg_config")
    if not pg_config_path:
        return False
    try:
        bindir = subprocess.check_output([pg_config_path, "--bindir"], text=True).strip()
    except Exception:
        return False
    return Path(bindir, "pg_ctl").exists()


if not _pg_binaries_available():  # pragma: no cover - depends on environment tooling
    pytest.skip(
        "PostgreSQL binaries not available; skipping PostgreSQL tests", allow_module_level=True
    )

pytest.importorskip("pytest_postgresql")

from pytest_postgresql import factories

from app.chat.postgres_store import PostgresChatStore
from app.chat.store import ConversationAccessError

pytestmark = pytest.mark.requires_postgres

postgresql_proc = factories.postgresql_proc()


@pytest.fixture()
def postgres_dsn(postgresql_proc) -> str:
    """Provision a dedicated temporary database and return its DSN."""

    db_name = f"kb_test_{uuid.uuid4().hex}"
    admin_dsn = make_conninfo(
        dbname="postgres",
        user=postgresql_proc.user,
        password=postgresql_proc.password or None,
        host=postgresql_proc.host,
        port=postgresql_proc.port,
        options=postgresql_proc.options or None,
    )

    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))

    dsn = make_conninfo(
        dbname=db_name,
        user=postgresql_proc.user,
        password=postgresql_proc.password or None,
        host=postgresql_proc.host,
        port=postgresql_proc.port,
        options=postgresql_proc.options or None,
    )

    try:
        yield dsn
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                            sql.Identifier(db_name)
                        )
                    )
                except psycopg.errors.SyntaxError:
                    cursor.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s"
                        ),
                        (db_name,),
                    )
                    cursor.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name))
                    )


@pytest.fixture()
def pg_store(postgres_dsn: str) -> PostgresChatStore:
    return PostgresChatStore(postgres_dsn)


def test_postgres_store_roundtrip(pg_store: PostgresChatStore) -> None:
    conversation_id = pg_store.ensure_conversation("alice", None)
    assert conversation_id
    assert pg_store.ensure_conversation("alice", conversation_id) == conversation_id

    with pytest.raises(ConversationAccessError):
        pg_store.ensure_conversation("bob", conversation_id)

    pg_store.record_exchange(conversation_id, "привет", "здравствуйте")
    pg_store.record_exchange(conversation_id, "как дела?", "всё отлично")

    history = pg_store.get_recent_messages(conversation_id)
    assert history == [
        ("user", "привет"),
        ("assistant", "здравствуйте"),
        ("user", "как дела?"),
        ("assistant", "всё отлично"),
    ]
    assert pg_store.messages_since_summary(conversation_id) == 4

    pg_store.save_summary(conversation_id, "краткий итог")
    assert pg_store.get_summary(conversation_id) == "краткий итог"
    assert pg_store.messages_since_summary(conversation_id) == 0


def test_postgres_store_recent_limit(pg_store: PostgresChatStore) -> None:
    conversation_id = pg_store.ensure_conversation("alice", None)

    pg_store.record_exchange(conversation_id, "вопрос 1", "ответ 1")
    pg_store.record_exchange(conversation_id, "вопрос 2", "ответ 2")
    pg_store.record_exchange(conversation_id, "вопрос 3", "ответ 3")

    limited = pg_store.get_recent_messages(conversation_id, limit=2)
    assert limited == [
        ("user", "вопрос 3"),
        ("assistant", "ответ 3"),
    ]
