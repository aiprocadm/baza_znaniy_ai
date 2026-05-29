"""Tests for kb_feedback table + KnowledgeBaseStore.store_feedback()."""

from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path):
    from app.services.kb_store import KnowledgeBaseStore

    s = KnowledgeBaseStore(db_path=tmp_path / "kb.sqlite")
    conv = s.create_conversation(title="test")
    user_msg = s.add_message(conversation_id=conv.id, role="user", content="hello?")
    asst_msg = s.add_message(conversation_id=conv.id, role="assistant", content="hi back")
    return s, conv.id, user_msg.id, asst_msg.id


def test_feedback_table_created(store) -> None:
    s, _conv, _user_msg, _asst_msg = store
    with s._connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kb_feedback'"
        ).fetchone()
    assert row is not None


def test_store_feedback_persists_and_returns_id(store) -> None:
    s, conv, _user_msg, asst_msg = store
    fid = s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=1,
        comment=None,
        alternative_answer=None,
    )
    assert isinstance(fid, str) and len(fid) >= 8
    with s._connect() as conn:
        row = conn.execute(
            "SELECT rating, comment FROM kb_feedback WHERE id=?", (fid,)
        ).fetchone()
    assert row[0] == 1
    assert row[1] is None


def test_store_feedback_rejects_invalid_rating(store) -> None:
    import sqlite3

    s, conv, _user_msg, asst_msg = store
    with pytest.raises((sqlite3.IntegrityError, ValueError)):
        s.store_feedback(
            conversation_id=conv,
            message_id=asst_msg,
            user_id="u1",
            rating=5,
            comment=None,
            alternative_answer=None,
        )


def test_store_feedback_accepts_alternative_answer(store) -> None:
    s, conv, _user_msg, asst_msg = store
    fid = s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment="плохо",
        alternative_answer="лучше так",
    )
    with s._connect() as conn:
        row = conn.execute(
            "SELECT alternative_answer FROM kb_feedback WHERE id=?", (fid,)
        ).fetchone()
    assert row[0] == "лучше так"
