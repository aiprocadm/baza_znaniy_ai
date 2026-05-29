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


def test_iter_feedback_pairs_emits_alt_when_thumbs_down(store) -> None:
    """Thumbs-down with alternative_answer → (alt is chosen, assistant is rejected)."""
    from app.services.dpo_dataset import DPOPair, RejectStrategy

    s, conv, _user_msg, asst_msg = store
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer="более точный ответ",
    )

    pairs = list(s.iter_feedback_pairs())
    assert len(pairs) == 1
    p = pairs[0]
    assert isinstance(p, DPOPair)
    assert p.strategy is RejectStrategy.LIVE_ALT
    assert p.prompt == "hello?"
    assert p.chosen == "более точный ответ"
    assert p.rejected == "hi back"
    assert p.source == "live"
    assert len(p.feedback_ids) == 1


def test_iter_feedback_pairs_skips_when_no_alt_and_thumbs_down(store) -> None:
    """Thumbs-down without alternative_answer is insufficient signal — skip."""
    s, conv, _user_msg, asst_msg = store
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer=None,
    )
    assert list(s.iter_feedback_pairs()) == []


def test_iter_feedback_pairs_uses_most_recent_per_user(store) -> None:
    """If a user flips rating, only the latest counts."""
    s, conv, _user_msg, asst_msg = store
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=1,
        comment=None,
        alternative_answer=None,
    )
    s.store_feedback(
        conversation_id=conv,
        message_id=asst_msg,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer="лучше",
    )

    pairs = list(s.iter_feedback_pairs())
    assert len(pairs) == 1
    assert pairs[0].chosen == "лучше"


def test_iter_feedback_pairs_skips_orphan_assistant_messages(tmp_path) -> None:
    """Assistant message with no preceding user message → skip with debug log."""
    from app.services.kb_store import KnowledgeBaseStore

    s2 = KnowledgeBaseStore(db_path=tmp_path / "kb_orphan.sqlite")
    conv = s2.create_conversation(title="orphan")
    asst_msg = s2.add_message(
        conversation_id=conv.id, role="assistant", content="answer"
    )
    s2.store_feedback(
        conversation_id=conv.id,
        message_id=asst_msg.id,
        user_id="u1",
        rating=-1,
        comment=None,
        alternative_answer="alt",
    )
    assert list(s2.iter_feedback_pairs()) == []
