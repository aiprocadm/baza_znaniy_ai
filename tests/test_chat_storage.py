"""Tests for chat persistence and summarization layers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from app.chat.store import ChatStore, ConversationAccessError
from app.chat.summarizer import ConversationSummarizer
from app.memory.store import MemoryStore


@pytest.fixture()
def chat_store(tmp_path: Path) -> ChatStore:
    db_path = tmp_path / "chat.sqlite3"
    return ChatStore(str(db_path))


def test_ensure_conversation_creation_and_ownership(chat_store: ChatStore) -> None:
    conversation_id = chat_store.ensure_conversation("alice", None)
    assert isinstance(conversation_id, str)
    assert chat_store.ensure_conversation("alice", conversation_id) == conversation_id

    with pytest.raises(ConversationAccessError):
        chat_store.ensure_conversation("bob", conversation_id)


def test_record_exchange_and_summary_tracking(chat_store: ChatStore) -> None:
    conversation_id = chat_store.ensure_conversation("alice", None)

    chat_store.record_exchange(conversation_id, "привет", "здравствуйте")
    chat_store.record_exchange(conversation_id, "как дела?", "всё отлично")

    history = chat_store.get_recent_messages(conversation_id)
    assert history == [
        ("user", "привет"),
        ("assistant", "здравствуйте"),
        ("user", "как дела?"),
        ("assistant", "всё отлично"),
    ]
    assert chat_store.messages_since_summary(conversation_id) == 4

    chat_store.save_summary(conversation_id, "краткий итог")
    assert chat_store.get_summary(conversation_id) == "краткий итог"
    assert chat_store.messages_since_summary(conversation_id) == 0


def test_conversation_summarizer_uses_existing_summary(tmp_path: Path) -> None:
    store = ChatStore(str(tmp_path / "summary.sqlite3"))
    conversation_id = store.ensure_conversation("alice", None)

    store.record_exchange(conversation_id, "первый вопрос", "первый ответ")
    store.save_summary(conversation_id, "предыдущее саммари")
    store.record_exchange(conversation_id, "второй вопрос", "второй ответ")

    prompts: List[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "новое саммари"

    summarizer = ConversationSummarizer(store, fake_llm, max_history=10)
    summary = summarizer.summarize(conversation_id)

    assert summary == "новое саммари"
    assert store.get_summary(conversation_id) == "новое саммари"
    assert prompts, "LLM should have been invoked"
    assert "Текущее саммари: предыдущее саммари" in prompts[0]
    assert "user: второй вопрос" in prompts[0]
    assert "assistant: второй ответ" in prompts[0]


def test_conversation_summarizer_ignores_blank_summary(tmp_path: Path) -> None:
    store = ChatStore(str(tmp_path / "blank.sqlite3"))
    conversation_id = store.ensure_conversation("alice", None)

    store.record_exchange(conversation_id, "вопрос", "ответ")

    prompts: List[str] = []

    def blank_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "   "

    summarizer = ConversationSummarizer(store, blank_llm, max_history=10)

    summary = summarizer.summarize(conversation_id)

    assert summary is None
    assert store.get_summary(conversation_id) is None
    assert store.messages_since_summary(conversation_id) == 2
    assert prompts, "LLM should have been invoked"


def test_memory_store_respects_ttl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memory.sqlite3"), ttl_days=1, summary_trigger=5, max_tokens=200)

    current = {"value": 1_000_000}

    def fake_time() -> float:
        return current["value"]

    monkeypatch.setattr("app.memory.store.time.time", fake_time)

    store.record("user-1", "conv-a", "старое сообщение", "старый ответ")

    current["value"] += store.ttl + 10
    store.record("user-1", "conv-a", "новое сообщение", "новый ответ")

    transcript = store.load_context("user-1", "conv-a")
    assert transcript == "user: новое сообщение\nassistant: новый ответ"


def test_memory_store_truncates_to_token_limit(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memory_tokens.sqlite3"), ttl_days=1, summary_trigger=5, max_tokens=2)

    store.record("user-2", None, "очень длинное сообщение", "такой же длинный ответ")
    transcript = store.load_context("user-2", None)

    assert transcript.startswith("user")
    assert len(transcript) == store.max_tokens * 2
