"""Tests for configuring the chat memory store via environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import main as app_main


@pytest.fixture(autouse=True)
def clear_chat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure chat memory environment variables do not leak between tests."""

    for key in [
        "CHAT_MEMORY_ENABLED",
        "CHAT_MEMORY_TTL_DAYS",
        "CHAT_SUMMARY_TRIGGER",
        "CHAT_MEMORY_MAXTOK",
        "MEMORY_ENABLED",
        "MEMORY_TTL_DAYS",
        "MEMORY_SUMMARY_TRIGGER",
        "MEMORY_MAX_TOKENS",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_init_memory_store_uses_chat_memory_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyMemoryStore:
        def __init__(
            self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int
        ) -> None:
            captured["instance"] = self
            captured["db_path"] = db_path
            captured["ttl_days"] = ttl_days
            captured["summary_trigger"] = summary_trigger
            captured["max_tokens"] = max_tokens

    monkeypatch.setattr(app_main, "MemoryStore", DummyMemoryStore)

    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "true")
    monkeypatch.setenv("CHAT_MEMORY_TTL_DAYS", "15")
    monkeypatch.setenv("CHAT_SUMMARY_TRIGGER", "7")
    monkeypatch.setenv("CHAT_MEMORY_MAXTOK", "1234")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "chat_memory.sqlite"))

    store = app_main._init_memory_store()

    assert isinstance(store, DummyMemoryStore)
    assert captured["db_path"] == str(tmp_path / "chat_memory.sqlite")
    assert captured["ttl_days"] == 15
    assert captured["summary_trigger"] == 7
    assert captured["max_tokens"] == 1234


def test_init_memory_store_falls_back_to_legacy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyMemoryStore:
        def __init__(
            self, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int
        ) -> None:
            captured["instance"] = self
            captured["db_path"] = db_path
            captured["ttl_days"] = ttl_days
            captured["summary_trigger"] = summary_trigger
            captured["max_tokens"] = max_tokens

    monkeypatch.setattr(app_main, "MemoryStore", DummyMemoryStore)

    monkeypatch.setenv("MEMORY_ENABLED", "yes")
    monkeypatch.setenv("MEMORY_TTL_DAYS", "21")
    monkeypatch.setenv("MEMORY_SUMMARY_TRIGGER", "9")
    monkeypatch.setenv("MEMORY_MAX_TOKENS", "987")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "legacy_chat_memory.sqlite"))

    store = app_main._init_memory_store()

    assert isinstance(store, DummyMemoryStore)
    assert captured["db_path"] == str(tmp_path / "legacy_chat_memory.sqlite")
    assert captured["ttl_days"] == 21
    assert captured["summary_trigger"] == 9
    assert captured["max_tokens"] == 987
