"""Tests for configuring the chat memory store via environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.service_stubs import install_service_stubs

import sys
import types


def _load_service_main(tmp_path: Path):
    package_name = "kb_service_memory_env"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    install_service_stubs()

    from app.memory.store import MemoryStore
    from srv.projects.kb.app.config import get_settings

    get_settings.cache_clear()

    module = types.ModuleType(package_name)

    def _init_memory_store(settings):
        if not settings.chat_memory_enabled:
            return None

        memory_path = settings.memory_db_path_resolved
        memory_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            store_cls = module.MemoryStore
            return store_cls(
                db_path=str(memory_path),
                ttl_days=settings.chat_memory_ttl_days,
                summary_trigger=settings.chat_summary_trigger,
                max_tokens=settings.chat_memory_max_tokens,
            )
        except Exception:
            return None

    module.MemoryStore = MemoryStore
    module.get_settings = get_settings
    module._init_memory_store = _init_memory_store
    return module


@pytest.fixture(autouse=True)
def clear_chat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure chat memory environment variables do not leak between tests."""

    for key in [
        "CHAT_MEMORY_ENABLED",
        "CHAT_MEMORY_TTL_DAYS",
        "CHAT_SUMMARY_TRIGGER",
        "CHAT_MEMORY_MAXTOK",
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

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    app_main = _load_service_main(tmp_path)
    monkeypatch.setattr(app_main, "MemoryStore", DummyMemoryStore)

    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "true")
    monkeypatch.setenv("CHAT_MEMORY_TTL_DAYS", "15")
    monkeypatch.setenv("CHAT_SUMMARY_TRIGGER", "7")
    monkeypatch.setenv("CHAT_MEMORY_MAXTOK", "1234")
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "chat_memory.sqlite"))

    settings = app_main.get_settings()
    store = app_main._init_memory_store(settings)

    assert isinstance(store, DummyMemoryStore)
    assert captured["db_path"] == str(tmp_path / "chat_memory.sqlite")
    assert captured["ttl_days"] == 15
    assert captured["summary_trigger"] == 7
    assert captured["max_tokens"] == 1234


def test_init_memory_store_accepts_bare_filename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    app_main = _load_service_main(tmp_path)
    monkeypatch.setattr(app_main, "MemoryStore", DummyMemoryStore)

    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_DB_PATH", "memory.sqlite3")

    settings = app_main.get_settings()
    store = app_main._init_memory_store(settings)

    assert isinstance(store, DummyMemoryStore)
    assert captured["db_path"] == "memory.sqlite3"
    assert captured["ttl_days"] == settings.chat_memory_ttl_days
    assert captured["summary_trigger"] == settings.chat_summary_trigger
    assert captured["max_tokens"] == settings.chat_memory_max_tokens


