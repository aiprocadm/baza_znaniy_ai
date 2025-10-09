"""Tests for the service initialisation helpers in ``app.core.services``."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

import app.core  # noqa: F401 - ensure package exists before injecting cleaned config module

CONFIG_MODULE_NAME = "app.core.config"
if CONFIG_MODULE_NAME in sys.modules:
    del sys.modules[CONFIG_MODULE_NAME]

config_path = Path(__file__).resolve().parents[1] / "app/core/config.py"
raw_lines = config_path.read_text().splitlines()
filtered_lines: list[str] = []
for index, line in enumerate(raw_lines):
    stripped = line.strip()
    if stripped in {"codex/add-dependencies-to-requirements.txt", "main"}:
        continue
    filtered_lines.append(line)
    if stripped.startswith('validation_alias=AliasChoices("LLM_MAX_TOKENS"'):
        next_line = raw_lines[index + 1].strip() if index + 1 < len(raw_lines) else ""
        if not next_line.startswith(")"):
            filtered_lines.append("    )")

filtered_source = "\n".join(filtered_lines)
config_module = types.ModuleType(CONFIG_MODULE_NAME)
config_module.__file__ = str(config_path)
config_module.__package__ = "app.core"
exec(compile(filtered_source, str(config_path), "exec"), config_module.__dict__)
sys.modules[CONFIG_MODULE_NAME] = config_module
setattr(sys.modules["app.core"], "config", config_module)

from app.chat.store import ChatStore
from app.core.config import Settings
from app.core.services import init_chat_store, init_memory_store
from app.memory.store import MemoryStore


@pytest.mark.parametrize("backend, expect_warning", [(None, False), ("mystery", True)])
def test_init_chat_store_uses_sqlite_and_creates_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, backend: str | None, expect_warning: bool
) -> None:
    data_dir = tmp_path / "data"
    kwargs: dict[str, object] = {"data_dir": data_dir}
    if backend is not None:
        kwargs["chat_db_backend"] = backend

    settings = Settings(**kwargs)
    expected_path = settings.chat_db_path_resolved
    expected_parent = expected_path.parent
    assert not expected_parent.exists()

    if expect_warning:
        caplog.set_level(logging.WARNING)

    store = init_chat_store(settings)

    assert isinstance(store, ChatStore)
    assert Path(store.db_path) == expected_path
    assert expected_parent.exists()

    if expect_warning:
        assert any("Unknown CHAT_DB_BACKEND" in record.message for record in caplog.records)


def test_init_chat_store_postgres_requires_dsn(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", chat_db_backend="postgres")

    with pytest.raises(RuntimeError) as exc:
        init_chat_store(settings)

    assert "CHAT_DB_BACKEND=postgres" in str(exc.value)


def test_init_chat_store_postgres_fallbacks_to_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        chat_db_backend="postgres",
        chat_db_dsn="postgresql://user:pass@localhost/db",
    )
    expected_path = settings.chat_db_path_resolved

    class ExplodingStore:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, D401 - test double
            raise RuntimeError("boom")

    monkeypatch.setattr("app.chat.postgres_store.PostgresChatStore", ExplodingStore)

    store = init_chat_store(settings)

    assert isinstance(store, ChatStore)
    assert Path(store.db_path) == expected_path


def test_init_memory_store_disabled(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", chat_memory_enabled=False)

    assert init_memory_store(settings) is None


def test_init_memory_store_enabled(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        chat_memory_enabled=True,
        chat_memory_ttl_days=5,
        chat_summary_trigger=3,
        chat_memory_max_tokens=123,
    )
    expected_path = settings.memory_db_path_resolved
    expected_parent = expected_path.parent
    assert not expected_parent.exists()

    store = init_memory_store(settings)

    assert isinstance(store, MemoryStore)
    assert Path(store.db_path) == expected_path
    assert store.ttl == 5 * 86400
    assert store.trigger == 3
    assert store.max_tokens == 123
    assert expected_parent.exists()


def test_init_memory_store_logs_and_returns_none_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(data_dir=tmp_path / "data", chat_memory_enabled=True)

    class ExplodingMemoryStore:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001, D401 - test double
            raise RuntimeError("boom")

    monkeypatch.setattr("app.core.services.MemoryStore", ExplodingMemoryStore)

    caplog.set_level(logging.ERROR)

    store = init_memory_store(settings)

    assert store is None
    assert any("Failed to initialise memory store" in record.message for record in caplog.records)
