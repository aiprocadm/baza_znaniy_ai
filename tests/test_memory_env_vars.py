"""Tests for configuring the chat memory store via environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

import importlib.util
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "srv" / "projects" / "kb" / "app"


def _load_service_main(tmp_path: Path):
    package_name = "kb_service_memory_env"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    config_spec = importlib.util.spec_from_file_location(
        f"{package_name}.config", SERVICE_ROOT / "config.py"
    )
    assert config_spec and config_spec.loader
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)
    config_module.get_settings.cache_clear()

    main_spec = importlib.util.spec_from_file_location(
        f"{package_name}.main", SERVICE_ROOT / "main.py"
    )
    assert main_spec and main_spec.loader
    main_module = importlib.util.module_from_spec(main_spec)
    sys.modules[main_spec.name] = main_module
    main_spec.loader.exec_module(main_module)
    return main_module


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


