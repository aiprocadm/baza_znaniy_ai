"""Async tests for :mod:`app.llm.manager`."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, List

import pytest

@dataclass
class _DummySettings:
    """Minimal ``Settings`` replacement for the tests."""

    llm_model_name: str = "test-model"
    llama_cpp_model_path: Path | None = None


class _MinimalSettings:
    """Settings stub exposing only the required attributes at runtime."""

    llm_model_name: str = "test-model"


@dataclass
class _RecordingFactory:
    """Callable factory that records adapter interactions."""

    instances: List["_RecordingLlama"] = field(default_factory=list)
    load_adapter_calls: List[tuple[str, str, float]] = field(default_factory=list)
    set_adapter_calls: List[str] = field(default_factory=list)
    unload_adapter_calls: List[str] = field(default_factory=list)

    def __call__(self) -> "_RecordingLlama":  # pragma: no cover - helper used in tests only
        llama = _RecordingLlama(self)
        self.instances.append(llama)
        return llama


class _RecordingLlama:
    """Minimal llama-like stub that records method invocations."""

    def __init__(self, recorder: _RecordingFactory) -> None:
        self._recorder = recorder

    def load_adapter(self, path: str, adapter_name: str, scale: float) -> None:
        self._recorder.load_adapter_calls.append((path, adapter_name, scale))

    def set_adapter(self, adapter_name: str) -> None:
        self._recorder.set_adapter_calls.append(adapter_name)

    def unload_adapter(self, adapter_name: str) -> None:
        self._recorder.unload_adapter_calls.append(adapter_name)


def _load_manager_module(*, include_settings: bool = True) -> ModuleType:
    """Import ``app.llm.manager`` with a configurable stub configuration module."""

    repo_root = Path(__file__).resolve().parents[1]
    manager_path = repo_root / "app" / "llm" / "manager.py"

    # Provide a minimal ``app`` package structure.
    app_package = sys.modules.setdefault("app", ModuleType("app"))
    if not hasattr(app_package, "__path__"):
        app_package.__path__ = [str(repo_root / "app")]

    llm_package = ModuleType("app.llm")
    llm_package.__path__ = [str(repo_root / "app" / "llm")]
    sys.modules["app.llm"] = llm_package

    _missing = object()
    original_config = sys.modules.get("app.core.config", _missing)

    try:
        # Inject a stub ``app.core.config`` module so the manager can import ``Settings``.
        config_module = ModuleType("app.core.config")
        if include_settings:
            config_module.Settings = _DummySettings
        sys.modules["app.core.config"] = config_module

        spec = importlib.util.spec_from_file_location("app.llm.manager", manager_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["app.llm.manager"] = module
        spec.loader.exec_module(module)
    finally:
        if original_config is _missing:
            sys.modules.pop("app.core.config", None)
        else:
            sys.modules["app.core.config"] = original_config

    importlib.import_module("app.core.config")
    return module


@pytest.fixture
def manager_module() -> ModuleType:
    return _load_manager_module()


def test_manager_imports_without_settings() -> None:
    module = _load_manager_module(include_settings=False)
    assert hasattr(module, "LlamaLoraManager")


def test_manager_supports_stub_settings_when_settings_missing(tmp_path: Path) -> None:
    module = _load_manager_module(include_settings=False)
    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_text("stub")

    async def scenario() -> None:
        manager, factory = _make_manager_with_factory(
            module.LlamaLoraManager, settings=_MinimalSettings()
        )
        status = await manager.load_adapter(adapter_path, 0.5)

        assert status.loaded is True
        assert factory.load_adapter_calls == [
            (str(adapter_path.resolve()), status.adapter_name, 0.5),
        ]
        assert factory.set_adapter_calls == [status.adapter_name]

    asyncio.run(scenario())


def test_manager_missing_required_attribute_raises_clear_error_on_use(
    tmp_path: Path,
) -> None:
    module = _load_manager_module(include_settings=False)

    class _IncompleteSettings:
        llama_cpp_model_path: Path | None = None

    manager = module.LlamaLoraManager(_IncompleteSettings())

    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_text("stub")

    async def scenario() -> None:
        assert (await manager.get_status()).loaded is False
        with pytest.raises(
            AttributeError,
            match="requires settings with attribute 'llm_model_name'",
        ):
            await manager.load_adapter(adapter_path, 0.5)

    asyncio.run(scenario())


def _make_manager_with_factory(
    manager_cls: type,
    *,
    settings: Any | None = None,
) -> tuple[Any, _RecordingFactory]:
    factory = _RecordingFactory()
    manager = manager_cls(settings or _DummySettings(), llama_factory=factory)
    return manager, factory


def test_load_adapter_returns_status_and_reuses_adapter_name(
    tmp_path: Path, manager_module: ModuleType
) -> None:
    adapter_path = tmp_path / "My Fancy Adapter.safetensors"
    adapter_path.write_text("stub")

    async def scenario() -> None:
        manager_cls = manager_module.LlamaLoraManager
        manager, factory = _make_manager_with_factory(manager_cls)

        scaling = 0.75
        status = await manager.load_adapter(adapter_path, scaling)

        resolved = adapter_path.resolve()
        expected_adapter_name = manager._adapter_name_from_path(resolved)

        assert status.loaded is True
        assert status.path == resolved
        assert status.scaling == scaling
        assert status.adapter_name == expected_adapter_name

        assert factory.load_adapter_calls == [
            (str(resolved), expected_adapter_name, scaling),
        ]
        assert factory.set_adapter_calls == [expected_adapter_name]
        assert len(factory.instances) == 1

        with pytest.raises(manager_module.AdapterAlreadyLoadedError):
            await manager.load_adapter(adapter_path, scaling)

        # The failed load should not rebuild the llama instance.
        assert len(factory.instances) == 1

    asyncio.run(scenario())


def test_load_adapter_missing_file_raises(
    tmp_path: Path, manager_module: ModuleType
) -> None:
    async def scenario() -> None:
        manager_cls = manager_module.LlamaLoraManager
        manager, _ = _make_manager_with_factory(manager_cls)

        with pytest.raises(FileNotFoundError):
            await manager.load_adapter(tmp_path / "missing.safetensors", 1.0)

    asyncio.run(scenario())


def test_unload_adapter_error_conditions(
    tmp_path: Path, manager_module: ModuleType
) -> None:
    async def scenario() -> None:
        manager_cls = manager_module.LlamaLoraManager
        manager, factory = _make_manager_with_factory(manager_cls)

        with pytest.raises(manager_module.AdapterNotLoadedError):
            await manager.unload_adapter()

        adapter_path = tmp_path / "adapter.safetensors"
        adapter_path.write_text("data")
        await manager.load_adapter(adapter_path, 0.5)

        with pytest.raises(manager_module.AdapterNotLoadedError):
            await manager.unload_adapter(tmp_path / "different.safetensors")

        # Ensure the adapter is still active after the failed unload.
        status_after_failed = await manager.get_status()
        assert status_after_failed.loaded is True
        assert factory.unload_adapter_calls == []

    asyncio.run(scenario())


def test_unload_adapter_resets_status_and_rebuilds(
    tmp_path: Path, manager_module: ModuleType
) -> None:
    async def scenario() -> None:
        manager_cls = manager_module.LlamaLoraManager
        manager, factory = _make_manager_with_factory(manager_cls)

        adapter_path = tmp_path / "adapter.safetensors"
        adapter_path.write_text("data")
        await manager.load_adapter(adapter_path, 0.25)
        status = await manager.get_status()
        assert status.adapter_name is not None
        first_llama = factory.instances[0]

        unloaded_status = await manager.unload_adapter(adapter_path)

        assert factory.unload_adapter_calls == [status.adapter_name]
        assert unloaded_status.loaded is False
        assert unloaded_status.path is None
        assert unloaded_status.scaling is None
        assert unloaded_status.adapter_name is None

        # ``unload_adapter`` rebuilds the llama instance via the factory.
        assert len(factory.instances) == 2
        assert factory.instances[0] is first_llama
        assert factory.instances[1] is not first_llama

        # The new llama should not have any adapter interactions yet.
        assert factory.load_adapter_calls == [
            (str(adapter_path.resolve()), status.adapter_name, 0.25),
        ]
        assert factory.set_adapter_calls == [status.adapter_name]

        # Manager status is cleared after unload.
        final_status = await manager.get_status()
        assert final_status.loaded is False
        assert final_status.path is None
        assert final_status.scaling is None
        assert final_status.adapter_name is None

    asyncio.run(scenario())
