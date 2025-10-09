from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from app import _module_reset


def _healthy_config_module() -> ModuleType:
    module = ModuleType("app.core.config")
    module.__file__ = str(_module_reset._PROJECT_ROOT / "app" / "core" / "config.py")

    class Settings:  # noqa: D401 - helper for tests
        """Settings stub exposing all required attributes."""

        ingest_max_retries = 3

    module.Settings = Settings
    return module


@pytest.fixture(autouse=True)
def _patch_config_module(monkeypatch):
    """Ensure config module looks healthy so ``ensure_core_modules`` skips reload."""

    module = _healthy_config_module()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    yield


def test_ensure_core_modules_skips_reload_during_initialisation(monkeypatch):
    module = ModuleType("app.core.app")
    module.__spec__ = SimpleNamespace(_initializing=True)
    monkeypatch.setitem(sys.modules, "app.core.app", module)

    reload_called = False

    def fake_reload(target):
        nonlocal reload_called
        reload_called = True
        return target

    monkeypatch.setattr(_module_reset.importlib, "reload", fake_reload)

    _module_reset.ensure_core_modules()

    assert not reload_called


def test_ensure_core_modules_reload_when_initialisation_complete(monkeypatch):
    module = ModuleType("app.core.app")
    module.__spec__ = SimpleNamespace(_initializing=False)
    monkeypatch.setitem(sys.modules, "app.core.app", module)

    reloaded = False

    def fake_reload(target):
        nonlocal reloaded
        reloaded = True
        return target

    monkeypatch.setattr(_module_reset.importlib, "reload", fake_reload)

    _module_reset.ensure_core_modules()

    assert reloaded


def test_is_stub_module_without_file():
    module = ModuleType("tests.some_stub")
    assert _module_reset._is_stub_module(module)


def test_is_stub_module_within_tests_directory(tmp_path):
    stub_file = tmp_path / "tests" / "stub.py"
    stub_file.parent.mkdir(parents=True, exist_ok=True)
    stub_file.write_text("# stub")

    module = ModuleType("tests.generated")
    module.__file__ = str(stub_file)

    # Simulate project layout by pointing _TESTS_ROOT to the temporary directory.
    original_root = _module_reset._TESTS_ROOT
    try:
        _module_reset._TESTS_ROOT = tmp_path
        assert _module_reset._is_stub_module(module)
    finally:
        _module_reset._TESTS_ROOT = original_root


def test_is_stub_module_outside_tests_directory(tmp_path):
    file_path = tmp_path / "app" / "core" / "module.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("print('ok')")

    module = ModuleType("app.core.module")
    module.__file__ = str(file_path)

    original_root = _module_reset._TESTS_ROOT
    try:
        _module_reset._TESTS_ROOT = tmp_path / "tests"
        assert not _module_reset._is_stub_module(module)
    finally:
        _module_reset._TESTS_ROOT = original_root


def test_purge_modules_drops_stub_and_keeps_real(monkeypatch, tmp_path):
    stub_path = tmp_path / "tests" / "stub_module.py"
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text("# stub")

    real_path = tmp_path / "app" / "core" / "real_module.py"
    real_path.parent.mkdir(parents=True, exist_ok=True)
    real_path.write_text("# real")

    stub_module = ModuleType("app.services.files")
    stub_module.__file__ = str(stub_path)
    real_module = ModuleType("app.core.services")
    real_module.__file__ = str(real_path)

    original_root = _module_reset._TESTS_ROOT
    try:
        _module_reset._TESTS_ROOT = tmp_path / "tests"
        monkeypatch.setitem(sys.modules, stub_module.__name__, stub_module)
        monkeypatch.setitem(sys.modules, real_module.__name__, real_module)

        _module_reset._purge_modules([stub_module.__name__, real_module.__name__])

        assert stub_module.__name__ not in sys.modules
        assert real_module.__name__ in sys.modules
    finally:
        _module_reset._TESTS_ROOT = original_root
