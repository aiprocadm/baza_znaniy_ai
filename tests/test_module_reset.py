from types import ModuleType, SimpleNamespace

import sys

from app import _module_reset


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
