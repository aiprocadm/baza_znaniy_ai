"""Utilities for reloading application modules after test stubs."""

from __future__ import annotations

import importlib
from importlib import import_module
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _PROJECT_ROOT / "tests"


def _is_stub_module(module: ModuleType) -> bool:
    """Return ``True`` when *module* is a lightweight test stub."""

    module_file = getattr(module, "__file__", None)
    if not module_file:
        # A PEP 420 namespace package (e.g. ``app.services``, which has no
        # ``__init__.py``) also reports ``__file__ is None`` — yet it is the real
        # package, not a stub. Purging it from ``sys.modules`` orphans its
        # already-imported submodules: the namespace is rebuilt lazily and never
        # re-binds them as attributes, so a later
        # ``monkeypatch.setattr("app.services.kb_embeddings.get_embedder")`` can
        # no longer resolve the dotted path. Treat a module backed by on-disk
        # source directories outside ``tests/`` as genuine; only bare stub
        # modules (without a real ``__path__``) are stubs.
        for entry in getattr(module, "__path__", None) or ():
            try:
                resolved = Path(entry).resolve()
            except Exception:  # pragma: no cover - defensive
                continue
            if resolved.is_dir() and _TESTS_ROOT not in resolved.parents:
                return False
        return True

    try:
        path = Path(module_file).resolve()
    except Exception:  # pragma: no cover - extremely defensive
        return False

    try:
        return _TESTS_ROOT in path.parents
    except Exception:  # pragma: no cover - defensive for unexpected parents
        return False


def _purge_modules(names: Iterable[str]) -> None:
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        if _is_stub_module(module):
            sys.modules.pop(name, None)


def ensure_core_modules() -> None:
    """Reload core modules when earlier tests registered stubs."""

    core_app = sys.modules.get("app.core.app")

    config_module = sys.modules.get("app.core.config")
    if config_module is not None:
        settings_cls = getattr(config_module, "Settings", None)
        settings_ok = hasattr(settings_cls, "ingest_max_retries") if settings_cls else False
        if getattr(config_module, "__file__", None) is None or not settings_ok:
            importlib.reload(import_module("app.core.config"))

    _purge_modules(
        (
            "app.core.config",
            "app.core.services",
            "app.api",
            "app.api.router",
            "app.api.routes",
            "app.ingest",
            "app.ingest.service",
            "app.services",
            "app.services.vectorstore",
            "app.retriever",
            "app.chat",
            "app.chat.summarizer",
            "app.llm",
            "app.llm.manager",
            "app.models.lora",
            "app.services.files",
        )
    )

    if core_app is not None:
        spec = getattr(core_app, "__spec__", None)
        if getattr(spec, "_initializing", False):
            # ``ensure_core_modules`` may run while ``app.core.app`` is still being
            # imported (for example, when triggered from ``app.ingest.service``).
            # Attempting to reload the module at that point raises an import error
            # because the module is only partially initialised, so we bail out and
            # allow the original import to complete before any reload happens.
            return

        importlib.reload(core_app)


__all__ = ["ensure_core_modules"]
