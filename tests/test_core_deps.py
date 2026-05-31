"""Unit tests covering upload-related dependency helpers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


MANDATED_EXTENSIONS = ("docx", "md", "pdf", "pptx", "txt", "xlsx")


@pytest.fixture
def deps_module(monkeypatch: pytest.MonkeyPatch):
    """Provide the :mod:`app.core.deps` module with a stub configuration."""

    stub_config = ModuleType("app.core.config")

    def _get_settings() -> SimpleNamespace:
        return SimpleNamespace(max_upload_mb=25)

    stub_config.get_settings = _get_settings  # type: ignore[attr-defined]

    stub_ingest_pkg = ModuleType("app.ingest")
    stub_ingest_pkg.__path__ = []  # type: ignore[attr-defined]

    stub_ingest_service = ModuleType("app.ingest.service")
    stub_ingest_service.IngestService = type("IngestService", (), {})  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "app.core.config", stub_config)
    monkeypatch.setitem(sys.modules, "app.ingest", stub_ingest_pkg)
    monkeypatch.setitem(sys.modules, "app.ingest.service", stub_ingest_service)

    module = importlib.import_module("app.core.deps")
    module = importlib.reload(module)
    yield module

    # Ensure future imports use a clean module.
    sys.modules.pop("app.core.deps", None)


def test_get_upload_limits_converts_legacy_byte_limit(
    deps_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``UPLOAD_MAX_SIZE`` values in bytes should convert to megabytes."""

    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    monkeypatch.setenv("UPLOAD_MAX_SIZE", str(5 * 1024 * 1024))
    monkeypatch.delenv("UPLOAD_ALLOWED_EXTS", raising=False)

    limits = deps_module.get_upload_limits()

    assert limits.max_upload_mb == 5
    assert limits.max_size == 5 * 1024 * 1024


@pytest.mark.parametrize("extension", MANDATED_EXTENSIONS)
def test_get_upload_limits_allows_required_extensions(
    extension: str, deps_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All mandated extensions should be accepted by default."""

    for env_name in ("MAX_UPLOAD_MB", "UPLOAD_MAX_SIZE", "UPLOAD_ALLOWED_EXTS"):
        monkeypatch.delenv(env_name, raising=False)

    limits = deps_module.get_upload_limits()

    assert extension in limits.allowed_extensions


def test_get_upload_limits_parses_explicit_extensions(
    deps_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit UPLOAD_ALLOWED_EXTS list parses into the expected set."""

    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("UPLOAD_MAX_SIZE", raising=False)
    monkeypatch.setenv("UPLOAD_ALLOWED_EXTS", "pdf,docx")

    limits = deps_module.get_upload_limits()

    assert limits.allowed_extensions == {"pdf", "docx"}
