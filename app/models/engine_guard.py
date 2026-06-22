"""Ensure SQLAlchemy engines expose the synchronous SQLModel API surface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

from sqlalchemy.engine import URL, Engine, make_url

FALLBACK_MARKER = "__kb_ai_engine_fallback__"


def mark_fallback(value: Any) -> None:
    """Annotate ``value`` so tests can detect synthetic helpers."""

    try:
        setattr(value, FALLBACK_MARKER, True)
    except Exception:  # pragma: no cover - best-effort for foreign objects
        pass


def is_fallback_value(value: Any) -> bool:
    """Return ``True`` when ``value`` was produced by :func:`mark_fallback`."""

    try:
        return bool(getattr(value, FALLBACK_MARKER))
    except Exception:  # pragma: no cover - foreign descriptors may raise
        return False


class _FallbackResult:
    """Minimal Result stub exposing ``scalar`` for compatibility."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:  # pragma: no cover - trivial accessor
        return self._value


class _FallbackConnection:
    """Context manager returning a light-weight connection wrapper."""

    __slots__ = ()

    def __enter__(self) -> "_FallbackConnection":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:  # pragma: no cover - trivial
        return False

    def execute(self, statement: Any) -> _FallbackResult:
        return _FallbackResult(statement)


def _dialect_from_url(url: str) -> SimpleNamespace:
    try:
        parsed = make_url(url)
    except Exception:  # pragma: no cover - fallback to simple split
        parsed = None

    if parsed is not None and hasattr(parsed, "get_backend_name"):
        backend = parsed.get_backend_name()
        driver = getattr(parsed, "get_driver_name", lambda: backend)() or backend
    else:
        scheme = str(url).split(":", 1)[0]
        if "+" in scheme:
            backend, driver = scheme.split("+", 1)
        else:
            backend = driver = scheme or "sqlite"

    dialect = SimpleNamespace(name=backend or "sqlite", driver=driver or backend)
    mark_fallback(dialect)
    return dialect


class SyncEngineGuard:
    """Mutate SQLAlchemy engines to expose synchronous attributes."""

    __slots__ = ("_engine", "_url")

    def __init__(self, engine: Engine, url: str) -> None:
        self._engine = engine
        self._url = url

    def ensure_sync(self) -> Engine:
        """Guarantee the wrapped engine exposes sync-only helpers."""

        self._ensure_dialect()
        self._ensure_url()
        self._ensure_dispose()
        self._ensure_connect()
        return self._engine

    # ------------------------------------------------------------------
    # Attribute guards
    # ------------------------------------------------------------------
    def _ensure_dialect(self) -> None:
        dialect = getattr(self._engine, "dialect", None)
        if hasattr(dialect, "name") and hasattr(dialect, "driver"):
            return

        fallback = _dialect_from_url(self._url)
        self._set_attr("dialect", fallback)

    def _ensure_url(self) -> None:
        candidate = getattr(self._engine, "url", None)
        if candidate is not None:
            return

        fallback: URL | str  # declared here so mypy sees both branch types
        try:
            fallback = make_url(self._url)
        except Exception:
            fallback = self._url
        self._set_attr("url", fallback)

    def _ensure_dispose(self) -> None:
        dispose = getattr(self._engine, "dispose", None)
        if callable(dispose):
            return

        def _noop_dispose(*_: Any, **__: Any) -> None:  # pragma: no cover - trivial
            return None

        mark_fallback(_noop_dispose)
        self._set_attr("dispose", _noop_dispose)

    def _ensure_connect(self) -> None:
        connect = getattr(self._engine, "connect", None)
        if callable(connect):
            return

        def _connect(*_: Any, **__: Any) -> _FallbackConnection:
            return _FallbackConnection()

        mark_fallback(_connect)
        self._set_attr("connect", _connect)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _set_attr(self, name: str, value: Any) -> None:
        try:
            setattr(self._engine, name, value)
        except Exception as exc:  # pragma: no cover - propagate configuration errors
            raise RuntimeError(
                f"Unable to assign fallback attribute '{name}' to SQLAlchemy engine"
            ) from exc


__all__ = ["FALLBACK_MARKER", "SyncEngineGuard", "is_fallback_value", "mark_fallback"]
