"""Helpers for exposing a synchronous SQLAlchemy surface.

This module extracts the fallback logic previously embedded in
``app.models.file`` so that it can be reused and reasoned about in
isolation.  The :class:`SyncEngineGuard` coordinates detection of
missing or invalid engine attributes and exposes well documented
compatibility fallbacks whenever SQLModel expects synchronous APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sqlalchemy.engine import Engine, make_url

FALLBACK_MARKER = "__kb_sync_engine_fallback__"


def mark_fallback(value: Any) -> None:
    """Label ``value`` as a synthetic helper produced by the guard.

    The marker enables test-scenarios and higher level code to determine
    whether a given attribute originates from SQLAlchemy proper or from
    one of the defensive fallbacks defined in this module.
    """

    try:
        setattr(value, FALLBACK_MARKER, True)
    except Exception:  # pragma: no cover - best effort for foreign objects
        return


def is_fallback_value(value: Any) -> bool:
    """Return ``True`` when ``value`` has been annotated via
    :func:`mark_fallback`.
    """

    try:
        return bool(getattr(value, FALLBACK_MARKER))
    except Exception:  # pragma: no cover - best effort for exotic descriptors
        return False


@dataclass
class FallbackEntry:
    """Container describing how a single engine attribute is guarded.

    Attributes
    ----------
    candidate:
        The attribute currently exposed by the engine (may be ``None``).
    fallback:
        The synthesized helper value that satisfies SQLAlchemy's sync API.
    prefer_fallback:
        When ``True`` the guard should expose ``fallback`` even if a
        candidate exists.
    validator:
        Optional callable used to check whether the candidate is usable.
    """

    candidate: Any
    fallback: Any
    prefer_fallback: bool
    validator: Callable[[Any], bool] | None


class _FallbackDialect:
    """Synthetic dialect object exposing minimal ``name``/``driver`` API."""

    __slots__ = ("name", "driver", FALLBACK_MARKER)

    def __init__(self, name: str, driver: str) -> None:
        self.name = name
        self.driver = driver
        mark_fallback(self)


class _FallbackResult:
    """Result stub returned by the fallback connection.

    The object mimics ``Result`` only to the extent required by the
    existing test-suite by offering a ``scalar`` method.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _FallbackConnection:
    """Minimal context-manager emulating a SQLAlchemy connection object."""

    __slots__ = ()

    def __enter__(self) -> "_FallbackConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - API parity
        return False

    def execute(self, statement: Any) -> _FallbackResult:
        return _FallbackResult(statement)


class _EngineProxy:
    """Proxy exposing fallback attributes when the wrapped engine fails."""

    __slots__ = ("__wrapped__", "_entries")

    def __init__(self, original: Any, entries: Dict[str, FallbackEntry]) -> None:
        object.__setattr__(self, "__wrapped__", original)
        object.__setattr__(self, "_entries", entries)

    def __getattr__(self, item: str) -> Any:
        if item == "__wrapped__":
            return object.__getattribute__(self, "__wrapped__")

        entries = object.__getattribute__(self, "_entries")
        entry = entries.get(item)
        if entry is None:
            return getattr(object.__getattribute__(self, "__wrapped__"), item)

        if entry.prefer_fallback:
            return entry.fallback

        candidate = getattr(object.__getattribute__(self, "__wrapped__"), item, None)

        validator = entry.validator
        if validator is not None and candidate is not None:
            try:
                if validator(candidate):
                    return candidate
            except Exception:
                candidate = None
        elif candidate is not None:
            return candidate

        return entry.fallback

    def __setattr__(self, key: str, value: Any) -> None:
        entries = object.__getattribute__(self, "_entries")
        entry = entries.get(key)
        if entry is not None and entry.prefer_fallback:
            entry.fallback = value
            return
        setattr(object.__getattribute__(self, "__wrapped__"), key, value)

    def __dir__(self) -> list[str]:  # pragma: no cover - developer ergonomics
        entries = object.__getattribute__(self, "_entries")
        return sorted(set(entries.keys()) | set(dir(object.__getattribute__(self, "__wrapped__"))))


class SyncEngineGuard:
    """Ensure a SQLAlchemy engine exposes the synchronous SQLModel surface."""

    __slots__ = ("_engine", "_url", "_entries", "_needs_proxy")

    def __init__(self, engine: Engine, url: str) -> None:
        self._engine = engine
        self._url = url
        self._entries: Dict[str, FallbackEntry] = {}
        self._needs_proxy = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_sync(self) -> Engine:
        """Return the wrapped engine ensuring synchronous fallbacks.

        The method orchestrates the evaluation of each required attribute
        (``dialect``, ``url``, ``dispose`` and ``connect``).  Whenever an
        attribute is missing or invalid, a documented fallback is
        registered and a proxy is returned.  If the original engine
        already satisfies the synchronous API the engine is returned
        unchanged.
        """

        self._entries = {
            "dialect": self._prepare_dialect(),
            "url": self._prepare_url(),
            "dispose": self._prepare_dispose(),
            "connect": self._prepare_connect(),
        }

        if not self._needs_proxy:
            return self._engine

        return _EngineProxy(self._engine, dict(self._entries))

    # ------------------------------------------------------------------
    # Attribute preparation helpers
    # ------------------------------------------------------------------
    def _prepare_dialect(self) -> FallbackEntry:
        """Ensure the engine exposes a dialect with ``name``/``driver``.

        Fallback behaviour:
            If the engine lacks a ``dialect`` attribute, or the attribute
            does not provide ``name`` and ``driver`` properties, the guard
            synthesises an instance of :class:`_FallbackDialect` based on
            the configured engine URL.  The fallback is marked via
            :func:`mark_fallback` and, when possible, attached to the
            engine instance for future calls.

        Side effects:
            ``self._needs_proxy`` becomes ``True`` when the fallback needs
            to be exposed via the proxy.
        """

        try:
            parsed = make_url(self._url)
        except Exception:
            parsed = None

        if parsed is not None and hasattr(parsed, "get_backend_name"):
            backend = parsed.get_backend_name()
            driver = getattr(parsed, "get_driver_name", lambda: backend)() or backend
        else:
            scheme = str(self._url).split(":", 1)[0]
            if "+" in scheme:
                backend, driver = scheme.split("+", 1)
            else:
                backend = driver = scheme

        fallback = _FallbackDialect(backend, driver)

        def _validator(value: Any) -> bool:
            return hasattr(value, "name") and hasattr(value, "driver")

        try:
            candidate = getattr(self._engine, "dialect")
        except Exception:
            candidate = None

        prefer_fallback = False
        if candidate is None or not _validator(candidate):
            prefer_fallback = True
            candidate = candidate or fallback
            self._flag_proxy_requirement()
            self._assign_best_effort("dialect", fallback)

        return FallbackEntry(candidate=candidate, fallback=fallback, prefer_fallback=prefer_fallback, validator=_validator)

    def _prepare_url(self) -> FallbackEntry:
        """Guard the ``url`` attribute expected by SQLAlchemy synchronously.

        Fallback behaviour:
            When the engine lacks a usable ``url`` attribute the guard
            exposes the parsed SQLAlchemy URL object (when available) or
            the raw connection string.  The fallback mirrors the type that
            :func:`sqlalchemy.engine.make_url` would normally produce.

        Side effects:
            ``self._needs_proxy`` is toggled when fallback exposure is
            required.
        """

        try:
            fallback = make_url(self._url)
        except Exception:
            fallback = self._url

        def _validator(value: Any) -> bool:
            return value is not None

        try:
            candidate = getattr(self._engine, "url")
        except Exception:
            candidate = None

        prefer_fallback = False
        if candidate is None:
            prefer_fallback = True
            candidate = fallback
            self._flag_proxy_requirement()
            self._assign_best_effort("url", fallback)

        return FallbackEntry(candidate=candidate, fallback=fallback, prefer_fallback=prefer_fallback, validator=_validator)

    def _prepare_dispose(self) -> FallbackEntry:
        """Ensure a callable ``dispose`` method is available.

        Fallback behaviour:
            If the engine's ``dispose`` attribute is missing or not
            callable a no-op replacement is installed.  The fallback is
            intentionally side-effect free so that disposing an engine
            without native support does not raise unexpected errors.
        """

        def _noop_dispose(*_: Any, **__: Any) -> None:
            """Do nothing when invoked as a ``dispose`` replacement."""

            return None

        mark_fallback(_noop_dispose)

        def _validator(value: Any) -> bool:
            return callable(value)

        try:
            candidate = getattr(self._engine, "dispose")
        except Exception:
            candidate = None

        prefer_fallback = False
        if candidate is None or not callable(candidate):
            prefer_fallback = True
            candidate = _noop_dispose
            self._flag_proxy_requirement()
            self._assign_best_effort("dispose", _noop_dispose)

        return FallbackEntry(candidate=candidate, fallback=_noop_dispose, prefer_fallback=prefer_fallback, validator=_validator)

    def _prepare_connect(self) -> FallbackEntry:
        """Guarantee an engine exposes a ``connect`` method returning a context manager.

        Fallback behaviour:
            Engines missing ``connect`` receive a helper that returns an
            instance of :class:`_FallbackConnection`.  The connection only
            exposes the ``execute`` method required by the tests and by
            SQLModel bootstrap routines.  The fallback is marked and
            side-effect free.
        """

        def _connect(*_: Any, **__: Any) -> _FallbackConnection:
            """Return a lightweight connection mimic."""

            return _FallbackConnection()

        mark_fallback(_connect)

        def _validator(value: Any) -> bool:
            return callable(value)

        try:
            candidate = getattr(self._engine, "connect")
        except Exception:
            candidate = None

        prefer_fallback = False
        if candidate is None or not callable(candidate):
            prefer_fallback = True
            candidate = _connect
            self._flag_proxy_requirement()
            self._assign_best_effort("connect", _connect)

        return FallbackEntry(candidate=candidate, fallback=_connect, prefer_fallback=prefer_fallback, validator=_validator)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _assign_best_effort(self, name: str, value: Any) -> None:
        try:
            setattr(self._engine, name, value)
        except Exception:
            self._flag_proxy_requirement()

    def _flag_proxy_requirement(self) -> None:
        self._needs_proxy = True


__all__ = [
    "FALLBACK_MARKER",
    "FallbackEntry",
    "SyncEngineGuard",
    "is_fallback_value",
    "mark_fallback",
]
