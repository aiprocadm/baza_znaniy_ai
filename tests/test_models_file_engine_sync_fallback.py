from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.models.engine_guard import FALLBACK_MARKER, SyncEngineGuard, is_fallback_value


def _assert_sqlite_dialect(dialect) -> None:
    assert getattr(dialect, "name") == "sqlite"
    assert getattr(dialect, "driver") in {"sqlite", "pysqlite"}


def test_guard_synthesizes_dialect_when_missing(tmp_path: Path) -> None:
    """A missing ``dialect`` attribute should be replaced with a fallback."""

    class EngineWithoutDialect:
        def __init__(self, url: str) -> None:
            self.url = url

    url = f"sqlite:///{tmp_path/'dialect.sqlite'}"
    engine = EngineWithoutDialect(url)

    guarded = SyncEngineGuard(engine, url).ensure_sync()

    assert guarded is engine
    _assert_sqlite_dialect(engine.dialect)
    assert getattr(engine.dialect, FALLBACK_MARKER)


def test_guard_provides_url_fallback(tmp_path: Path) -> None:
    """Engines without ``url`` should expose the connection string via fallback."""

    class EngineWithoutUrl:
        def __init__(self) -> None:
            self.dialect = SimpleNamespace(name="sqlite", driver="sqlite")
            mark = getattr(self.dialect, FALLBACK_MARKER, False)
            if mark:
                delattr(self.dialect, FALLBACK_MARKER)

    url = f"sqlite:///{tmp_path/'url.sqlite'}"
    engine = EngineWithoutUrl()

    guarded = SyncEngineGuard(engine, url).ensure_sync()

    assert guarded is engine
    assert str(engine.url).startswith("sqlite")
    assert engine.url is not None


def test_guard_replaces_dispose_when_missing(tmp_path: Path) -> None:
    """The guard should install a no-op dispose fallback when required."""

    class EngineWithoutDispose:
        def __init__(self, url: str) -> None:
            self.url = url
            self.dialect = SimpleNamespace(name="sqlite", driver="sqlite")

    url = f"sqlite:///{tmp_path/'dispose.sqlite'}"
    engine = EngineWithoutDispose(url)

    guarded = SyncEngineGuard(engine, url).ensure_sync()

    assert guarded is engine
    dispose = engine.dispose
    assert callable(dispose)
    assert getattr(dispose, FALLBACK_MARKER, False)
    dispose()


def test_guard_replaces_connect_when_missing(tmp_path: Path) -> None:
    """The guard should inject a lightweight connection helper when missing."""

    class EngineWithoutConnect:
        def __init__(self, url: str) -> None:
            self.url = url
            self.dialect = SimpleNamespace(name="sqlite", driver="sqlite")
            self.dispose = lambda: None  # noqa: E731 - simple stub

    url = f"sqlite:///{tmp_path/'connect.sqlite'}"
    engine = EngineWithoutConnect(url)

    guarded = SyncEngineGuard(engine, url).ensure_sync()

    assert guarded is engine
    connect = engine.connect
    assert callable(connect)
    assert getattr(connect, FALLBACK_MARKER, False)
    with connect() as connection:
        result = connection.execute("SELECT 1")
        assert result.scalar() == "SELECT 1"


def test_guard_returns_original_engine_when_complete(tmp_path: Path) -> None:
    """Fully featured SQLAlchemy engines should pass through untouched."""

    class CompleteEngine:
        def __init__(self, url: str) -> None:
            self.url = url
            self.dialect = SimpleNamespace(name="sqlite", driver="sqlite")
            self._disposed = False

        def dispose(self) -> None:
            self._disposed = True

        def connect(self):  # noqa: D401 - simple context manager stub
            class _Connection:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb) -> bool:
                    return False

                def execute(self_inner, statement):
                    return SimpleNamespace(scalar=lambda: statement)

            return _Connection()

    engine = CompleteEngine(f"sqlite:///{tmp_path/'complete.sqlite'}")
    guarded = SyncEngineGuard(engine, engine.url).ensure_sync()
    assert guarded is engine
    assert not is_fallback_value(engine.dispose)
    engine.dispose()
    assert engine._disposed
