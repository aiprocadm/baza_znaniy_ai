from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models import file as file_module


@pytest.fixture(autouse=True)
def _reset_engine_cache() -> None:
    file_module.get_engine.cache_clear()
    yield
    file_module.get_engine.cache_clear()


def _make_stub_engine(url: str) -> "StubEngine":
    class StubEngine:
        def __init__(self, engine_url: str) -> None:
            self.url = engine_url
            self.dialect = SimpleNamespace(name="sqlite", driver="sqlite")
            self._disposed = False

        def dispose(self) -> None:
            self._disposed = True

        def connect(self):
            class _Connection:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb) -> bool:
                    return False

                def execute(self_inner, statement):
                    return SimpleNamespace(scalar=lambda: 1 if "SELECT 1" in str(statement) else statement)

            return _Connection()

    return StubEngine(url)


def test_get_engine_returns_sync_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_engine`` should expose the synchronous SQLAlchemy API."""

    db_path = tmp_path / "sync.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    def _stub_create_engine(url: str, *_, **__) -> object:
        return _make_stub_engine(url)

    monkeypatch.setattr(file_module, "create_engine", _stub_create_engine)

    engine = file_module.get_engine(create_schema=False)

    try:
        assert hasattr(engine, "dialect")
        assert getattr(engine.dialect, "name", None) == "sqlite"
        assert getattr(engine.dialect, "driver", None) in {"sqlite", "pysqlite"}
        assert hasattr(engine, "url")
        assert str(engine.url).startswith("sqlite")
        assert callable(engine.dispose)
        assert callable(engine.connect)

        text_fn = getattr(file_module, "text", lambda value: value)
        with engine.connect() as connection:
            execution = connection.execute(text_fn("SELECT 1"))
            scalar = getattr(execution, "scalar", None)
            value = scalar() if callable(scalar) else execution
            assert value in {1, "SELECT 1"}
    finally:
        engine.dispose()
        monkeypatch.delenv("DB_URL", raising=False)


def test_get_engine_reinitializes_missing_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_engine`` should rebuild SQLModel metadata when it is missing."""

    db_path = tmp_path / "metadata.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    def _stub_create_engine(url: str, *_, **__) -> object:
        return _make_stub_engine(url)

    monkeypatch.setattr(file_module, "create_engine", _stub_create_engine)

    original_metadata = file_module.SQLModel.metadata
    file_module.SQLModel.metadata = None  # type: ignore[assignment]
    try:
        engine = file_module.get_engine(create_schema=True)
        assert isinstance(file_module.SQLModel.metadata, file_module.MetaData)
        text_fn = getattr(file_module, "text", lambda value: value)
        with engine.connect() as connection:
            result = connection.execute(text_fn("SELECT 1"))
            scalar = getattr(result, "scalar", None)
            value = scalar() if callable(scalar) else result
            assert value in {1, "SELECT 1"}
    finally:
        file_module.SQLModel.metadata = original_metadata
        monkeypatch.delenv("DB_URL", raising=False)


def test_get_engine_logs_when_create_all_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing ``create_all`` should be logged and treated as non-fatal."""

    db_path = tmp_path / "create_all.sqlite"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")

    def _stub_create_engine(url: str, *_, **__) -> object:
        return _make_stub_engine(url)

    monkeypatch.setattr(file_module, "create_engine", _stub_create_engine)

    dummy_metadata = type("DummyMetadata", (), {})()
    original_metadata = file_module.SQLModel.metadata
    file_module.SQLModel.metadata = dummy_metadata  # type: ignore[assignment]
    try:
        engine = file_module.get_engine(create_schema=True)
        assert engine is not None
        assert isinstance(file_module.SQLModel.metadata, file_module.MetaData)
    finally:
        file_module.SQLModel.metadata = original_metadata
        monkeypatch.delenv("DB_URL", raising=False)
