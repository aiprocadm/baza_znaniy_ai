import json
import os
import sys
import tarfile
import types
from pathlib import Path


def _install_config_stub() -> None:
    try:
        import importlib

        importlib.import_module("app.core.config")
        return
    except Exception:
        sys.modules.pop("app.core.config", None)

    module = types.ModuleType("app.core.config")

    class _StubSettings:
        def __init__(self) -> None:
            data_dir = Path(os.environ.get("DATA_DIR", ".")).resolve()
            self.data_dir = data_dir
            self.chat_db_path_resolved = data_dir / "db" / "chat_history.sqlite"
            self.memory_db_path_resolved = data_dir / "db" / "memory.sqlite"

    def _get_settings() -> _StubSettings:
        return _StubSettings()

    def _cache_clear() -> None:  # pragma: no cover - trivial stub
        pass

    _get_settings.cache_clear = _cache_clear  # type: ignore[attr-defined]
    module.get_settings = _get_settings  # type: ignore[attr-defined]
    module.Settings = _StubSettings  # type: ignore[attr-defined]
    sys.modules["app.core.config"] = module


def _install_retriever_stub() -> None:
    try:
        import importlib

        importlib.import_module("app.retriever")
        return
    except Exception:
        sys.modules.pop("app.retriever", None)

    module = types.ModuleType("app.retriever")

    def _unpatched_get_vector_store(*args, **kwargs):  # pragma: no cover - stub safety
        raise RuntimeError("get_vector_store must be patched in tests")

    module.get_vector_store = _unpatched_get_vector_store  # type: ignore[attr-defined]
    sys.modules["app.retriever"] = module


_install_config_stub()
_install_retriever_stub()

from app.core.config import get_settings
from scripts import export_all as export_module
from scripts import import_all as import_module


class DummyExportStore:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = payloads
        self.ready = False

    def ensure_ready(self) -> None:
        self.ready = True

    def export_payloads(self):
        for payload in self._payloads:
            yield payload


class DummyImportStore:
    def __init__(self) -> None:
        self.reset_called = False
        self.ready = False
        self.imported: list[dict[str, object]] = []

    def ensure_ready(self) -> None:
        self.ready = True

    def reset_collection(self) -> None:
        self.reset_called = True

    def import_payloads(self, payloads):
        self.imported.extend(payloads)


def _prepare_data_dir(base: Path) -> tuple[Path, Path]:
    chat_db = base / "db" / "chat_history.sqlite"
    memory_db = base / "db" / "memory.sqlite"
    chat_db.parent.mkdir(parents=True, exist_ok=True)
    chat_db.write_text("chat data")
    memory_db.write_text("memory data")
    return chat_db, memory_db


def test_export_all_creates_tarball(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    chat_db, memory_db = _prepare_data_dir(data_dir)

    payloads = [
        {"id": "1", "vector": (1.0, 2.0, 3.0), "text": "chunk", "sha256": "one"},
    ]

    store = DummyExportStore(payloads)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    monkeypatch.setattr(export_module, "get_vector_store", lambda settings: store)

    archive = tmp_path / "export.tar.gz"
    count = export_module.export_all(archive)

    assert count == len(payloads)
    assert store.ready is True
    assert archive.is_file()

    with tarfile.open(archive, "r:gz") as tar:
        members = set(tar.getnames())
        assert export_module.PAYLOADS_FILE in members
        assert f"{export_module.DB_DIR}/{chat_db.name}" in members
        assert f"{export_module.DB_DIR}/{memory_db.name}" in members
        assert export_module.MANIFEST_FILE in members
        extract_dir = tmp_path / "extracted"
        tar.extractall(extract_dir)

    data = json.loads((extract_dir / export_module.PAYLOADS_FILE).read_text())
    assert data[0]["vector"] == [1.0, 2.0, 3.0]
    assert (extract_dir / export_module.DB_DIR / chat_db.name).read_text() == "chat data"
    assert (extract_dir / export_module.DB_DIR / memory_db.name).read_text() == "memory data"


def test_import_all_restores_tarball(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    chat_db, memory_db = _prepare_data_dir(data_dir)

    payloads = [
        {"id": "1", "vector": [5.0, 6.0], "text": "hello", "sha256": "abc"},
        {"id": "2", "vector": (7.0, 8.0), "text": "world", "sha256": "def"},
    ]

    export_store = DummyExportStore(payloads)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    monkeypatch.setattr(export_module, "get_vector_store", lambda settings: export_store)

    archive = tmp_path / "export.tar.gz"
    export_module.export_all(archive)

    chat_db.write_text("stale")
    memory_db.unlink()

    import_store = DummyImportStore()
    monkeypatch.setattr(import_module, "get_vector_store", lambda settings: import_store)

    get_settings.cache_clear()
    restored = import_module.import_all(archive, reset=True)

    assert restored == len(payloads)
    assert import_store.reset_called is True
    assert import_store.ready is False  # ensure_ready should not run when reset is available
    vectors = import_store.imported
    assert len(vectors) == len(payloads)
    assert vectors[0]["vector"] == [5.0, 6.0]
    assert vectors[1]["vector"] == [7.0, 8.0]

    assert chat_db.read_text() == "chat data"
    assert memory_db.read_text() == "memory data"
