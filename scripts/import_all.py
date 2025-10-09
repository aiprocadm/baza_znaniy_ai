#!/usr/bin/env python3
"""Import vector payloads and SQLite stores from a tarball."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

from app.core.config import get_settings
from app.retriever import get_vector_store

PAYLOADS_FILE = "vector_payloads.json"
MANIFEST_FILE = "manifest.json"
DB_DIR = "db"


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(payload)
    vector = normalised.get("vector")
    if vector is not None and hasattr(vector, "tolist"):
        normalised["vector"] = list(vector.tolist())
    elif isinstance(vector, tuple):
        normalised["vector"] = list(vector)
    return normalised


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_FILE
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _safe_members(members: Iterable[tarfile.TarInfo]) -> Iterable[tarfile.TarInfo]:
    """Yield members that are safe to extract under the current directory."""

    for member in members:
        name = member.name or ""
        if not name:
            continue
        if os.path.isabs(name):
            continue
        path = Path(name)
        if any(part == ".." for part in path.parts):
            continue
        yield member


def _restore_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def import_all(path: Path, *, reset: bool = False) -> int:
    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        with tarfile.open(path, "r:gz") as tar:
            safe_members = list(_safe_members(tar.getmembers()))
            tar.extractall(root, members=safe_members, filter="data")

        manifest = _load_manifest(root)

        payloads_file = manifest.get("vector_payloads") or PAYLOADS_FILE
        payloads_path = root / payloads_file
        if not payloads_path.exists():
            raise FileNotFoundError(f"Не найден файл с векторами: {payloads_file}")

        chat_entry = manifest.get("chat_db")
        if isinstance(chat_entry, str):
            chat_source = root / chat_entry
        else:
            chat_source = root / DB_DIR / settings.chat_db_path_resolved.name
        if chat_source.exists():
            _restore_sqlite(chat_source, settings.chat_db_path_resolved)

        memory_entry = manifest.get("memory_db")
        if isinstance(memory_entry, str):
            memory_source = root / memory_entry
        else:
            memory_source = root / DB_DIR / settings.memory_db_path_resolved.name
        if memory_source.exists():
            _restore_sqlite(memory_source, settings.memory_db_path_resolved)

        data = json.loads(payloads_path.read_text())
    vector_store = get_vector_store(settings)
    if reset:
        if hasattr(vector_store, "reset_collection"):
            vector_store.reset_collection()
        else:  # pragma: no cover - utility guard
            raise NotImplementedError("Active vector store does not support reset")
    else:
        vector_store.ensure_ready()

    if not isinstance(data, list):
        raise ValueError("Ожидается список записей в JSON")

    payloads = [_normalise_payload(item) for item in data if isinstance(item, dict)]
    if hasattr(vector_store, "import_payloads"):
        vector_store.import_payloads(payloads)
    else:  # pragma: no cover - utility guard
        raise NotImplementedError("Active vector store does not support import")
    return len(payloads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Путь к tar.gz архиву экспорта",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Очистить коллекцию перед импортом",
    )
    args = parser.parse_args()

    count = import_all(args.input, reset=args.reset)
    print(f"Импортировано записей: {count}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
