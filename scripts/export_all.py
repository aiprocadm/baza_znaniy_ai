#!/usr/bin/env python3
"""Export vector payloads and SQLite stores into a tarball."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.retriever import get_vector_store

PAYLOADS_FILE = "vector_payloads.json"
MANIFEST_FILE = "manifest.json"
DB_DIR = "db"


def _serialise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    serialised = dict(payload)
    vector = serialised.get("vector")
    if vector is not None:
        if hasattr(vector, "tolist"):
            serialised["vector"] = list(vector.tolist())
        else:
            serialised["vector"] = list(vector)
    return serialised


def export_all(output: Path) -> int:
    settings = get_settings()
    vector_store = get_vector_store(settings)
    vector_store.ensure_ready()

    exported: list[dict[str, Any]] = []
    for payload in vector_store.export_payloads():
        exported.append(_serialise_payload(payload))

    manifest: dict[str, str | None] = {
        "vector_payloads": PAYLOADS_FILE,
        "chat_db": None,
        "memory_db": None,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        payloads_path = root / PAYLOADS_FILE
        payloads_path.write_text(json.dumps(exported, ensure_ascii=False, indent=2))

        db_root = root / DB_DIR
        archive_items: list[tuple[Path, str]] = [(payloads_path, payloads_path.name)]

        chat_db_path = settings.chat_db_path_resolved
        if chat_db_path.exists():
            chat_dest = db_root / chat_db_path.name
            chat_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chat_db_path, chat_dest)
            manifest["chat_db"] = f"{DB_DIR}/{chat_db_path.name}"
            archive_items.append((chat_dest, manifest["chat_db"]))

        memory_db_path = settings.memory_db_path_resolved
        if memory_db_path.exists():
            memory_dest = db_root / memory_db_path.name
            memory_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(memory_db_path, memory_dest)
            manifest["memory_db"] = f"{DB_DIR}/{memory_db_path.name}"
            archive_items.append((memory_dest, manifest["memory_db"]))

        manifest_path = root / MANIFEST_FILE
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        archive_items.append((manifest_path, manifest_path.name))

        with tarfile.open(output, "w:gz") as tar:
            for source, arcname in archive_items:
                tar.add(source, arcname=arcname)
    return len(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("export.tar.gz"),
        help="Путь к tar.gz архиву (по умолчанию export.tar.gz)",
    )
    args = parser.parse_args()

    count = export_all(args.output)
    print(f"Экспортировано записей: {count} -> {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
