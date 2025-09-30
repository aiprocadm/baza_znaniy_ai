#!/usr/bin/env python3
"""Drop and rebuild the vector index from disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.retriever import get_vector_store

from .ingest_path import ingest_path


def rebuild_index(path: Path | None = None) -> int:
    settings = get_settings()
    target = path or Path(settings.data_dir)
    vector_store = get_vector_store(settings)
    vector_store.reset_collection()
    return ingest_path(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Каталог с документами (по умолчанию DATA_DIR)",
    )
    args = parser.parse_args()

    total = rebuild_index(args.path)
    print(f"Переиндексировано чанков: {total}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
