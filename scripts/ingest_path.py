#!/usr/bin/env python3
"""CLI helper to ingest documents from a local path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from app.core.config import get_settings
from app.ingest import parse_and_chunk
from app.retriever import get_vector_store

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _iter_documents(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
    elif root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def ingest_path(path: Path) -> int:
    settings = get_settings()
    vector_store = get_vector_store(settings)
    vector_store.ensure_ready()

    total_chunks = 0
    for document in _iter_documents(path):
        chunks = parse_and_chunk(document.name, document.read_bytes())
        if not chunks:
            continue
        vector_store.upsert(chunks)
        total_chunks += len(chunks)
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Файл или каталог с документами")
    args = parser.parse_args()

    total = ingest_path(args.path)
    print(f"Индексировано чанков: {total}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
