#!/usr/bin/env python3
"""Import vector payloads from a JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.retriever import get_vector_store


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(payload)
    vector = normalised.get("vector")
    if vector is not None and hasattr(vector, "tolist"):
        normalised["vector"] = list(vector.tolist())
    elif isinstance(vector, tuple):
        normalised["vector"] = list(vector)
    return normalised


def import_all(path: Path, *, reset: bool = False) -> int:
    settings = get_settings()
    vector_store = get_vector_store(settings)
    if reset:
        vector_store.reset_collection()
    else:
        vector_store.ensure_collection()

    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Ожидается список записей в JSON")

    payloads = [_normalise_payload(item) for item in data if isinstance(item, dict)]
    vector_store.import_payloads(payloads)
    return len(payloads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Путь к JSON-файлу экспорта")
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
