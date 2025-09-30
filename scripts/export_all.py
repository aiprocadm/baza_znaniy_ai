#!/usr/bin/env python3
"""Export all vector payloads to a JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.retriever import get_vector_store


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
    vector_store.ensure_collection()

    exported: list[dict[str, Any]] = []
    for payload in vector_store.export_payloads():
        exported.append(_serialise_payload(payload))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(exported, ensure_ascii=False, indent=2))
    return len(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("export.json"),
        help="Путь к JSON-файлу (по умолчанию export.json)",
    )
    args = parser.parse_args()

    count = export_all(args.output)
    print(f"Экспортировано записей: {count} -> {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
