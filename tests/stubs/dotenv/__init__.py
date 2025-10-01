"""Minimal stub for python-dotenv used in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dotenv(*_: Any, **__: Any) -> bool:
    return False


def dotenv_values(path: str | Path | None = None, *_: Any, **__: Any) -> dict[str, str]:
    if path is None:
        raise ValueError("path must be provided in test stub")

    file_path = Path(path)
    data: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data
