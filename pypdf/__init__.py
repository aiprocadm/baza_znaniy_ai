"""Minimal stub of the :mod:`pypdf` API used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class _Page:
    text: str

    def extract_text(self) -> str:
        return self.text


class PdfReader:
    def __init__(self, stream: Iterable[bytes]) -> None:
        data = stream.read() if hasattr(stream, "read") else bytes(stream)
        text = data.decode("utf-8", errors="ignore")
        parts = text.split("\f") or [text]
        self.pages: List[_Page] = [_Page(part) for part in parts if part]
