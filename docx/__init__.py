"""Very small stub of the python-docx API used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Paragraph:
    text: str


class Document:
    def __init__(self, stream: Iterable[bytes]) -> None:
        data = stream.read() if hasattr(stream, "read") else bytes(stream)
        text = data.decode("utf-8", errors="ignore")
        self.paragraphs = [Paragraph(line) for line in text.splitlines() if line]
