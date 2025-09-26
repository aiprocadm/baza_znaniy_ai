"""Simple persistence for the knowledge base."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List
from uuid import uuid4

from .models import Document, DocumentCreate


class DocumentMemory:
    """Thread-safe in-memory document store with disk persistence."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = storage_path
        self._documents: Dict[str, Document] = {}
        self._lock = RLock()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        with self._storage_path.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)
        for item in raw_items:
            doc = Document.model_validate(item)
            self._documents[doc.id] = doc

    def _persist(self) -> None:
        with self._storage_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [doc.model_dump(mode="json") for doc in self._documents.values()],
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def all(self) -> List[Document]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: str) -> Document | None:
        with self._lock:
            return self._documents.get(document_id)

    def add(self, payload: DocumentCreate) -> Document:
        with self._lock:
            candidate = payload.id or payload.content
            document_id = self._generate_unique_id(candidate)
            document = Document(
                id=document_id,
                content=payload.content,
                tags=payload.tags,
            )
            self._documents[document.id] = document
            self._persist()
            return document

    def bulk_add(self, payloads: Iterable[DocumentCreate]) -> List[Document]:
        return [self.add(item) for item in payloads]

    def remove(self, document_id: str) -> bool:
        with self._lock:
            removed = self._documents.pop(document_id, None) is not None
            if removed:
                self._persist()
            return removed

    def _generate_unique_id(self, source: str | None) -> str:
        """Create a URL-safe identifier that does not collide with existing ones."""

        base = self._slugify(source or "")
        if not base:
            base = uuid4().hex

        candidate = base
        counter = 2
        while candidate in self._documents:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    @staticmethod
    def _slugify(value: str) -> str:
        """Return a lowercase slug limited to URL-safe characters."""

        normalized = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = normalized.replace(" ", "-")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", normalized)
        slug = slug.strip("-_")
        return slug.lower()
