"""Compatibility helpers for test suite."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List

from app.models import Document, DocumentCreate


class DocumentMemory:
    """Minimal persistent storage used by the legacy tests."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._documents: Dict[str, Document] = {}
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
            doc_id = payload.id or f"doc-{len(self._documents) + 1}"
            document = Document(id=doc_id, content=payload.content, tags=payload.tags)
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


__all__ = ["DocumentMemory"]
