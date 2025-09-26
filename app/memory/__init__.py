        codex/fix-top_k-to-10-in-vector-search
"""Compatibility re-export for legacy DocumentMemory tests."""

from srv.projects.kb.app.memory import DocumentMemory  # type: ignore F401
from srv.projects.kb.app.models import DocumentCreate  # type: ignore F401

__all__ = ["DocumentMemory", "DocumentCreate"]

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
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        with self._storage_path.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)
        for item in raw_items:
            doc = Document.model_validate(item)
            self._documents[doc.id] = doc
        self._reset_next_id()

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
            if payload.id:
                doc_id = payload.id
                self._update_next_id_from(doc_id)
            else:
                doc_id = self._generate_id()
            document = Document(id=doc_id, content=payload.content, tags=payload.tags)
            self._documents[document.id] = document
            self._persist()
            return document

    def _generate_id(self) -> str:
        doc_id = f"doc-{self._next_id}"
        self._next_id += 1
        return doc_id

    def _update_next_id_from(self, doc_id: str) -> None:
        suffix = self._parse_numeric_suffix(doc_id)
        if suffix is not None and suffix >= self._next_id:
            self._next_id = suffix + 1

    def _reset_next_id(self) -> None:
        max_suffix = 0
        for existing_id in self._documents:
            suffix = self._parse_numeric_suffix(existing_id)
            if suffix is not None and suffix > max_suffix:
                max_suffix = suffix
        self._next_id = max_suffix + 1 if max_suffix >= 1 else 1

    @staticmethod
    def _parse_numeric_suffix(doc_id: str) -> int | None:
        if not doc_id.startswith("doc-"):
            return None
        suffix = doc_id[4:]
        if suffix.isdigit():
            return int(suffix)
        return None

    def bulk_add(self, payloads: Iterable[DocumentCreate]) -> List[Document]:
        return [self.add(item) for item in payloads]

    def remove(self, document_id: str) -> bool:
        with self._lock:
            removed = self._documents.pop(document_id, None) is not None
            if removed:
                self._persist()
            return removed


__all__ = ["DocumentMemory"]
        main
