        codex/refactor-modules-to-remove-codex-markers
"""Compatibility helpers for the legacy document memory tests."""

"""Compatibility helpers for test suite."""
        main

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List

from app.models import Document, DocumentCreate

        codex/refactor-modules-to-remove-codex-markers
LOGGER = logging.getLogger(__name__)

__all__ = ["DocumentMemory"]
        main


class DocumentMemory:
    """Thread-safe document storage with JSON persistence."""

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
        try:
            with self._storage_path.open("r", encoding="utf-8") as handle:
                raw_items = json.load(handle)
        codex/refactor-modules-to-remove-codex-markers
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Resetting invalid document storage %s: %s", self._storage_path, exc)

        except (json.JSONDecodeError, OSError):
        main
            self._documents.clear()
            self._persist()
            return
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
                document_id = payload.id
                self._update_next_id_from(document_id)
            else:
                document_id = self._generate_id()
            document = Document(id=document_id, content=payload.content, tags=payload.tags)
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

    def _generate_id(self) -> str:
        document_id = f"doc-{self._next_id}"
        self._next_id += 1
        return document_id

    def _update_next_id_from(self, document_id: str) -> None:
        suffix = self._parse_numeric_suffix(document_id)
        if suffix is not None and suffix >= self._next_id:
            self._next_id = suffix + 1

    def _reset_next_id(self) -> None:
        max_suffix = 0
        for doc_id in self._documents:
            suffix = self._parse_numeric_suffix(doc_id)
            if suffix is not None and suffix > max_suffix:
                max_suffix = suffix
        self._next_id = max_suffix + 1 if max_suffix >= 1 else 1

    @staticmethod
    def _parse_numeric_suffix(document_id: str) -> int | None:
        if not document_id.startswith("doc-"):
            return None
        suffix = document_id[4:]
        if suffix.isdigit():
            return int(suffix)
        return None
        codex/refactor-modules-to-remove-codex-markers


__all__ = ["DocumentMemory", "DocumentCreate"]

        main
