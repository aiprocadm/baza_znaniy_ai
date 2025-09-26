"""Thread-safe document memory helpers used by the API and tests."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List

from app.models import Document, DocumentCreate

LOGGER = logging.getLogger(__name__)

__all__ = ["DocumentMemory", "DocumentCreate"]


class DocumentMemory:
    """Thread-safe document storage with JSON persistence."""

    def __init__(self, storage_path: Path) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._documents: Dict[str, Document] = {}
        self._next_id = 1
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            with self._storage_path.open("r", encoding="utf-8") as handle:
                raw_items = json.load(handle)
        except Exception:  # pragma: no cover - defensive recover
            LOGGER.warning("Failed to load memory storage, starting fresh", exc_info=True)
            self._documents.clear()
            self._next_id = 1
            self._save()
            return

        for item in raw_items or []:
            try:
                document = Document(**item)
            except Exception:  # pragma: no cover - skip corrupt entries
                LOGGER.warning("Skipping invalid document entry: %r", item)
                continue
            self._documents[document.id] = document
            self._update_counter_from_id(document.id)

    def _save(self) -> None:
        data = [doc.model_dump() for doc in self._documents.values()]
        with self._storage_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def _update_counter_from_id(self, identifier: str) -> None:
        match = re.search(r"(\d+)$", identifier)
        if match:
            candidate = int(match.group(1)) + 1
            if candidate > self._next_id:
                self._next_id = candidate

    # ------------------------------------------------------------------
    # Identifier handling
    # ------------------------------------------------------------------
    def _generate_id(self) -> str:
        while True:
            candidate = f"doc-{self._next_id}"
            self._next_id += 1
            if candidate not in self._documents:
                return candidate

    def _normalise_id(self, raw: str | None) -> str:
        if not raw:
            return self._generate_id()
        base = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-_")
        if not base:
            return self._generate_id()
        candidate = base
        counter = 1
        while candidate in self._documents:
            counter += 1
            candidate = f"{base}-{counter}"
        self._update_counter_from_id(candidate)
        return candidate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def all(self) -> List[Document]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: str) -> Document | None:
        with self._lock:
            return self._documents.get(document_id)

    def add(self, payload: DocumentCreate) -> Document:
        with self._lock:
            identifier = self._normalise_id(payload.id)
            document = Document(id=identifier, content=payload.content, tags=list(payload.tags))
            self._documents[identifier] = document
            self._save()
            return document

    def remove(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False
            self._documents.pop(document_id)
            self._save()
            return True

    # Aliases for backwards compatibility with tests
    def all_ids(self) -> Iterable[str]:  # pragma: no cover - legacy helper
        with self._lock:
            return list(self._documents.keys())
