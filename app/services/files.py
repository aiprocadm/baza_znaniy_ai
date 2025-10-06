"""In-memory storage of uploaded files and ingestion status."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from app.core.datetime_utils import utc_now

class IngestStatus(str):
    """String-based status enum used in API responses."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FileRecord:
    """Metadata about an uploaded file."""

    id: str
    filename: str
    tenant: str
    path: Path
    size: int
    uploaded_at: datetime = field(default_factory=utc_now)
    status: str = field(default=IngestStatus.PENDING)
    chunks: int = 0
    error: Optional[str] = None


class FileStore:
    """Thread-safe storage for :class:`FileRecord` entries."""

    def __init__(self) -> None:
        self._records: Dict[str, FileRecord] = {}
        self._lock = Lock()

    def add(self, record: FileRecord) -> None:
        with self._lock:
            self._records[record.id] = record

    def get(self, file_id: str) -> Optional[FileRecord]:
        with self._lock:
            return self._records.get(file_id)

    def all(self) -> List[FileRecord]:
        with self._lock:
            return list(self._records.values())

    def remove(self, file_id: str) -> Optional[FileRecord]:
        with self._lock:
            return self._records.pop(file_id, None)

    def update_status(
        self,
        file_id: str,
        *,
        status: str,
        chunks: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Optional[FileRecord]:
        with self._lock:
            record = self._records.get(file_id)
            if record is None:
                return None
            record.status = status
            if chunks is not None:
                record.chunks = chunks
            record.error = error
            return record

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class IngestQueue:
    """Simple FIFO queue tracking pending file identifiers."""

    def __init__(self) -> None:
        self._items: List[str] = []
        self._lock = Lock()

    def enqueue(self, file_id: str) -> None:
        with self._lock:
            if file_id not in self._items:
                self._items.append(file_id)

    def dequeue(self) -> Optional[str]:
        with self._lock:
            return self._items.pop(0) if self._items else None

    def remove(self, file_id: str) -> None:
        with self._lock:
            try:
                self._items.remove(file_id)
            except ValueError:
                pass

    def __contains__(self, file_id: object) -> bool:
        with self._lock:
            return file_id in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def items(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._items)


__all__ = [
    "FileRecord",
    "FileStore",
    "IngestQueue",
    "IngestStatus",
]
