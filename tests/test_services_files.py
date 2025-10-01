import uuid
from pathlib import Path

import pytest

from app.services.files import FileRecord, FileStore, IngestQueue, IngestStatus


def test_file_store_operations(tmp_path: Path) -> None:
    store = FileStore()
    record_id = str(uuid.uuid4())
    record_path = tmp_path / "example.txt"
    record_path.write_text("example content")

    record = FileRecord(
        id=record_id,
        filename="example.txt",
        tenant="tenant-1",
        path=record_path,
        size=record_path.stat().st_size,
    )

    store.add(record)

    # Retrieval should return the same record instance
    assert store.get(record_id) is record
    assert store.all() == [record]

    # Status updates should mutate the stored record and allow overrides
    updated = store.update_status(record_id, status=IngestStatus.PROCESSING, chunks=5)
    assert updated is record
    assert record.status == IngestStatus.PROCESSING
    assert record.chunks == 5
    assert record.error is None

    updated = store.update_status(
        record_id,
        status=IngestStatus.FAILED,
        chunks=8,
        error="ingest failed",
    )
    assert updated is record
    assert record.status == IngestStatus.FAILED
    assert record.chunks == 8
    assert record.error == "ingest failed"

    # Removing should return the record and empty the store
    removed = store.remove(record_id)
    assert removed is record
    assert store.get(record_id) is None
    assert store.all() == []

    # Clearing should remove all entries
    second = FileRecord(
        id=str(uuid.uuid4()),
        filename="second.txt",
        tenant="tenant-1",
        path=record_path,
        size=0,
    )
    third = FileRecord(
        id=str(uuid.uuid4()),
        filename="third.txt",
        tenant="tenant-2",
        path=record_path,
        size=0,
    )
    store.add(second)
    store.add(third)
    assert len(store.all()) == 2
    store.clear()
    assert store.all() == []


def test_ingest_queue_behaviour() -> None:
    queue = IngestQueue()

    queue.enqueue("file1")
    queue.enqueue("file2")
    queue.enqueue("file1")  # should be deduplicated

    assert len(queue) == 2
    assert "file1" in queue
    assert queue.items() == ("file1", "file2")

    # FIFO order
    assert queue.dequeue() == "file1"
    assert len(queue) == 1

    queue.enqueue("file3")

    # Remove should be safe even if the item exists or not
    queue.remove("file2")
    queue.remove("missing")

    assert "file2" not in queue
    assert len(queue) == 1
    assert queue.items() == ("file3",)

    assert queue.dequeue() == "file3"
    assert queue.dequeue() is None
    assert len(queue) == 0
    assert queue.items() == ()
