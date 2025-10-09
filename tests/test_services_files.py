import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.datetime_utils import utc_now
from app.models import file as file_models
from app.models.entities import TenantRecord
from app.models.file import FileRecord as DbFileRecord, FileStatus
from app.services.file_stats import compute_file_stats
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


def test_compute_file_stats_handles_mixed_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "ingest.db"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")
    file_models.get_engine.cache_clear()

    try:
        engine = file_models.get_engine()
        with Session(engine) as session:
            session.add(TenantRecord(tenant_id="tenant", slug="tenant"))
            session.commit()

            base_time = utc_now() - timedelta(hours=3)
            expected_oldest = base_time
            expected_newest = base_time + timedelta(hours=2)
            records = [
                DbFileRecord(
                    tenant_id="tenant",
                    sha256="hash-1",
                    path="/tmp/one.txt",
                    filename="one.txt",
                    size=120,
                    status=FileStatus.COMPLETED,
                    chunks=4,
                    created_at=base_time,
                    updated_at=base_time,
                ),
                DbFileRecord(
                    tenant_id="tenant",
                    sha256="hash-2",
                    path="/tmp/two.txt",
                    filename="two.txt",
                    size=80,
                    status=FileStatus.PROCESSING,
                    chunks=None,
                    created_at=base_time + timedelta(hours=1),
                    updated_at=base_time + timedelta(hours=1),
                ),
                DbFileRecord(
                    tenant_id="tenant",
                    sha256="hash-3",
                    path="/tmp/three.txt",
                    filename="three.txt",
                    size=200,
                    status=FileStatus.COMPLETED,
                    chunks=1,
                    created_at=base_time + timedelta(hours=2),
                    updated_at=base_time + timedelta(hours=2),
                ),
            ]
            session.add_all(records)
            session.commit()

            stats = compute_file_stats(session, "tenant")

        assert stats.total_files == 3
        assert stats.total_size_bytes == 400
        assert stats.total_chunks == 5
        assert stats.status_counts[FileStatus.COMPLETED] == 2
        assert stats.status_counts[FileStatus.PROCESSING] == 1
        assert stats.status_counts[FileStatus.QUEUED] == 0
        assert stats.status_counts[FileStatus.FAILED] == 0
        assert stats.average_size_bytes == pytest.approx(400 / 3)
        assert stats.oldest_upload == expected_oldest
        assert stats.newest_upload == expected_newest
    finally:
        file_models.get_engine.cache_clear()
