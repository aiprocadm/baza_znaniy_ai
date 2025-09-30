"""Tests for the ingestion queue and worker."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.ingest.service import IngestService, IngestWorker
from app.models import file as file_models
from app.models.file import ChunkRecord, FileRecord, PageRecord


@pytest.fixture
def sqlite_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "ingest.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DB_URL", db_url)
    file_models.get_engine.cache_clear()
    yield db_url
    file_models.get_engine.cache_clear()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "document.txt"
    file_path.write_text("First page\nSecond page\nThird line", encoding="utf-8")
    return file_path


def test_enqueue_creates_file_and_prevents_duplicates(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=1, backoff_seconds=0)
        job = await service.enqueue("tenant", str(sample_file))
        assert job is not None

        duplicate = await service.enqueue("tenant", str(sample_file))
        assert duplicate is None

        with Session(service.engine) as session:
            files = session.exec(select(FileRecord)).all()
            assert len(files) == 1
            assert files[0].status == file_models.FileStatus.QUEUED

    asyncio.run(scenario())


def test_worker_processes_file_and_creates_pages(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=1, backoff_seconds=0)
        worker = IngestWorker(service, embed_batch_size=2)

        job = await service.enqueue("tenant", str(sample_file))
        assert job is not None

        await worker._process(job)

        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == job.file_id)).one()
            assert file_obj.status == file_models.FileStatus.COMPLETED
            pages = session.exec(select(PageRecord).where(PageRecord.file_id == file_obj.id)).all()
            assert pages
            chunks = session.exec(
                select(ChunkRecord).where(ChunkRecord.page_id.in_([p.id for p in pages]))
            ).all()
            assert chunks

        # Re-enqueue should be ignored once completed
        assert await service.enqueue("tenant", str(sample_file)) is None

    asyncio.run(scenario())


def test_worker_retries_failed_jobs(
    monkeypatch: pytest.MonkeyPatch, sqlite_db: str, sample_file: Path
) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=2, backoff_seconds=0)
        worker = IngestWorker(service)

        job = await service.enqueue("tenant", str(sample_file))
        assert job is not None

        async def failing_ingest(self, _job):
            raise RuntimeError("boom")

        monkeypatch.setattr(IngestWorker, "_ingest_file", failing_ingest)

        await worker._process(job)

        requeued_job = await service.queue.get()
        service.queue.task_done()

        assert requeued_job.attempt == 1

        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == job.file_id)).one()
            assert file_obj.status == file_models.FileStatus.QUEUED
            assert file_obj.retries == 1
            assert file_obj.error == "boom"

    asyncio.run(scenario())


def test_hash_is_based_on_contents(sqlite_db: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text("hello", encoding="utf-8")
        second.write_text("hello", encoding="utf-8")

        service = IngestService(max_retries=0, backoff_seconds=0)

        job_one = await service.enqueue("tenant", str(first))
        job_two = await service.enqueue("tenant", str(second))

        assert job_one is not None
        assert job_two is None

        digest = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.sha256 == digest)).one()
            assert file_obj.path == str(first)

    asyncio.run(scenario())
