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


def test_register_creates_file_and_prevents_duplicates(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=1, backoff_seconds=0)
        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
        )
        assert queued is True
        assert record.status == file_models.FileStatus.QUEUED

        duplicate, again = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
        )
        assert again is False
        assert duplicate.id == record.id

        with Session(service.engine) as session:
            files = session.exec(select(FileRecord)).all()
            assert len(files) == 1
            assert files[0].status == file_models.FileStatus.QUEUED

    asyncio.run(scenario())


def test_worker_processes_file_and_creates_pages(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=1, backoff_seconds=0)
        worker = IngestWorker(service, embed_batch_size=2)

        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
        )
        assert queued is True

        job = await service.queue.get()
        await worker._process(job)
        service.queue.task_done()

        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == record.id)).one()
            assert file_obj.status == file_models.FileStatus.COMPLETED
            assert file_obj.chunks > 0
            pages = session.exec(select(PageRecord).where(PageRecord.file_id == file_obj.id)).all()
            assert pages
            chunks = session.exec(
                select(ChunkRecord).where(ChunkRecord.page_id.in_([p.id for p in pages]))
            ).all()
            assert chunks

        # Re-register should not enqueue once completed
        _, queued_again = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
        )
        assert queued_again is False

    asyncio.run(scenario())


def test_worker_retries_failed_jobs(
    monkeypatch: pytest.MonkeyPatch, sqlite_db: str, sample_file: Path
) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=2, backoff_seconds=0)
        worker = IngestWorker(service)

        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
        )
        assert queued is True

        async def failing_ingest(self, _job):
            raise RuntimeError("boom")

        monkeypatch.setattr(IngestWorker, "_ingest_file", failing_ingest)

        job = await service.queue.get()
        await worker._process(job)
        service.queue.task_done()

        requeued_job = await service.queue.get()
        service.queue.task_done()

        assert requeued_job.attempt == 1

        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == record.id)).one()
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

        record_one, queued_one = await service.register_file(
            "tenant",
            str(first),
            filename=first.name,
            size=first.stat().st_size,
        )
        record_two, queued_two = await service.register_file(
            "tenant",
            str(second),
            filename=second.name,
            size=second.stat().st_size,
        )

        assert queued_one is True
        assert queued_two is False
        assert record_two.id == record_one.id

        digest = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.sha256 == digest)).one()
            assert Path(file_obj.path).name == Path(first).name

    asyncio.run(scenario())


def test_service_uses_settings_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_MAX_RETRIES", "5")
    monkeypatch.setenv("INGEST_BACKOFF_SECONDS", "2.5")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    service = IngestService()

    assert service.max_retries == 5
    assert service.backoff_seconds == 2.5

    config_module.get_settings.cache_clear()
