"""Async ingestion queue and worker implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from sqlmodel import Session, delete, select

from app.core.config import get_settings
from app.ingest.chunking import _chunk, _get_tokenizer, iter_document_pages
from app.models.file import ChunkRecord, FileRecord, FileStatus, PageRecord, get_engine
from app.services.vectorstore import index_chunks


@dataclass
class IngestJob:
    """Descriptor for a queued ingestion task."""

    tenant_id: str
    path: str
    sha256: str
    file_id: int
    attempt: int = 0


class IngestService:
    """Service responsible for computing file hashes and enqueueing jobs."""

    def __init__(
        self,
        *,
        queue: Optional[asyncio.Queue[Optional[IngestJob]]] = None,
        max_retries: Optional[int] = None,
        backoff_seconds: Optional[float] = None,
        engine=None,
    ) -> None:
        settings = get_settings()
        retries = settings.ingest_max_retries if max_retries is None else max_retries
        backoff = (
            settings.ingest_backoff_seconds if backoff_seconds is None else backoff_seconds
        )
        self.queue: asyncio.Queue[Optional[IngestJob]] = queue or asyncio.Queue()
        self.max_retries = max(0, int(retries))
        self.backoff_seconds = max(0.0, float(backoff))
        self._engine = engine

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    @staticmethod
    def _hash_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _make_job(self, file_obj: FileRecord, *, attempt: int = 0) -> IngestJob:
        if file_obj.id is None:
            raise ValueError("FileRecord must be persisted before enqueueing")
        return IngestJob(
            tenant_id=file_obj.tenant_id,
            path=file_obj.path,
            sha256=file_obj.sha256,
            file_id=file_obj.id,
            attempt=attempt,
        )

    async def enqueue_job(self, file_obj: FileRecord, *, attempt: int = 0) -> IngestJob:
        job = self._make_job(file_obj, attempt=attempt)
        await self.queue.put(job)
        return job

    async def register_file(
        self,
        tenant_id: str,
        path: str,
        *,
        filename: str,
        size: int,
    ) -> Tuple[FileRecord, bool]:
        """Register a file upload and enqueue ingestion when necessary."""

        sha = self._hash_file(path)
        queued = False
        with Session(self.engine) as session:
            statement = select(FileRecord).where(
                FileRecord.tenant_id == tenant_id, FileRecord.sha256 == sha
            )
            file_obj = session.exec(statement).first()
            if file_obj is None:
                file_obj = FileRecord(
                    tenant_id=tenant_id,
                    sha256=sha,
                    path=path,
                    filename=filename,
                    size=size,
                    status=FileStatus.QUEUED,
                    retries=0,
                    error=None,
                    chunks=None,
                )
                session.add(file_obj)
                session.commit()
                session.refresh(file_obj)
                queued = True
            else:
                file_obj.updated_at = datetime.utcnow()
                if file_obj.status == FileStatus.FAILED:
                    file_obj.path = path
                    file_obj.filename = filename
                    file_obj.size = size
                    file_obj.status = FileStatus.QUEUED
                    file_obj.retries = 0
                    file_obj.error = None
                    file_obj.chunks = None
                    queued = True
                session.add(file_obj)
                session.commit()
                session.refresh(file_obj)

        if queued:
            await self.enqueue_job(file_obj)
        return file_obj, queued


class IngestWorker:
    """Worker that consumes the queue and stores parsing metadata."""

    def __init__(
        self,
        service: IngestService,
        *,
        embed_batch_size: Optional[int] = None,
    ) -> None:
        self.service = service
        settings = get_settings()
        default_batch = settings.embed_batch_size
        self.embed_batch_size = max(1, int(embed_batch_size or default_batch))
        self._tokenizer = _get_tokenizer()
        chunk = int(os.getenv("RAG_CHUNK", "900"))
        overlap = int(os.getenv("RAG_OVERLAP", "140"))
        self.chunk_size = chunk if chunk > 0 else 1
        self.overlap = overlap if overlap > 0 else 0
        if self.overlap >= self.chunk_size:
            self.overlap = max(0, self.chunk_size - 1)
        self._stop = False

    async def run(self) -> None:
        while True:
            if self._stop and self.service.queue.empty():
                break
            try:
                job = await self.service.queue.get()
            except asyncio.CancelledError:  # pragma: no cover - shutdown safety
                break
            if job is None:
                self.service.queue.task_done()
                break
            try:
                await self._process(job)
            except Exception:  # pragma: no cover - defensive
                await self._handle_failure(job)
            finally:
                self.service.queue.task_done()

    async def _process(self, job: IngestJob) -> None:
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            if not file_obj:
                return
            if file_obj.status == FileStatus.COMPLETED:
                return
            file_obj.status = FileStatus.PROCESSING
            file_obj.error = None
            file_obj.chunks = None
            file_obj.updated_at = datetime.utcnow()
            session.add(file_obj)
            session.commit()

            page_ids = session.exec(
                select(PageRecord.id).where(PageRecord.file_id == file_obj.id)
            ).all()
            if page_ids:
                session.exec(delete(ChunkRecord).where(ChunkRecord.page_id.in_(page_ids)))
                session.exec(delete(PageRecord).where(PageRecord.id.in_(page_ids)))
                session.commit()

        success = True
        error_message: Optional[str] = None
        chunk_count = 0
        try:
            chunk_count = await self._ingest_file(job)
        except Exception as exc:
            success = False
            error_message = str(exc)
        finally:
            with Session(self.service.engine) as session:
                file_obj = session.get(FileRecord, job.file_id)
                if not file_obj:
                    return
                if success:
                    file_obj.status = FileStatus.COMPLETED
                    file_obj.updated_at = datetime.utcnow()
                    file_obj.error = None
                    file_obj.chunks = chunk_count
                else:
                    file_obj.status = FileStatus.FAILED
                    file_obj.retries = job.attempt + 1
                    file_obj.updated_at = datetime.utcnow()
                    file_obj.error = error_message
                    file_obj.chunks = chunk_count or 0
                session.add(file_obj)
                session.commit()

        if not success:
            await self._handle_failure(job)

    async def _ingest_file(self, job: IngestJob) -> int:
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            if not file_obj:
                return 0
            filename = file_obj.filename

        with open(job.path, "rb") as handle:
            pages = list(iter_document_pages(job.path, handle))

        chunk_payloads: list[dict[str, object]] = []
        with Session(self.service.engine) as session:
            batch_index = 0
            chunk_counter = 0
            for page_number, text in pages:
                page_sha = hashlib.sha256(
                    f"{job.sha256}:{page_number}:{len(text)}".encode("utf-8")
                ).hexdigest()
                page = PageRecord(
                    file_id=job.file_id,
                    number=page_number,
                    sha256=page_sha,
                    text=text,
                )
                session.add(page)
                session.commit()
                session.refresh(page)

                chunks = _chunk(
                    text,
                    chunk=self.chunk_size,
                    overlap=self.overlap,
                    encoder=self._tokenizer,
                )
                for offset, chunk_text in enumerate(chunks, start=1):
                    chunk_sha = hashlib.sha256(
                        f"{job.sha256}:{page.number}:{offset}:{chunk_text}".encode("utf-8")
                    ).hexdigest()
                    chunk = ChunkRecord(
                        page_id=page.id,
                        index=offset,
                        sha256=chunk_sha,
                        text=chunk_text,
                        batch=batch_index,
                    )
                    session.add(chunk)
                    chunk_payloads.append(
                        {
                            "file": filename,
                            "page": page.number,
                            "sha256": chunk_sha,
                            "text": chunk_text,
                        }
                    )
                    chunk_counter += 1
                    if chunk_counter % self.embed_batch_size == 0:
                        batch_index += 1
                        session.commit()
                session.commit()

        index_chunks(chunk_payloads)
        return len(chunk_payloads)

    async def _handle_failure(self, job: IngestJob) -> None:
        job.attempt += 1
        if job.attempt > self.service.max_retries:
            return
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            if file_obj:
                file_obj.status = FileStatus.QUEUED
                file_obj.retries = job.attempt
                file_obj.updated_at = datetime.utcnow()
                session.add(file_obj)
                session.commit()
        delay = self.service.backoff_seconds * (2 ** (job.attempt - 1))
        if delay:
            await asyncio.sleep(delay)
        await self.service.queue.put(job)

    def stop(self) -> None:
        self._stop = True


__all__ = ["IngestJob", "IngestService", "IngestWorker"]
