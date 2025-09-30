"""Async ingestion queue and worker implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlmodel import Session, delete, select

from app.ingest.chunking import _chunk, _get_tokenizer, iter_document_pages
from app.models.file import ChunkRecord, FileRecord, FileStatus, PageRecord, get_engine


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
        queue: Optional[asyncio.Queue[IngestJob]] = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        engine=None,
    ) -> None:
        self.queue: asyncio.Queue[IngestJob] = queue or asyncio.Queue()
        self.max_retries = max(0, int(max_retries))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
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

    async def enqueue(self, tenant_id: str, path: str) -> Optional[IngestJob]:
        """Register a file for ingestion and push it into the queue."""

        sha = self._hash_file(path)
        with Session(self.engine) as session:
            statement = select(FileRecord).where(
                FileRecord.tenant_id == tenant_id, FileRecord.sha256 == sha
            )
            file_obj = session.exec(statement).first()
            if file_obj:
                if file_obj.status in {FileStatus.QUEUED, FileStatus.PROCESSING, FileStatus.COMPLETED}:
                    return None
            if not file_obj:
                file_obj = FileRecord(
                    tenant_id=tenant_id,
                    sha256=sha,
                    path=path,
                    status=FileStatus.QUEUED,
                    retries=0,
                )
                session.add(file_obj)
                session.commit()
                session.refresh(file_obj)
            else:
                file_obj.path = path
                file_obj.status = FileStatus.QUEUED
                file_obj.retries = 0
                file_obj.updated_at = datetime.utcnow()
                session.add(file_obj)
                session.commit()
            job = IngestJob(tenant_id=tenant_id, path=path, sha256=sha, file_id=file_obj.id)
        await self.queue.put(job)
        return job


class IngestWorker:
    """Worker that consumes the queue and stores parsing metadata."""

    def __init__(
        self,
        service: IngestService,
        *,
        embed_batch_size: Optional[int] = None,
    ) -> None:
        self.service = service
        self.embed_batch_size = max(1, int(embed_batch_size or os.getenv("EMBED_BATCH_SIZE", 32)))
        self._tokenizer = _get_tokenizer()
        chunk = int(os.getenv("RAG_CHUNK", "900"))
        overlap = int(os.getenv("RAG_OVERLAP", "140"))
        self.chunk_size = chunk if chunk > 0 else 1
        self.overlap = overlap if overlap > 0 else 0
        if self.overlap >= self.chunk_size:
            self.overlap = max(0, self.chunk_size - 1)
        self._stop = False

    async def run(self) -> None:
        while not self._stop:
            job = await self.service.queue.get()
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
        try:
            await self._ingest_file(job)
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
                else:
                    file_obj.status = FileStatus.FAILED
                    file_obj.retries = job.attempt + 1
                    file_obj.updated_at = datetime.utcnow()
                    file_obj.error = error_message
                session.add(file_obj)
                session.commit()

        if not success:
            await self._handle_failure(job)

    async def _ingest_file(self, job: IngestJob) -> None:
        with open(job.path, "rb") as handle:
            pages = list(iter_document_pages(job.path, handle))

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
                    chunk_counter += 1
                    if chunk_counter % self.embed_batch_size == 0:
                        batch_index += 1
                        session.commit()
                session.commit()

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
