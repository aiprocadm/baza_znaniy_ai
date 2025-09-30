        codex/create-sqlmodel-models-for-files-and-pages
"""Async ingestion queue and worker implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlmodel import Session, delete, select

from app.core.config import get_settings
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

"""Document ingestion helpers for parsing and chunking content."""

from __future__ import annotations

import hashlib
import io
import logging
import re
from functools import lru_cache
from typing import Iterable, List, NamedTuple, Optional, Protocol

from docx import Document
from pypdf import PdfReader

from app.core.config import get_settings

try:  # pragma: no cover - tokenizer optional in some environments
    import tiktoken
except ImportError:  # pragma: no cover - fallback used in tests
    tiktoken = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class _Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: List[int]) -> str:
        ...


class _CharTokenizer:
    """Fallback tokenizer operating on raw characters."""

    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(token) for token in tokens)


_TOKENIZER: Optional[_Tokenizer] = None


def _load_tiktoken(name: str) -> Optional[_Tokenizer]:
    if tiktoken is None:  # pragma: no cover - handled during tests
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to load tokenizer '%s': %s", name, exc)
        try:
            return tiktoken.encoding_for_model(name)
        except Exception:  # pragma: no cover - final fallback
            return None


@lru_cache(maxsize=1)
def _default_tokenizer() -> _Tokenizer:
    settings = get_settings()
    name = settings.rag_tokenizer_name
    tokenizer = _load_tiktoken(name)
    if tokenizer is None and name != "text-embedding-3-small":
        tokenizer = _load_tiktoken("text-embedding-3-small")
    return tokenizer or _CharTokenizer()


def _get_tokenizer() -> _Tokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _default_tokenizer()
    return _TOKENIZER


def _normalise_window_size(value: int, minimum: int = 1) -> int:
    value = int(value)
    return minimum if value < minimum else value


def _normalise_overlap(chunk: int, overlap: int) -> int:
    overlap = 0 if overlap < 0 else int(overlap)
    if chunk <= 1:
        return 0
    return min(overlap, chunk - 1)


def _clean(text: str) -> str:
    """Collapse whitespace and trim the provided *text*."""

    return re.sub(r"\s+", " ", text).strip()


class _WindowPlan(NamedTuple):
    token_ids: List[int]
    tokenizer: _Tokenizer


def _iterate_windows(
    token_ids: List[int], *, window: int, overlap: int, tokenizer: _Tokenizer
) -> List[str]:
    total = len(token_ids)
    if total == 0:
        return []

    step_overlap = _normalise_overlap(window, overlap)
    pieces: List[str] = []
    start = 0
    while start < total:
        end = min(start + window, total)
        pieces.append(tokenizer.decode(token_ids[start:end]))
        if end >= total:
            break
        next_start = end - step_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return pieces


def _handle_small_token_window(
    text: str,
    token_ids: List[int],
    *,
    window: int,
    overlap: int,
    tokenizer: _Tokenizer,
) -> Optional[_WindowPlan]:
    decoded_text = tokenizer.decode(token_ids)

    if window <= 1:
        fallback_text = decoded_text or text
        char_tokenizer = _CharTokenizer()
        char_token_ids = (
            char_tokenizer.encode(fallback_text) if fallback_text else []
        )
        return _WindowPlan(char_token_ids, char_tokenizer)

    if len(token_ids) > window:
        return None

    if decoded_text and len(decoded_text) > window:
        char_tokenizer = _CharTokenizer()
        char_token_ids = char_tokenizer.encode(decoded_text)
        return _WindowPlan(char_token_ids, char_tokenizer)

    if len(token_ids) == 1:
        fallback_text = decoded_text or text
        if fallback_text:
            char_tokenizer = _CharTokenizer()
            char_token_ids = char_tokenizer.encode(fallback_text)
            if char_token_ids:
                return _WindowPlan(char_token_ids, char_tokenizer)
        return _WindowPlan(token_ids, tokenizer)

    if decoded_text:
        try:
            reencoded = tokenizer.encode(decoded_text)
        except Exception:  # pragma: no cover - defensive fallback
            reencoded = []
        if reencoded:
            if len(reencoded) <= window and len(decoded_text) <= window:
                if reencoded == token_ids:
                    return _WindowPlan(token_ids, tokenizer)
                return _WindowPlan(reencoded, tokenizer)
            fallback_text = decoded_text
        else:
            fallback_text = decoded_text
    else:
        fallback_text = text

    if not fallback_text:
        return _WindowPlan(token_ids, tokenizer)
    char_tokenizer = _CharTokenizer()
    char_token_ids = char_tokenizer.encode(fallback_text)
    if not char_token_ids:
        return _WindowPlan(token_ids, tokenizer)

    return _WindowPlan(char_token_ids, char_tokenizer)


def _chunk(
    text: str,
    *,
    chunk: int = 900,
    overlap: int = 140,
    encoder: Optional[_Tokenizer] = None,
    token_ids: Optional[List[int]] = None,
) -> List[str]:
    """Split ``text`` into overlapping windows based on token counts."""

    if not text:
        return []

    window = _normalise_window_size(chunk)

    tokenizer = encoder or _get_tokenizer()
    working_token_ids = list(token_ids) if token_ids is not None else tokenizer.encode(text)
    if not working_token_ids:
        return []

    decoded_text: str = ""
    try:
        decoded_text = tokenizer.decode(working_token_ids)
    except Exception:  # pragma: no cover - defensive fallback
        decoded_text = ""

    if decoded_text:
        try:
            reencoded = tokenizer.encode(decoded_text)
        except Exception:  # pragma: no cover - defensive fallback
            reencoded = []
        if reencoded:
            use_char_tokenizer = len(reencoded) < len(working_token_ids) or len(
                reencoded
            ) >= window
            if use_char_tokenizer:
                char_tokenizer = _CharTokenizer()
                char_ids = char_tokenizer.encode(decoded_text)
                if char_ids:
                    working_token_ids = char_ids
                    tokenizer = char_tokenizer
    token_ids = working_token_ids

    small_window_plan = _handle_small_token_window(
        text,
        token_ids,
        window=window,
        overlap=overlap,
        tokenizer=tokenizer,
    )
    if small_window_plan is not None:
        token_ids, tokenizer = small_window_plan

    return _iterate_windows(
        token_ids,
        window=window,
        overlap=overlap,
        tokenizer=tokenizer,
    )


def _iter_pdf_text(data: bytes) -> Iterable[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - pypdf quirks
            text = ""
        cleaned = _clean(text)
        if cleaned:
            yield page_number, cleaned


def _iter_docx_text(data: bytes) -> Iterable[tuple[int, str]]:
    document = Document(io.BytesIO(data))
    text = _clean("\n".join(paragraph.text for paragraph in document.paragraphs))
    if text:
        yield 1, text


def _iter_txt_text(data: bytes) -> Iterable[tuple[int, str]]:
    text = _clean(data.decode("utf-8", errors="ignore"))
    if text:
        yield 1, text


def _hash_chunk(file: str, page: int, text: str) -> str:
    payload = f"{file}\u0000{page}\u0000{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_and_chunk(filename: str, data: bytes) -> List[dict[str, object]]:
    """Parse ``data`` according to ``filename`` extension and chunk the text."""

    name = (filename or "").strip()
    if not name or "." not in name:
        return []

    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        pages = list(_iter_pdf_text(data))
    elif ext == "docx":
        pages = list(_iter_docx_text(data))
    elif ext == "txt":
        pages = list(_iter_txt_text(data))
    else:
        return []

    if not pages:
        return []

    settings = get_settings()
    chunk_size = _normalise_window_size(settings.rag_chunk)
    overlap = _normalise_overlap(chunk_size, settings.rag_overlap)
    tokenizer = _get_tokenizer()

    chunks: List[dict[str, object]] = []
    for page_number, page_text in pages:
        for piece in _chunk(page_text, chunk=chunk_size, overlap=overlap, encoder=tokenizer):
            sha = _hash_chunk(name, page_number, piece)
            chunks.append(
                {
                    "file": name,
                    "page": page_number,
                    "sha256": sha,
                    "text": piece,
                }
            )
    return chunks


__all__ = ["_chunk", "_clean", "_get_tokenizer", "parse_and_chunk"]
        main
