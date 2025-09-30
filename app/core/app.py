"""FastAPI application factory and bootstrap helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.chat.summarizer import ConversationSummarizer
from app.core.config import get_settings
from app.core.services import init_chat_store, init_memory_store
        codex/create-file-upload-progress-ui-with-fastapi
from app.llm import LLMProvider, get_cached_provider
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
from app.services.files import FileStore, IngestQueue
from app.ui import router as ui_router

        codex/update-app.py-and-ingestion-workflow
from app.ingest import parse_and_chunk  # ensure package initialised for scripts
from app.llm import LLMProvider, get_cached_provider
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
from app.services.files import FileStore, IngestQueue

        main

logger = logging.getLogger(__name__)


        codex/create-file-upload-progress-ui-with-fastapi
def _prepare_cors_origins(origins: Sequence[str] | None) -> list[str]:
    if not origins:
        return ["*"]
    cleaned: list[str] = []
    for origin in origins:
        value = (origin or "").strip()
        if value:
            cleaned.append(value)
    return cleaned or ["*"]

from app.llm import LLMProvider, get_cached_provider
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
        main

logger = logging.getLogger(__name__)


def _initialise_reranker(settings) -> CrossEncoderReranker | None:
    """Create a reranker instance when enabled in configuration."""

    if not settings.rerank_enabled:
        return None
    try:  # pragma: no cover - optional dependency initialisation
        return get_reranker()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialise cross-encoder reranker")
        return None

        main

def create_app(provider: LLMProvider | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""

    settings = get_settings()
    application = FastAPI(title="kb")

    cors_origins = _prepare_cors_origins(settings.cors_allow_origins)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
        expose_headers=["*"],
    )

    chat_store = init_chat_store(settings)
    memory_store = init_memory_store(settings)
    llm_provider = provider or get_cached_provider(settings)
    vector_store = get_vector_store(settings)
        codex/create-file-upload-progress-ui-with-fastapi

    reranker: CrossEncoderReranker | None = None
    if settings.rerank_enabled:
        try:
            reranker = get_reranker()
        except Exception:  # pragma: no cover - optional dependency initialisation

        codex/update-app.py-and-ingestion-workflow
    memory_store = init_memory_store(settings)

    reranker: CrossEncoderReranker | None = None
    if settings.rerank_enabled:
        try:  # pragma: no cover - optional dependency initialisation
            reranker = get_reranker()
        except Exception:  # pragma: no cover - defensive log path
        main
            logger.exception("Failed to initialise cross-encoder reranker")
            reranker = None

    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)
        codex/create-file-upload-progress-ui-with-fastapi
    memory_store = init_memory_store(settings)


    file_store = FileStore()
    ingest_queue = IngestQueue()

    min_citations, max_citations = settings.citations_bounds

        main

    reranker = _initialise_reranker(settings)
    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)

        main
    application.state.settings = settings
    application.state.chat_store = chat_store
    application.state.llm_provider = llm_provider
    application.state.vector_store = vector_store
    application.state.memory_store = memory_store
        codex/create-file-upload-progress-ui-with-fastapi
    application.state.file_store = FileStore()
    application.state.ingest_queue = IngestQueue()
    application.state.fallback_index: list[dict[str, object]] = []
    application.state.chat_history_limit = settings.chat_history_limit
    application.state.retrieve_topk = settings.retrieve_topk
    application.state.rerank_topk = settings.rerank_topk
    application.state.min_citations = settings.chat_min_citations
    application.state.max_citations = settings.chat_max_citations
    application.state.chat_summary_trigger = settings.chat_summary_trigger

    application.include_router(ui_router)

        codex/update-app.py-and-ingestion-workflow
    application.state.summarizer = summarizer
    application.state.reranker = reranker
    application.state.file_store = file_store
    application.state.ingest_queue = ingest_queue
    application.state.fallback_index: list[dict[str, object]] = []

    application.state.chat_history_limit = settings.chat_history_limit
    application.state.retrieve_topk = settings.retrieve_topk
    application.state.rerank_topk = settings.rerank_limit
    application.state.min_citations = min_citations
    application.state.max_citations = max_citations
    application.state.rerank_enabled = settings.rerank_enabled
    application.state.chat_summary_trigger = settings.chat_summary_trigger

    application.state.ingest_worker = getattr(application.state, "ingest_worker", None)
    application.state.ingest_worker_task = None

    @application.on_event("startup")
    async def _startup_ingestion_worker() -> None:
        worker = getattr(application.state, "ingest_worker", None)
        if worker is None or not hasattr(worker, "run"):
            return
        try:
            application.state.ingest_worker_task = asyncio.create_task(worker.run())
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to start ingestion worker")
            application.state.ingest_worker_task = None

    @application.on_event("shutdown")
    async def _shutdown_ingestion_worker() -> None:
        worker = getattr(application.state, "ingest_worker", None)
        task = getattr(application.state, "ingest_worker_task", None)
        if worker is not None and hasattr(worker, "stop"):
            try:
                worker.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to signal ingestion worker shutdown")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - normal cancellation path
                pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error while awaiting ingestion worker shutdown")
            finally:
                application.state.ingest_worker_task = None

    application.state.fallback_index: list[dict[str, object]] = []
        main

        main
    application.include_router(api_router)
    return application


__all__ = ["create_app"]
