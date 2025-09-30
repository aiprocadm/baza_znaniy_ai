"""Application factory."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.chat.summarizer import ConversationSummarizer
from app.core.config import get_settings
from app.core.services import init_chat_store, init_memory_store
from app.ingest import (  # ensure package initialised for scripts
    IngestService,
    IngestWorker,
    parse_and_chunk,
)
        # codex/implement-reranking-functionality-and-tests
from app.llm import get_llm_client
from app.retriever import CrossEncoderReranker, get_vector_store


logger = logging.getLogger(__name__)

        # codex/create-llm-provider-package-and-implementations
from app.llm import LLMProvider, get_cached_provider
from app.retriever import get_vector_store
        # main

from app.llm import get_llm_client
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
        # main


def create_app(provider: LLMProvider | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""

    settings = get_settings()
    application = FastAPI(title="kb")

    chat_store = init_chat_store(settings)
    llm_provider = provider or get_cached_provider(settings)
    vector_store = get_vector_store(settings)
        # codex/implement-reranking-functionality-and-tests
    reranker = None
    if settings.rerank_enabled:
        try:
            reranker = CrossEncoderReranker()
        except Exception:  # pragma: no cover - optional dependency initialisation
            logger.exception("Failed to initialise cross-encoder reranker")

        # codex/create-llm-provider-package-and-implementations
    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)

        # main
    summarizer = ConversationSummarizer(chat_store, llm_client.generate)
    reranker: CrossEncoderReranker | None = None
    if settings.rerank_enabled:
        reranker = get_reranker()
        # main
    memory_store = init_memory_store(settings)

    ingest_service = IngestService(
        max_retries=settings.ingest_max_retries,
        backoff_seconds=settings.ingest_backoff_seconds,
    )
    ingest_worker = IngestWorker(ingest_service)

    application.state.settings = settings
    application.state.chat_store = chat_store
    application.state.llm_provider = llm_provider
    application.state.vector_store = vector_store
    application.state.reranker = reranker
    application.state.summarizer = summarizer
    application.state.memory_store = memory_store
    application.state.ingest_service = ingest_service
    application.state.ingest_worker = ingest_worker
    application.state.ingest_worker_task = None
    application.state.fallback_index: list[dict[str, object]] = []
    application.state.reranker = reranker

    @application.on_event("startup")
    async def _start_ingest_worker() -> None:
        task = asyncio.create_task(ingest_worker.run())
        application.state.ingest_worker_task = task

    @application.on_event("shutdown")
    async def _stop_ingest_worker() -> None:
        ingest_worker.stop()
        await ingest_service.queue.put(None)
        task = application.state.ingest_worker_task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:  # pragma: no cover - shutdown race
                pass

    application.include_router(api_router)
    return application


__all__ = ["create_app"]
