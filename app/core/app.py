"""Application factory."""

from __future__ import annotations

import logging
from typing import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.chat.summarizer import ConversationSummarizer
from app.core.config import get_settings
from app.core.services import init_chat_store, init_memory_store
from app.llm import LLMProvider, get_cached_provider
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
from app.services.files import FileStore, IngestQueue
from app.ui import router as ui_router

logger = logging.getLogger(__name__)


def _prepare_cors_origins(origins: Sequence[str] | None) -> list[str]:
    if not origins:
        return ["*"]
    cleaned: list[str] = []
    for origin in origins:
        value = (origin or "").strip()
        if value:
            cleaned.append(value)
    return cleaned or ["*"]


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
    llm_provider = provider or get_cached_provider(settings)
    vector_store = get_vector_store(settings)

    reranker: CrossEncoderReranker | None = None
    if settings.rerank_enabled:
        try:
            reranker = get_reranker()
        except Exception:  # pragma: no cover - optional dependency initialisation
            logger.exception("Failed to initialise cross-encoder reranker")
            reranker = None

    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)
    memory_store = init_memory_store(settings)

    application.state.settings = settings
    application.state.chat_store = chat_store
    application.state.llm_provider = llm_provider
    application.state.vector_store = vector_store
    application.state.reranker = reranker
    application.state.summarizer = summarizer
    application.state.memory_store = memory_store
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
    application.include_router(api_router)
    return application


__all__ = ["create_app"]
