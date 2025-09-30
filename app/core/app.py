"""FastAPI application factory and bootstrap helpers."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.chat.summarizer import ConversationSummarizer
from app.core.config import get_settings
from app.core.services import init_chat_store, init_memory_store
from app.llm import LLMProvider, get_cached_provider
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store

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


def create_app(provider: LLMProvider | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""

    settings = get_settings()
    application = FastAPI(title="kb")

    chat_store = init_chat_store(settings)
    memory_store = init_memory_store(settings)
    llm_provider = provider or get_cached_provider(settings)
    vector_store = get_vector_store(settings)
    reranker = _initialise_reranker(settings)
    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)

    application.state.settings = settings
    application.state.chat_store = chat_store
    application.state.llm_provider = llm_provider
    application.state.vector_store = vector_store
    application.state.reranker = reranker
    application.state.summarizer = summarizer
    application.state.memory_store = memory_store
    application.state.fallback_index: list[dict[str, object]] = []

    application.include_router(api_router)
    return application


__all__ = ["create_app"]
