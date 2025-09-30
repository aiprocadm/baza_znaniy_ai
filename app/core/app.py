"""Application factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.chat.summarizer import ConversationSummarizer
from app.core.config import get_settings
from app.core.services import init_chat_store, init_memory_store
from app.ingest import parse_and_chunk  # ensure package initialised for scripts
        codex/create-llm-provider-package-and-implementations
from app.llm import LLMProvider, get_cached_provider
from app.retriever import get_vector_store

from app.llm import get_llm_client
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
        main


def create_app(provider: LLMProvider | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""

    settings = get_settings()
    application = FastAPI(title="kb")

    chat_store = init_chat_store(settings)
    llm_provider = provider or get_cached_provider(settings)
    vector_store = get_vector_store(settings)
        codex/create-llm-provider-package-and-implementations
    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)

    summarizer = ConversationSummarizer(chat_store, llm_client.generate)
    reranker: CrossEncoderReranker | None = None
    if settings.rerank_enabled:
        reranker = get_reranker()
        main
    memory_store = init_memory_store(settings)

    application.state.settings = settings
    application.state.chat_store = chat_store
    application.state.llm_provider = llm_provider
    application.state.vector_store = vector_store
    application.state.summarizer = summarizer
    application.state.memory_store = memory_store
    application.state.fallback_index: list[dict[str, object]] = []
    application.state.reranker = reranker

    application.include_router(api_router)
    return application


__all__ = ["create_app"]
