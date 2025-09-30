"""Compatibility wrapper exposing context helpers for tests."""

from app.rag.context import build_context, detokenize, select_citations, tokenize

__all__ = ["build_context", "detokenize", "select_citations", "tokenize"]
