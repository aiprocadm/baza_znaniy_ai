"""Compose a RAG-aware SFT dataset from a KB corpus + teacher LLM.

This module is the pure-logic core of Workstream 3 (RAG-aware
fine-tuning) in the Pack B++ ML strengthening plan. It builds on
W1's :mod:`app.services.synthetic_qa` for seed Q&A generation and
on :class:`app.services.kb_store.KnowledgeBaseStore` for retrieval.

The module is intentionally I/O free: provider, retriever, and chunk
source are injected so the logic is deterministic in tests.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)
