"""Generate synthetic supervised Q&A datasets from a KB corpus.

This module is the pure-logic core of Workstream 1 (Synthetic Data
Generation) in the Pack B++ ML strengthening plan. A teacher LLM is
prompted with document chunks and asked to produce diverse Q&A pairs.
The CLI wrapper is in ``scripts/generate_synthetic_qa.py``.

The module is intentionally I/O free: all dependencies (LLM provider,
chunk source) are injected, making the logic deterministic in tests.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

__all__: list[str] = []
