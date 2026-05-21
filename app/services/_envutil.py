"""Tiny env-reading helper shared by MVP services.

Centralised so ``app/services/kb_llm.py`` and ``app/services/kb_embeddings.py``
both apply the same null/empty-string convention to environment variables.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional


def env(name: str, source: Mapping[str, str] | None = None) -> Optional[str]:
    """Return ``source[name]`` (or ``os.environ[name]``) stripped, or ``None``.

    Treats both missing keys and whitespace-only values as ``None`` so call
    sites can write ``if env("DEEPSEEK_API_KEY"):`` without separate
    "is empty" guards.
    """

    raw = os.environ.get(name) if source is None else source.get(name)
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


__all__ = ["env"]
