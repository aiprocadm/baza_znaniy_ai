"""One-shot startup status line: what LLM/embedder/mode is actually active."""

from __future__ import annotations

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def format_preflight(
    *,
    llm_name: Optional[str],
    llm_model: Optional[str],
    embedder_name: str,
    mode: str,
) -> str:
    llm = f"{llm_name}({llm_model or '?'})" if llm_name else "none -> extractive fallback"
    return f"KB.AI ready · LLM={llm} · Embedder={embedder_name} · Mode={mode}"


def log_preflight(
    *,
    llm_name: Optional[str],
    llm_model: Optional[str],
    embedder_name: str,
    mode: str,
) -> None:
    LOGGER.info(
        format_preflight(
            llm_name=llm_name,
            llm_model=llm_model,
            embedder_name=embedder_name,
            mode=mode,
        )
    )
