"""LangChain factory helpers used by chat orchestrator.

This module keeps runtime integration minimal and dependency-light so tests can
stub `build_chat_chain` without requiring LangChain packages installed.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping


def rewrite_with_history(message: str, history: str) -> str:
    """Return history-aware query rewrite.

    If history is empty we keep the original message untouched.
    """

    history_text = history.strip()
    if not history_text:
        return message
    return f"{history_text}\n\n{message}".strip()


def build_chat_chain(settings: Any) -> Callable[..., Mapping[str, Any]]:
    """Build chain callable.

    Expected callable signature: ``chain(payload=..., context=...) -> Mapping``.
    The default implementation is a stub to keep feature-flag flow safe when
    LangChain is not wired yet.
    """

    def _chain(*, payload: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError(
            "LANGCHAIN_ENABLED=true but LangChain chain is not configured. "
            "Provide app.langchain.factory.build_chat_chain implementation."
        )

    return _chain
