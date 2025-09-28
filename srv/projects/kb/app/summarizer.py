"""Chat conversation summarisation utilities."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

from .store import ChatStore

__all__ = ["ConversationSummarizer"]


class ConversationSummarizer:
    """Summarise chat conversations using a language model."""

    def __init__(
        self,
        store: ChatStore,
        llm_generate: Callable[[str], str],
        max_history: int = 50,
    ) -> None:
        self.store = store
        self._llm_generate = llm_generate
        self.max_history = max_history

    def _format_history(self, history: Sequence[Tuple[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{role}: {content}" for role, content in history]
        return "\n".join(lines)

    def summarize(self, conversation_id: str) -> str | None:
        """Generate and persist a short summary for *conversation_id*."""

        history = self.store.get_recent_messages(conversation_id, limit=self.max_history)
        if not history:
            return None
        current_summary = self.store.get_summary(conversation_id) or ""

        prompt_parts = [
            "Суммаризируй приведённый ниже диалог на русском языке.",
            "Выдели ключевые решения и контекст для следующих сообщений.",
        ]
        if current_summary:
            prompt_parts.append(f"Текущее саммари: {current_summary}")
        prompt_parts.append("Диалог:")
        prompt_parts.append(self._format_history(history))
        prompt_parts.append("Новая краткая выжимка:")
        prompt = "\n\n".join(part for part in prompt_parts if part)

        try:
            summary = self._llm_generate(prompt).strip()
        except Exception:  # pragma: no cover - defensive logging
            return None

        if not summary:
            return None

        self.store.save_summary(conversation_id, summary)
        return summary
