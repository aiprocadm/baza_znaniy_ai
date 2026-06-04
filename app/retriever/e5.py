"""e5-family query/passage prefixing (https://huggingface.co/intfloat/multilingual-e5-small)."""

from __future__ import annotations


def _is_e5(model: str) -> bool:
    return "e5" in (model or "").lower()


def e5_prefix(text: str, *, role: str, model: str, enabled: bool) -> str:
    """Prepend 'query: ' / 'passage: ' for e5 models when enabled; else return text unchanged."""
    if not enabled or not _is_e5(model):
        return text
    if role not in ("query", "passage"):
        raise ValueError(f"role must be 'query' or 'passage', got {role!r}")
    return f"{role}: {text}"
