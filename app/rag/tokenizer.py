"""Utility functions for tokenising and detokenising text chunks.

The chunking pipeline in :mod:`app.rag.ingest` operates on the token
representation defined here.  By keeping the tokenizer logic in a single
module we can ensure that other components (for example, prompt context
construction) use the exact same notion of a "token" and therefore obey the
same limits.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def tokenize(text: str) -> List[str]:
    """Tokenise *text* into a list of tokens.

    The current chunking strategy slices documents by character count.  To keep
    compatibility we consider each character to be a token.  A dedicated
    function is still useful because it gives us a single place to update the
    tokenisation strategy in the future without touching the rest of the code.
    """

    return list(text)


def detokenize(tokens: Iterable[str]) -> str:
    """Turn an iterable of *tokens* back into a string."""

    return "".join(tokens)


def truncate_tokens(tokens: Sequence[str], limit: int) -> List[str]:
    """Return at most *limit* tokens from *tokens*.

    The helper ensures that negative limits never raise by returning an empty
    list.
    """

    if limit <= 0:
        return []
    if len(tokens) <= limit:
        return list(tokens)
    return list(tokens[:limit])


def count_tokens(text: str) -> int:
    """Count how many tokens are present in *text*."""

    return len(text)
