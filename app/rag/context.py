"""Utilities for preparing retrieval context and citations."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .tokenizer import detokenize, tokenize


def build_context(hits: Iterable[Dict], token_limit: int = 3000) -> str:
    """Build a context string from *hits* while respecting *token_limit*.

    The function reuses the same tokeniser that powers the chunking pipeline, so
    the limit matches the ingestion behaviour.
    """

    remaining = token_limit
    context_tokens: List[str] = []
    first = True

    for hit in hits:
        text = hit.get("text", "") or ""
        if not text:
            continue

        text_tokens = tokenize(text)
        if not text_tokens:
            continue

        if not first:
            separator_tokens = tokenize("\n\n")
            to_take = min(len(separator_tokens), remaining)
            context_tokens.extend(separator_tokens[:to_take])
            remaining -= to_take
            if remaining <= 0:
                break
        first = False

        to_take = min(len(text_tokens), remaining)
        context_tokens.extend(text_tokens[:to_take])
        remaining -= to_take
        if remaining <= 0:
            break

    return detokenize(context_tokens)


def select_citations(hits: Sequence[Dict], minimum: int = 3, maximum: int = 5) -> List[Dict]:
    """Return a slice of *hits* for citation output.

    The function guarantees at least *minimum* items by duplicating the last
    available hit when the search does not return enough distinct results.  This
    fulfils the API contract that the client can always expect a fixed number of
    citations.
    """

    if maximum < minimum:
        raise ValueError("maximum cannot be smaller than minimum")

    if not hits:
        return []

    count = min(len(hits), maximum)
    count = max(count, minimum)

    selected = list(hits[: min(len(hits), count)])
    while len(selected) < count:
        selected.append(hits[-1])

    return selected
