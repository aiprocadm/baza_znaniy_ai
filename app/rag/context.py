"""Utilities for preparing retrieval context and citations."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

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


def _citation_key(hit: Dict) -> Tuple:
    """Build a key for identifying unique citation hits."""

    file_id = hit.get("file")
    page = hit.get("page")
    if file_id is None and page is None:
        # Fall back to stable identifiers when available to avoid treating
        # different documents as the same citation.
        return (
            hit.get("chunk_id"),
            hit.get("id"),
            hit.get("text"),
        )
    return (file_id, page)


def select_citations(
    hits: Sequence[Dict], minimum: int = 3, maximum: int = 5
) -> Tuple[List[Dict], bool]:
    """Return distinct citation hits and whether the minimum was satisfied.

    The function filters duplicate citations based on their document and page,
    ensures that no more than *maximum* items are returned, and reports whether
    at least *minimum* unique citations were available.
    """

    if maximum < minimum:
        raise ValueError("maximum cannot be smaller than minimum")

    unique: List[Dict] = []
    seen = set()
    for hit in hits:
        key = _citation_key(hit)
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(hit)
        # Hoisted to one local so isinstance can narrow it; don't fold back into
        # a double ``enriched.get("meta")`` — mypy can't narrow across two calls.
        raw_meta = enriched.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        enriched.setdefault("article", meta.get("article"))
        enriched.setdefault("clause", meta.get("clause"))
        enriched.setdefault("revision", meta.get("revision"))
        enriched.setdefault(
            "revision_date", meta.get("effective_date") or meta.get("adoption_date")
        )
        unique.append(enriched)
        if len(unique) >= maximum:
            break

    has_minimum = len(unique) >= minimum
    return unique, has_minimum
