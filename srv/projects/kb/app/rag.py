"""Utilities for preparing retrieval context and citations."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .ingest import _get_tokenizer


def _tokenize(text: str) -> List[str]:
    tokenizer = _get_tokenizer()
    return tokenizer.encode(text)


def _detokenize(tokens: Iterable[int]) -> str:
    tokenizer = _get_tokenizer()
    return tokenizer.decode(list(tokens))


def build_context(hits: Iterable[Dict], token_limit: int = 3000) -> str:
    """Build a context string from *hits* while respecting *token_limit*."""

    remaining = token_limit
    context_tokens: List[int] = []
    first = True

    for hit in hits:
        text = hit.get("text", "") or ""
        if not text:
            continue

        text_tokens = _tokenize(text)
        if not text_tokens:
            continue

        if not first:
            separator = _tokenize("\n\n")
            take = min(len(separator), remaining)
            context_tokens.extend(separator[:take])
            remaining -= take
            if remaining <= 0:
                break
        first = False

        take = min(len(text_tokens), remaining)
        context_tokens.extend(text_tokens[:take])
        remaining -= take
        if remaining <= 0:
            break

    return _detokenize(context_tokens)


def _citation_key(hit: Dict) -> Tuple:
    file_id = hit.get("file")
    page = hit.get("page")
    if file_id is None and page is None:
        return (
            hit.get("sha256"),
            hit.get("id"),
            hit.get("text"),
        )
    return (file_id, page)


def select_citations(
    hits: Sequence[Dict], minimum: int = 3, maximum: int = 5
) -> Tuple[List[Dict], bool]:
    if maximum < minimum:
        raise ValueError("maximum cannot be smaller than minimum")

    unique: List[Dict] = []
    seen = set()
    for hit in hits:
        key = _citation_key(hit)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
        if len(unique) >= maximum:
            break

    has_minimum = len(unique) >= minimum
    return unique, has_minimum


__all__ = ["build_context", "select_citations"]
