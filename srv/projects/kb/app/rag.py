"""Retrieval-Augmented Generation primitives."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Tuple

from .models import Document

_WORD_RE = re.compile(r"[\w']+")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _tf(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def _cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    common = set(vec_a.keys()) & set(vec_b.keys())
    numerator = sum(vec_a[token] * vec_b[token] for token in common)
    if numerator == 0:
        return 0.0
    sum1 = sum(value ** 2 for value in vec_a.values())
    sum2 = sum(value ** 2 for value in vec_b.values())
    return numerator / math.sqrt(sum1 * sum2)


def retrieve(query: str, documents: Iterable[Document], limit: int = 3) -> List[Tuple[Document, float]]:
    """Return the top matching documents for the query."""

    query_tokens = _tokenize(query)
    query_vec = _tf(query_tokens)
    scored: List[Tuple[Document, float]] = []
    for document in documents:
        doc_vec = _tf(_tokenize(document.content))
        score = _cosine_similarity(query_vec, doc_vec)
        if score > 0:
            scored.append((document, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]
