import io
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List

import tiktoken
from docx import Document
from pypdf import PdfReader
from tiktoken.core import Encoding


LOGGER = logging.getLogger(__name__)


def _clean(t: str) -> str:
    t = re.sub(r'\s+', ' ', t).strip()
    return t


@lru_cache(maxsize=1)
def _get_tokenizer() -> Encoding:
    """Return the tokenizer used to measure token lengths."""

    name = os.getenv("RAG_TOKENIZER_NAME", "cl100k_base")
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover - handled by tests via behaviour
        LOGGER.warning(
            "Falling back to byte-level tokenizer for '%s': %s",
            name,
            exc,
        )
        mergeable_ranks = {bytes([i]): i for i in range(256)}
        return Encoding(
            name="byte_fallback",
            pat_str=r"(?s:.)",
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )


def _chunk(text: str, chunk=900, overlap=140):
    chunk = max(int(chunk), 1)
    overlap = max(int(overlap), 0)
    if overlap >= chunk:
        overlap = chunk - 1

    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    if not tokens:
        return []

    out = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + chunk, n)
        token_slice = tokens[i:j]
        out.append(tokenizer.decode(token_slice))
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out


def parse_and_chunk(filename: str, data: bytes) -> List[Dict]:
    ext = filename.rsplit('.',1)[-1].lower()
    text_pages = []
    if ext == "pdf":
        r = PdfReader(io.BytesIO(data))
        for p, page in enumerate(r.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            text_pages.append((p, _clean(txt)))
    elif ext == "docx":
        d = Document(io.BytesIO(data))
        txt = _clean("\n".join(p.text for p in d.paragraphs))
        text_pages = [(1, txt)]
    elif ext == "txt":
        txt = _clean(data.decode("utf-8", errors="ignore"))
        text_pages = [(1, txt)]
    else:
        return []

    chunks = []
    csize = int(os.getenv("RAG_CHUNK","900"))
    cover = int(os.getenv("RAG_OVERLAP","140"))
    for page, txt in text_pages:
        if not txt: continue
        for ch in _chunk(txt, chunk=csize, overlap=cover):
            chunks.append({"file": filename, "page": page, "text": ch})
    return chunks
