import os
from pypdf import PdfReader
from docx import Document
import io, re
from typing import List, Dict, Optional, Protocol

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency is provided via requirements
    tiktoken = None  # type: ignore


class _Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: List[int]) -> str:
        ...


class _CharTokenizer:
    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)


_TOKENIZER: Optional[_Tokenizer] = None


def _get_tokenizer() -> _Tokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        if tiktoken is not None:
            try:
                _TOKENIZER = tiktoken.get_encoding("cl100k_base")
            except Exception:
                try:
                    _TOKENIZER = tiktoken.encoding_for_model("text-embedding-3-small")
                except Exception:
                    _TOKENIZER = _CharTokenizer()
        else:
            _TOKENIZER = _CharTokenizer()
    return _TOKENIZER

def _clean(t: str) -> str:
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _chunk(text: str, chunk=900, overlap=140, encoder: Optional[_Tokenizer] = None):
    if not text:
        return []

    encoder = encoder or _get_tokenizer()
    token_ids = encoder.encode(text)
    if not token_ids:
        return []

    chunk = max(int(chunk), 1)
    overlap = max(int(overlap), 0)
    if overlap >= chunk:
        overlap = chunk - 1 if chunk > 1 else 0

    out = []
    start = 0
    total = len(token_ids)

    while start < total:
        end = min(start + chunk, total)
        window_tokens = token_ids[start:end]
        out.append(encoder.decode(window_tokens))
        if end >= total:
            break
        start = max(end - overlap, 0)

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
    encoder = _get_tokenizer()

    for page, txt in text_pages:
        if not txt: continue
        for ch in _chunk(txt, chunk=csize, overlap=cover, encoder=encoder):
            chunks.append({"file": filename, "page": page, "text": ch})
    return chunks
