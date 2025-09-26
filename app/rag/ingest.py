import os
from pypdf import PdfReader
from docx import Document
import io, re
from typing import List, Dict

from .tokenizer import detokenize, tokenize

def _clean(t: str) -> str:
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _chunk(text: str, chunk=900, overlap=140):
    chunk = max(int(chunk), 1)
    overlap = max(int(overlap), 0)
    if overlap >= chunk:
        overlap = chunk - 1
    tokens = tokenize(text)
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + chunk, n)
        out.append(detokenize(tokens[i:j]))
        i = j - overlap if j < n else j
        if i < 0: i = 0
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
