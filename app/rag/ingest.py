import os
from pypdf import PdfReader
from docx import Document
import io, re
from typing import List, Dict

def _clean(t: str) -> str:
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _chunk(text: str, chunk=900, overlap=140):
    chunk = max(1, int(chunk))
    overlap = max(0, int(overlap))

    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk, n)
        out.append(text[i:j])

        next_i = j - overlap if j < n else j
        if next_i <= i:
            next_i = min(i + 1, n)
        i = next_i
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
    if csize <= 0:
        csize = 1
    if cover < 0:
        cover = 0
    if csize > 0:
        cover = min(cover, csize - 1)
    for page, txt in text_pages:
        if not txt: continue
        for ch in _chunk(txt, chunk=csize, overlap=cover):
            chunks.append({"file": filename, "page": page, "text": ch})
    return chunks
