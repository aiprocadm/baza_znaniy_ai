import os, sqlite3
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np, faiss

BASE = "/srv/projects/kb/data/storage/qdrant"
IDX_PATH = os.path.join(BASE,"kb.index")
DB_PATH  = os.path.join(BASE,"meta.sqlite")
DIM = 384

_model = None
_index = None

def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("EMBED_MODEL","intfloat/multilingual-e5-small"))
    return _model

def _norm(v):
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def ensure_collection():
    os.makedirs(BASE, exist_ok=True)
    _init_db()
    _load_index()

def _init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            file TEXT, page INTEGER, text TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_file ON chunks(file)")
        c.commit()

def _load_index():
    global _index
    if _index is not None: return
    if os.path.exists(IDX_PATH):
        _index = faiss.read_index(IDX_PATH)
        return
    base = faiss.IndexFlatIP(DIM)
    _index = faiss.IndexIDMap2(base)
    faiss.write_index(_index, IDX_PATH)

def upsert_chunks(chunks: List[Dict]):
    ensure_collection()
    texts = [c["text"] for c in chunks]
    embs = _norm(_embedder().encode(texts, convert_to_numpy=True))
    ids = []
    with sqlite3.connect(DB_PATH) as c:
        for ch in chunks:
            cur = c.execute("INSERT INTO chunks(file,page,text) VALUES(?,?,?)",
                            (ch.get("file"), int(ch.get("page") or 0), ch["text"]))
            ids.append(cur.lastrowid)
        c.commit()
    import numpy as np
    ids_np = np.array(ids, dtype=np.int64)
    _index.add_with_ids(embs, ids_np)
    faiss.write_index(_index, IDX_PATH)

def search_chunks(query: str, top_k: int = 10) -> List[Dict]:
    ensure_collection()
    q = _norm(_embedder().encode([query], convert_to_numpy=True))
    D, I = _index.search(q, top_k)
    res = []
    with sqlite3.connect(DB_PATH) as c:
        for idx, score in zip(I[0], D[0]):
            if int(idx) < 0: continue
            row = c.execute("SELECT file,page,text FROM chunks WHERE id=?", (int(idx),)).fetchone()
            if row:
                res.append({"file":row[0], "page":row[1], "text":row[2], "score":float(score)})
    return res
