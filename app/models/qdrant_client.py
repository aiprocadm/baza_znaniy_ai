import os, sqlite3
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np, faiss

BASE = "/srv/projects/kb/data/storage/qdrant"
IDX_PATH = os.path.join(BASE,"kb.index")
DB_PATH  = os.path.join(BASE,"meta.sqlite")

_model = None
_index = None
_expected_dim = None

def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("EMBED_MODEL","intfloat/multilingual-e5-small"))
    return _model

def _embedding_dim():
    global _expected_dim
    if _expected_dim is None:
        model = _embedder()
        if hasattr(model, "get_sentence_embedding_dimension"):
            _expected_dim = int(model.get_sentence_embedding_dimension())
        else:
            sample = model.encode([""], convert_to_numpy=True)
            _expected_dim = int(sample.shape[1])
    return _expected_dim

def _norm(v):
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def ensure_collection():
    os.makedirs(BASE, exist_ok=True)
    dim = _embedding_dim()
    _load_index(dim)
    _init_db()

def _init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            file TEXT, page INTEGER, text TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_file ON chunks(file)")
        c.commit()

def _create_index(dim: int):
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)

def _handle_dim_mismatch(saved_dim: int, expected_dim: int):
    global _index, _expected_dim
    if os.path.exists(IDX_PATH):
        os.remove(IDX_PATH)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    _index = _create_index(expected_dim)
    faiss.write_index(_index, IDX_PATH)
    _init_db()
    _expected_dim = expected_dim

def _ensure_index_dim(expected_dim: int):
    global _index
    if _index is not None and getattr(_index, "d", None) != expected_dim:
        _handle_dim_mismatch(getattr(_index, "d", None), expected_dim)

def _load_index(expected_dim: int):
    global _index
    if _index is not None:
        _ensure_index_dim(expected_dim)
        return
    if os.path.exists(IDX_PATH):
        idx = faiss.read_index(IDX_PATH)
        if getattr(idx, "d", None) != expected_dim:
            _handle_dim_mismatch(getattr(idx, "d", None), expected_dim)
            return
        _index = idx
        return
    _index = _create_index(expected_dim)
    faiss.write_index(_index, IDX_PATH)

def upsert_chunks(chunks: List[Dict]):
    ensure_collection()
    dim = _embedding_dim()
    _ensure_index_dim(dim)
    texts = [c["text"] for c in chunks]
    embs = _norm(_embedder().encode(texts, convert_to_numpy=True))
    if embs.shape[1] != dim:
        raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {embs.shape[1]}")
    ids = []
    with sqlite3.connect(DB_PATH) as c:
        for ch in chunks:
            cur = c.execute("INSERT INTO chunks(file,page,text) VALUES(?,?,?)",
                            (ch.get("file"), int(ch.get("page") or 0), ch["text"]))
            ids.append(cur.lastrowid)
        c.commit()
    ids_np = np.array(ids, dtype=np.int64)
    _index.add_with_ids(embs, ids_np)
    faiss.write_index(_index, IDX_PATH)

def search_chunks(query: str, top_k: int = 10) -> List[Dict]:
    ensure_collection()
    dim = _embedding_dim()
    _ensure_index_dim(dim)
    q = _norm(_embedder().encode([query], convert_to_numpy=True))
    if q.shape[1] != dim:
        raise ValueError(f"Query embedding dimension mismatch: expected {dim}, got {q.shape[1]}")
    D, I = _index.search(q, top_k)
    res = []
    with sqlite3.connect(DB_PATH) as c:
        for idx, score in zip(I[0], D[0]):
            if int(idx) < 0: continue
            row = c.execute("SELECT file,page,text FROM chunks WHERE id=?", (int(idx),)).fetchone()
            if row:
                res.append({"file":row[0], "page":row[1], "text":row[2], "score":float(score)})
    return res
