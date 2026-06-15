"""A Retriever backed by precomputed (frozen) embeddings — no model load.

Loads committed numeric vectors (``.npz``) plus a JSON sidecar of string
keys/texts and ranks passages by cosine similarity. Pure numpy and fully
serialization-safe: the ``.npz`` holds ONLY float arrays (no object dtype), so
``np.load`` runs on its safe default with no ``allow_pickle`` flag — the strings
live in the JSON sidecar instead. That split keeps a committed fixture from
becoming an arbitrary-code-execution vector (spec 2026-06-06 §8).

This is the engine behind the offline CI gate (``tests/test_eval_frozen.py``):
the public-corpus passage/query vectors were frozen once by
``scripts/build_frozen_embeddings.py`` under the real bge-m3 embedder, so CI can
re-rank them with nothing but numpy and assert the retrieval metrics have not
regressed — no model download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.eval.adapter import EvalHit, Retriever


def make_frozen_retriever(npz_path: str | Path, keys_path: str | Path) -> Retriever:
    """Build a Retriever that ranks frozen passages by cosine similarity.

    The returned callable maps a golden *question* (which must be one of the
    frozen ``query_texts``) to its top-k passage keys. Vectors are assumed
    L2-normalized by the builder, so the dot product is the cosine.
    """
    vecs = np.load(Path(npz_path))  # numeric-only arrays; safe default loader
    pvecs = np.asarray(vecs["passage_vecs"], dtype=np.float32)  # (N, d) L2-normalized
    qvecs = np.asarray(vecs["query_vecs"], dtype=np.float32)  # (M, d) L2-normalized
    meta = json.loads(Path(keys_path).read_text(encoding="utf-8"))
    pkeys: list[str] = list(meta["passage_keys"])
    q_index = {text: i for i, text in enumerate(meta["query_texts"])}

    def _retrieve(query: str, top_k: int) -> list[EvalHit]:
        qi = q_index.get(query)
        if qi is None:
            raise KeyError(f"query not in frozen set: {query!r}")
        sims = pvecs @ qvecs[qi]
        order = np.argsort(-sims)[:top_k]
        return [EvalHit(chunk_key=pkeys[i], text="", title="") for i in order]

    return _retrieve
