"""Reusable eval guards (loud, never silent)."""

from __future__ import annotations

from app.eval.dataset import CorpusSignature


def ensure_real_embedder(sig: CorpusSignature, *, allow_hashing: bool) -> None:
    if sig.embedder_name == "hash" and not allow_hashing:
        raise SystemExit(
            "Refusing to produce a baseline on the hashing embedder (near-random "
            "results). Configure KB_EMBEDDINGS_BACKEND=ollama|api (+ model/base), or "
            "pass --allow-hashing for a throwaway smoke run."
        )
