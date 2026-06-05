"""Unit tests for the in-process sentence-transformers embedder backend."""

from __future__ import annotations

from app.services.kb_embeddings import SentenceTransformerEmbedder, _build_from_env


class _FakeST:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.last: str | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, text, **kwargs):
        import numpy as np

        self.last = text
        # Deterministic, text-dependent, fixed-length vector.
        seed = float(len(text) % 97 + 1)
        return np.full((self._dim,), seed, dtype=np.float32)


def test_st_embedder_name_dim_and_embed() -> None:
    emb = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", model=_FakeST(8))
    assert emb.name == "st"
    assert emb.model == "BAAI/bge-m3"
    assert emb.dimension == 8
    vec = emb.embed("привет мир")
    assert isinstance(vec, list) and len(vec) == 8
    assert all(isinstance(v, float) for v in vec)


def test_st_backend_is_selected_by_env_without_loading() -> None:
    # Building from env must NOT load a real model (no `model=` injected).
    emb = _build_from_env({"KB_EMBEDDINGS_BACKEND": "st", "ST_EMBED_MODEL": "BAAI/bge-m3"})
    assert emb.name == "st"
    assert getattr(emb, "model", None) == "BAAI/bge-m3"
