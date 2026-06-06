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


class _RecordingST:
    def __init__(self) -> None:
        self.last: str | None = None

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, text, **kwargs):
        import numpy as np

        self.last = text
        return np.zeros((4,), dtype=np.float32)


def test_e5_passage_and_query_prefixes_when_enabled() -> None:
    rec = _RecordingST()
    emb = SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-base", e5_prefix_enabled=True, model=rec
    )
    emb.embed("текст документа")
    assert rec.last == "passage: текст документа"
    emb.embed_query("мой вопрос")
    assert rec.last == "query: мой вопрос"


def test_no_prefix_for_bge_even_when_enabled() -> None:
    rec = _RecordingST()
    emb = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", e5_prefix_enabled=True, model=rec)
    emb.embed("текст")
    assert rec.last == "текст"


class _RecordingEmbedder:
    name = "st"
    dimension = 4

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.1, 0.2, 0.3, 0.4]


def test_store_uses_embed_query_for_search(tmp_path) -> None:
    from app.services.kb_store import KnowledgeBaseStore

    fake = _RecordingEmbedder()
    store = KnowledgeBaseStore(db_path=str(tmp_path / "kb.sqlite"), embedder=fake)
    store.add_document("doc", text="первый чанк текста. второй чанк текста.")
    store.search("поисковый запрос", top_k=3)
    assert fake.query_calls and fake.query_calls[-1] == "поисковый запрос"
