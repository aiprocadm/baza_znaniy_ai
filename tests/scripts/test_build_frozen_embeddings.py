"""Unit tests for the frozen-embeddings builder (DI — no real model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.eval.dataset import GoldenItem, save_golden
from app.services.kb_store import KnowledgeBaseStore
from scripts.build_frozen_embeddings import build_frozen, write_frozen


class _FakeEmbedder:
    name = "fake-st"
    dimension = 4

    def embed(self, text: str) -> list[float]:
        seed = float(len(text) % 7 + 1)
        return [seed, 0.0, 0.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, float(len(text) % 5 + 1), 0.0, 0.0]


def _store_with_chunks(tmp_path) -> KnowledgeBaseStore:
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"), embedder=_FakeEmbedder())
    store.add_document("Док", text="первый чанк. " * 40, filename="doc.md")
    return store


def test_build_frozen_shapes_keys_and_normalization(tmp_path) -> None:
    store = _store_with_chunks(tmp_path)
    golden = tmp_path / "golden.jsonl"
    save_golden(golden, [GoldenItem("вопрос один?", ("doc.md:0",), "a")])

    frozen = build_frozen(store, _FakeEmbedder(), golden)

    assert frozen.passage_vecs.dtype == np.float32
    assert frozen.passage_vecs.shape[0] == len(frozen.passage_keys) >= 1
    assert all(":" in k for k in frozen.passage_keys)
    assert frozen.query_vecs.shape == (1, 4)
    assert frozen.query_texts == ["вопрос один?"]
    norms = np.linalg.norm(frozen.passage_vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # L2-normalized


def test_write_frozen_loads_without_object_arrays(tmp_path) -> None:
    store = _store_with_chunks(tmp_path)
    golden = tmp_path / "golden.jsonl"
    save_golden(golden, [GoldenItem("вопрос?", ("doc.md:0",), "a")])
    frozen = build_frozen(store, _FakeEmbedder(), golden)

    npz, keys = write_frozen(frozen, tmp_path / "out", "fake-st")

    loaded = np.load(npz)  # default loader (no object arrays) — must not raise
    assert set(loaded.files) == {"passage_vecs", "query_vecs"}
    meta = json.loads(Path(keys).read_text(encoding="utf-8"))
    assert meta["passage_keys"] == list(frozen.passage_keys)
    assert meta["query_texts"] == ["вопрос?"]
