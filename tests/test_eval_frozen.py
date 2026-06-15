"""Offline frozen-embeddings retrieval gate.

Three layers:

* ``test_frozen_retriever_ranks_by_cosine`` — pure unit test on a tiny
  hand-built ``.npz`` + sidecar (no committed data, no model): proves the
  retriever ranks by cosine and resolves query text → top-k passage keys.
* ``test_public_corpus_meets_committed_floors`` — THE GATE. Loads the committed
  frozen bge-m3 vectors and ranks the answerable ``golden_public`` items with
  pure numpy, then asserts every aggregate metric stays at or above its floor in
  ``data/eval/ci_thresholds.json``. Deterministic, downloads no model — this is
  what runs in the ``eval-gate`` CI job.
* ``test_frozen_query_vectors_match_live_bge_m3`` — ``@pytest.mark.integration``
  staleness check: re-embeds a frozen query with the real bge-m3 and asserts it
  still matches the committed vector. Skips loudly when ST/bge-m3 is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.eval.dataset import load_golden
from app.eval.frozen import make_frozen_retriever
from app.eval.retrieval_eval import evaluate

CORPUS_DIR = Path("data/eval/corpus_public")
NPZ = CORPUS_DIR / "frozen_bge-m3.npz"
KEYS = CORPUS_DIR / "frozen_bge-m3.keys.json"
GOLDEN = Path("data/eval/golden_public.jsonl")
THRESHOLDS = Path("data/eval/ci_thresholds.json")


def _write_tiny_frozen(tmp_path: Path) -> tuple[Path, Path]:
    """A 3-passage / 2-query frozen set with known cosine ordering."""
    passage_vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.70710677, 0.70710677]], dtype=np.float32)
    query_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    npz = tmp_path / "frozen_tiny.npz"
    keys = tmp_path / "frozen_tiny.keys.json"
    np.savez_compressed(npz, passage_vecs=passage_vecs, query_vecs=query_vecs)
    keys.write_text(
        json.dumps(
            {
                "passage_keys": ["a.md:0", "b.md:0", "c.md:0"],
                "query_texts": ["про a", "про b"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return npz, keys


def test_frozen_retriever_ranks_by_cosine(tmp_path) -> None:
    npz, keys = _write_tiny_frozen(tmp_path)
    retrieve = make_frozen_retriever(npz, keys)

    # Query "про a" aligns with passage a; c (45°) outranks the orthogonal b.
    top = retrieve("про a", 3)
    assert [h.chunk_key for h in top] == ["a.md:0", "c.md:0", "b.md:0"]
    # top_k truncates.
    assert [h.chunk_key for h in retrieve("про b", 1)] == ["b.md:0"]
    # A query absent from the frozen set is a loud failure, not a silent miss.
    with pytest.raises(KeyError):
        retrieve("неизвестный вопрос", 5)


def test_public_corpus_meets_committed_floors() -> None:
    """The gate: committed frozen vectors must still clear every metric floor."""
    floors: dict[str, float] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["floors"]
    retrieve = make_frozen_retriever(NPZ, KEYS)
    answerable = [it for it in load_golden(GOLDEN) if it.relevant_chunks]
    assert answerable, "golden_public has no answerable items — corpus drift?"

    aggregate = evaluate(answerable, retrieve)["aggregate"]

    failures = {
        metric: (aggregate[metric], floor)
        for metric, floor in floors.items()
        if aggregate.get(metric, 0.0) + 1e-9 < floor
    }
    assert not failures, f"retrieval regression below committed floors: {failures}"


@pytest.mark.integration
def test_frozen_query_vectors_match_live_bge_m3() -> None:
    """Re-embedding a frozen query with the live model must match the commit.

    Guards against silent embedder drift (a model/version change that would make
    the committed vectors — and therefore the gate — meaningless). Needs the real
    bge-m3; skips loudly when it cannot be loaded.
    """
    import os

    if os.environ.get("KB_EMBEDDINGS_BACKEND", "st") != "st":
        pytest.skip("KB_EMBEDDINGS_BACKEND is not 'st' — cannot validate bge-m3 freeze")
    try:
        from app.services.kb_embeddings import get_embedder

        embedder = get_embedder()
    except Exception as exc:  # noqa: BLE001 — any load failure is a loud skip
        pytest.skip(f"bge-m3 embedder unavailable: {exc}")

    if getattr(embedder, "dimension", 0) != 1024:
        pytest.skip(f"active embedder is not bge-m3 (dim={getattr(embedder, 'dimension', 0)})")

    meta = json.loads(KEYS.read_text(encoding="utf-8"))
    committed = np.asarray(np.load(NPZ)["query_vecs"], dtype=np.float32)
    query = meta["query_texts"][0]

    vec = np.asarray(embedder.embed_query(query), dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) or 1.0)
    cosine = float(vec @ committed[0])
    assert cosine > 0.999, f"live bge-m3 drifted from the frozen vector (cos={cosine:.5f})"
