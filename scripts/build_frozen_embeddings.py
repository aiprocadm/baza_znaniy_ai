"""Freeze public-corpus embeddings for the no-model CI gate (PR3).

Embeds every public-corpus passage (``embed`` → passage role) and every public
golden question (``embed_query`` → query role) with the active embedder, then
writes TWO files — a numeric ``.npz`` plus a JSON string sidecar. The split
keeps the numpy loader on its safe default (no object arrays): strings inside
the archive would force the unsafe loader flag, an arbitrary-code-execution
risk on a committed fixture (spec 2026-06-06 §8).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.eval.adapter import _build_key_map
from app.eval.dataset import load_golden

LOGGER = logging.getLogger(__name__)

OUT_DIR = Path("data/eval/corpus_public")
GOLDEN = Path("data/eval/golden_public.jsonl")


@dataclass(frozen=True)
class FrozenSet:
    passage_keys: tuple[str, ...]
    passage_vecs: "np.ndarray"  # (N, d) float32, L2-normalized
    query_texts: list[str]
    query_vecs: "np.ndarray"  # (M, d) float32, L2-normalized


def _l2(rows: list[list[float]]) -> "np.ndarray":
    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def build_frozen(store, embedder, golden_path: Path) -> FrozenSet:
    key_map = _build_key_map(store)
    with store._connect() as conn:  # noqa: SLF001 — same access the adapter uses
        rows = conn.execute(
            "SELECT document_id, chunk_index, text FROM kb_chunks "
            "ORDER BY document_id, chunk_index"
        ).fetchall()
    keys: list[str] = []
    passage_rows: list[list[float]] = []
    for doc_id, idx, text in rows:
        key = key_map.get((int(doc_id), int(idx)))
        if key is None:
            continue
        keys.append(key)
        passage_rows.append(embedder.embed(text))
        if len(keys) % 50 == 0:
            LOGGER.info("embedded %d passages", len(keys))

    questions = [it.question for it in load_golden(golden_path)]
    query_rows = [embedder.embed_query(q) for q in questions]
    return FrozenSet(tuple(keys), _l2(passage_rows), questions, _l2(query_rows))


def write_frozen(frozen: FrozenSet, out_dir: Path, embedder_tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"frozen_{embedder_tag}.npz"
    keys = out_dir / f"frozen_{embedder_tag}.keys.json"
    np.savez_compressed(npz, passage_vecs=frozen.passage_vecs, query_vecs=frozen.query_vecs)
    keys.write_text(
        json.dumps(
            {"passage_keys": list(frozen.passage_keys), "query_texts": frozen.query_texts},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return npz, keys


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="bge-m3", help="embedder tag in the output filenames")
    args = parser.parse_args(argv)

    from app.services.kb_embeddings import get_embedder
    from app.services.kb_store import get_store

    frozen = build_frozen(get_store(), get_embedder(), Path(args.golden))
    npz, keys = write_frozen(frozen, Path(args.out_dir), args.tag)
    print(
        f"OK: {len(frozen.passage_keys)} passages, "
        f"{len(frozen.query_texts)} queries -> {npz}, {keys}"
    )


if __name__ == "__main__":
    main()
