"""Lightweight FAISS stub used in the test environment."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


class IndexFlatIP:
    """Minimal in-memory index emulating the FAISS API used by the app."""

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._vectors: List[List[float]] = []

    def add(self, vectors) -> None:  # pragma: no cover - trivial stub
        if vectors is None:
            self._vectors = []
            return
        self._vectors = [list(map(float, row)) for row in vectors]

    def reset(self) -> None:  # pragma: no cover - trivial stub
        self._vectors = []

    def search(self, query, top_k: int) -> Tuple[list[list[float]], list[list[int]]]:
        scores = [[0.0 for _ in range(top_k)]]
        indices = [[-1 for _ in range(top_k)]]
        if not self._vectors:
            return scores, indices
        # Return a trivial match for deterministic behaviour in tests.
        scores[0][0] = 1.0
        indices[0][0] = 0
        return scores, indices


def write_index(index: IndexFlatIP, path: str) -> None:  # pragma: no cover - noop
    Path(path).touch()


def read_index(path: str) -> IndexFlatIP:  # pragma: no cover - minimal loader
    _ = Path(path)  # Ensure path exists
    return IndexFlatIP(1)


__all__ = ["IndexFlatIP", "read_index", "write_index"]
