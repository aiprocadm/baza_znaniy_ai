"""Common protocol describing embedding model behaviour."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface required by vector stores to embed text batches."""

    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimensionality of generated sentence embeddings."""

    def encode(
        self,
        texts: Sequence[str] | Iterable[str],
        *,
        convert_to_numpy: bool = True,
    ) -> NDArray[np.floating[np.float32]] | Sequence[Sequence[float]]:
        """Encode ``texts`` into dense vectors.

        Implementations should return numpy arrays when ``convert_to_numpy`` is ``True``.
        When ``False`` a nested sequence of floats is acceptable.  The caller is
        responsible for normalising or converting the result to ``numpy`` arrays.
        """

    # ``__call__`` is intentionally omitted; factory functions may return any
    # callable adhering to this protocol.
