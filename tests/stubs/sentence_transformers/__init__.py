"""Hermetic ``sentence_transformers`` stub for the test suite.

Why this exists
---------------
``app.retriever.rerank`` binds the cross-encoder at *import time*::

    from sentence_transformers import CrossEncoder

That static binding means whichever test imports ``app.retriever.rerank``
first decides — for the whole pytest session — whether the name resolves to
the real, weight-loading ``CrossEncoder`` (≈80 MB download + real inference)
or to a lightweight stub. When the real ``sentence-transformers`` wheel is
installed, a test that imports the module before any stub is registered (e.g.
``tests/test_reranking.py``) used to leak the real class into every later
test, making reranker behaviour order-dependent. The visible symptom was
``tests/test_service_api.py::test_chat_truncates_hits_and_formats_response``
passing in isolation but failing when paired with ``test_reranking.py`` —
real reranking reordered the chat hits.

Unlike ``llama_cpp``/``qdrant_client`` etc., ``sentence_transformers`` had no
entry here and was stubbed only by the *conditional* guard in
``tests/service_stubs.py`` (``if "sentence_transformers" not in sys.modules``),
which a pre-imported real package silently defeats.

``tests/conftest.py`` prepends ``tests/stubs`` to ``sys.path`` before any test
module is collected, so placing the stub here guarantees the import resolves
to these deterministic dummies *first*, independent of test order — exactly
how ``faiss``/``qdrant_client``/``trl`` are handled. The behaviour mirrors the
inline dummies in ``tests/service_stubs.py`` (zero embeddings, zero rerank
scores) so the two stub entry points stay equivalent.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SentenceTransformer", "CrossEncoder"]


class SentenceTransformer:
    """Deterministic bi-encoder stub returning zero vectors (dimension 384)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def encode(self, texts: Any, *, convert_to_numpy: bool = False, **_: Any):
        import numpy as np

        length = len(texts) if texts else 0
        vectors = np.zeros((length, 384), dtype=np.float32)
        if convert_to_numpy:
            return vectors
        return vectors.tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return 384


class CrossEncoder:
    """Deterministic cross-encoder stub scoring every pair as ``0.0``.

    Equal scores make :class:`app.retriever.rerank.CrossEncoderReranker`
    a stable, order-preserving truncator, which is what the unit tests in
    ``tests/test_service_api.py`` and ``tests/test_reranking.py`` assert.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def predict(self, pairs: Any):
        return [0.0 for _ in pairs]
