"""Test helpers providing lightweight stubs for optional dependencies."""

from __future__ import annotations

import sys
import types
from typing import Any


def install_service_stubs() -> None:
    """Ensure external dependencies are replaced with lightweight stubs."""

    if "qdrant_client" not in sys.modules:
        qdrant_module = types.ModuleType("qdrant_client")

        class DummyQdrantClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._args = args
                self._kwargs = kwargs

            def get_collection(self, *_: Any, **__: Any) -> None:
                raise UnexpectedResponse()

            def recreate_collection(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
                return None

            def create_payload_index(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
                return None

            def upsert(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
                return None

            def search(self, *args: Any, **kwargs: Any) -> list[Any]:  # pragma: no cover
                return []

        qdrant_module.QdrantClient = DummyQdrantClient

        exceptions_module = types.ModuleType("qdrant_client.http.exceptions")

        class UnexpectedResponse(Exception):
            pass

        exceptions_module.UnexpectedResponse = UnexpectedResponse

        models_module = types.ModuleType("qdrant_client.http.models")

        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs
                self.size = kwargs.get("size", 0)

        class HnswConfigDiff:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class PointStruct:
            def __init__(self, **kwargs: Any) -> None:
                self.id = kwargs.get("id")
                self.vector = kwargs.get("vector")
                self.payload = kwargs.get("payload")
                self.score = kwargs.get("score", 0.0)

        class SearchParams:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        models_module.Distance = Distance
        models_module.VectorParams = VectorParams
        models_module.HnswConfigDiff = HnswConfigDiff
        models_module.PointStruct = PointStruct
        models_module.SearchParams = SearchParams

        http_module = types.ModuleType("qdrant_client.http")
        http_module.models = models_module
        http_module.exceptions = exceptions_module

        qdrant_module.http = http_module

        sys.modules["qdrant_client"] = qdrant_module
        sys.modules["qdrant_client.http"] = http_module
        sys.modules["qdrant_client.http.models"] = models_module
        sys.modules["qdrant_client.http.exceptions"] = exceptions_module

    if "sentence_transformers" not in sys.modules:
        st_module = types.ModuleType("sentence_transformers")

        class DummySentenceTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

            def encode(self, texts: Any, convert_to_numpy: bool = False):
                import numpy as np

                length = len(texts) if texts else 0
                vectors = np.zeros((length, 1), dtype=np.float32)
                if convert_to_numpy:
                    return vectors
                return vectors.tolist()

            def get_sentence_embedding_dimension(self) -> int:
                return 1

        st_module.SentenceTransformer = DummySentenceTransformer
        
        class DummyCrossEncoder:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

            def predict(self, pairs: Any):
                return [0.0 for _ in pairs]

        st_module.CrossEncoder = DummyCrossEncoder
        sys.modules["sentence_transformers"] = st_module
