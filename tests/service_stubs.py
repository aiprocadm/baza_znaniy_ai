"""Test helpers providing lightweight stubs for optional dependencies."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


def install_service_stubs() -> None:
    """Ensure external dependencies are replaced with lightweight stubs."""

    if "llama_cpp" not in sys.modules:
        llama_module = types.ModuleType("llama_cpp")

        class DummyLlama:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs
                self.adapters: dict[str, dict[str, Any]] = {}
                self.active_adapter: str | None = None

            def load_adapter(self, path: str, adapter_name: str | None = None, scale: float | None = None) -> None:
                name = adapter_name or Path(path).stem or "adapter"
                self.adapters[name] = {"path": path, "scale": scale}
                self.active_adapter = name

            def set_adapter(self, adapter_name: str) -> None:
                if adapter_name not in self.adapters:
                    raise ValueError("Adapter not loaded")
                self.active_adapter = adapter_name

            def unload_adapter(self, adapter_name: str | None = None) -> None:
                target = adapter_name or self.active_adapter
                if target is None:
                    return
                self.adapters.pop(target, None)
                if self.active_adapter == target:
                    self.active_adapter = None

        llama_module.Llama = DummyLlama
        sys.modules["llama_cpp"] = llama_module
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

        class PayloadSchemaType:
            KEYWORD = "keyword"
            INTEGER = "integer"

        models_module.Distance = Distance
        models_module.VectorParams = VectorParams
        models_module.HnswConfigDiff = HnswConfigDiff
        models_module.PointStruct = PointStruct
        models_module.SearchParams = SearchParams
        models_module.PayloadSchemaType = PayloadSchemaType

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
                vectors = np.zeros((length, 384), dtype=np.float32)
                if convert_to_numpy:
                    return vectors
                return vectors.tolist()

            def get_sentence_embedding_dimension(self) -> int:
                return 384

        st_module.SentenceTransformer = DummySentenceTransformer
        
        class DummyCrossEncoder:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

            def predict(self, pairs: Any):
                return [0.0 for _ in pairs]

        st_module.CrossEncoder = DummyCrossEncoder
        sys.modules["sentence_transformers"] = st_module

    if "apscheduler" not in sys.modules:
        aps_module = types.ModuleType("apscheduler")
        jobstores_pkg = types.ModuleType("apscheduler.jobstores")
        base_module = types.ModuleType("apscheduler.jobstores.base")
        schedulers_pkg = types.ModuleType("apscheduler.schedulers")
        asyncio_module = types.ModuleType("apscheduler.schedulers.asyncio")
        triggers_pkg = types.ModuleType("apscheduler.triggers")
        cron_module = types.ModuleType("apscheduler.triggers.cron")
        interval_module = types.ModuleType("apscheduler.triggers.interval")

        class JobLookupError(Exception):
            """Placeholder job lookup error for scheduler stubs."""

        class AsyncIOScheduler:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs
                self.jobs: list[tuple[Any, dict[str, Any]]] = []

            def add_job(self, func: Any, trigger: Any = None, **kwargs: Any) -> None:
                self.jobs.append((func, kwargs))

            def start(self) -> None:  # pragma: no cover - noop
                return None

            def shutdown(self, wait: bool = True) -> None:  # pragma: no cover - noop
                return None

        class CronTrigger:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class IntervalTrigger:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        base_module.JobLookupError = JobLookupError
        jobstores_pkg.base = base_module
        aps_module.jobstores = jobstores_pkg
        asyncio_module.AsyncIOScheduler = AsyncIOScheduler
        schedulers_pkg.asyncio = asyncio_module
        aps_module.schedulers = schedulers_pkg
        cron_module.CronTrigger = CronTrigger
        interval_module.IntervalTrigger = IntervalTrigger
        triggers_pkg.cron = cron_module
        triggers_pkg.interval = interval_module
        aps_module.triggers = triggers_pkg

        sys.modules["apscheduler"] = aps_module
        sys.modules["apscheduler.jobstores"] = jobstores_pkg
        sys.modules["apscheduler.jobstores.base"] = base_module
        sys.modules["apscheduler.schedulers"] = schedulers_pkg
        sys.modules["apscheduler.schedulers.asyncio"] = asyncio_module
        sys.modules["apscheduler.triggers"] = triggers_pkg
        sys.modules["apscheduler.triggers.cron"] = cron_module
        sys.modules["apscheduler.triggers.interval"] = interval_module
