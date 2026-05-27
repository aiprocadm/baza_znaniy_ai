import pytest

import app.services.vectorstore as vectorstore_module
from app.services.vectorstore import clear_fallback, index_chunks, search

_TENANT = "test-tenant"


@pytest.fixture(autouse=True)
def _isolated_vector_store() -> None:
    """Wipe both the fallback list and the (stubbed) Qdrant collection."""

    clear_fallback()
    vectorstore_module._VECTOR_STORE = None
    yield
    clear_fallback()
    vectorstore_module._VECTOR_STORE = None


def test_golden_npa_retrieval_current_revision_priority() -> None:
    index_chunks(
        [
            {
                "sha256": "fzl-current",
                "tenant_id": _TENANT,
                "file": "fzl.txt",
                "page": 1,
                "text": "Федеральный закон 123 о данных",
                "meta": {"reg_number": "123", "is_active": True, "revision": "2025"},
            },
            {
                "sha256": "fzl-old",
                "tenant_id": _TENANT,
                "file": "fzl_old.txt",
                "page": 1,
                "text": "Федеральный закон 123 старая редакция",
                "meta": {"reg_number": "123", "is_active": False, "revision": "2020"},
            },
        ]
    )
    hits = search(
        "закон 123",
        top_k=2,
        reg_number="123",
        revision_mode="current",
        tenant_id=_TENANT,
    )
    assert hits
    assert hits[0]["file"] == "fzl.txt"


def test_golden_npa_retrieval_historical_mode() -> None:
    index_chunks(
        [
            {
                "sha256": "hist-77",
                "tenant_id": _TENANT,
                "file": "hist.txt",
                "page": 2,
                "text": "приказ 77 утратил силу",
                "meta": {"reg_number": "77", "is_active": False},
            },
        ]
    )
    hits = search(
        "приказ 77",
        top_k=5,
        revision_mode="historical",
        tenant_id=_TENANT,
    )
    assert hits and hits[0]["file"] == "hist.txt"
