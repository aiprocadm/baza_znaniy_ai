"""Contract tests for the in-tree ``qdrant_client.http.models`` stub.

The stub lives at ``tests/stubs/qdrant_client/`` and is used whenever the
real package is not importable. ``app/retriever/qdrant.py`` imports the
following symbols from ``qdrant_client.http.models`` (resolved
dynamically via ``qmodels.<name>`` at call time, so missing symbols
fail late with ``AttributeError`` deep inside production code):

* ``Distance``, ``VectorParams``, ``HnswConfigDiff`` — schema setup
* ``PayloadSchemaType`` (with the ``BOOL`` / ``BOOLEAN`` rename
  resolved at production import time)
* ``PointStruct`` — upsert payload shape
* ``FieldCondition``, ``Filter``, ``MatchValue``, ``MatchText`` —
  query-filter composition (see ``_to_qdrant_filter`` at line 305)
* ``CreateAlias``, ``CreateAliasOperation``, ``DeleteAlias``,
  ``DeleteAliasOperation`` — alias-management helpers
  (``create_alias`` / ``switch_alias`` / ``delete_alias``)

If any of these go missing from the stub, ~10 test files start
failing with non-obvious ``AttributeError`` at the call site rather
than a clear import-time error. This contract test is the early
sentinel.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

PROD_QDRANT_PY = Path(__file__).resolve().parent.parent / "app" / "retriever" / "qdrant.py"
STUBS_PATH = Path(__file__).resolve().parent / "stubs"


@pytest.fixture
def qmodels(monkeypatch):
    """Force-import the on-disk stub, bypassing any inline ``sys.modules`` shims.

    ``tests/test_vector_stores.py`` installs an *incomplete* inline
    ``qdrant_client.http.models`` module at import time. Without this
    fixture, our contract assertions would fire against that incomplete
    module rather than the real ``tests/stubs/qdrant_client/...``
    package. Mirror the discipline from ``test_prometheus_stub_contract.py``.
    """

    for name in list(sys.modules):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(STUBS_PATH))
    module = importlib.import_module("qdrant_client.http.models")
    assert "stubs" in (
        module.__file__ or ""
    ), f"Fixture should resolve to the in-tree stub, got {module.__file__!r}"
    return module


_EXPECTED_SYMBOLS = {
    # Schema / collection setup
    "Distance",
    "VectorParams",
    "HnswConfigDiff",
    "PayloadSchemaType",
    "PointStruct",
    # Filter composition
    "FieldCondition",
    "Filter",
    "MatchValue",
    "MatchText",
    # Alias management
    "CreateAlias",
    "CreateAliasOperation",
    "DeleteAlias",
    "DeleteAliasOperation",
}


def test_stub_exposes_all_symbols_used_by_production(qmodels):
    """Every ``qmodels.<X>`` reference in production must resolve in the stub."""

    missing = sorted(name for name in _EXPECTED_SYMBOLS if not hasattr(qmodels, name))
    assert not missing, (
        f"qdrant_client.http.models stub is missing: {missing}. "
        f"Add the dataclass(es) to tests/stubs/qdrant_client/http/models.py."
    )


def test_expected_symbols_matches_production_usage():
    """Guard against silent drift between this test's allow-list and prod code."""

    src = PROD_QDRANT_PY.read_text(encoding="utf-8")
    used_in_prod = set(re.findall(r"qmodels\.(\w+)", src))
    drift = used_in_prod - _EXPECTED_SYMBOLS
    assert not drift, (
        f"Production code uses qmodels.<{sorted(drift)}> but they are not in "
        f"the contract-test allow-list. Add them to _EXPECTED_SYMBOLS in this "
        f"file and to tests/stubs/qdrant_client/http/models.py."
    )


def test_match_value_stores_value_attribute(qmodels):
    match = qmodels.MatchValue(value="tenant-a")
    assert match.value == "tenant-a"


def test_match_text_stores_text_attribute(qmodels):
    match = qmodels.MatchText(text="минюст")
    assert match.text == "минюст"


def test_field_condition_carries_key_and_match(qmodels):
    cond = qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value="t1"))
    assert cond.key == "tenant_id"
    assert isinstance(cond.match, qmodels.MatchValue)
    assert cond.match.value == "t1"


def test_filter_must_default_is_none(qmodels):
    """Filter without explicit ``must`` is permissive — stub mirrors that."""

    flt = qmodels.Filter()
    assert flt.must is None
    assert flt.should is None
    assert flt.must_not is None


def test_payload_schema_type_keeps_bool_alias(qmodels):
    """Production resolves BOOL via getattr; stub keeps the historical name."""

    assert getattr(qmodels.PayloadSchemaType, "BOOL", None) is not None


def test_create_alias_operation_wraps_create_alias(qmodels):
    op = qmodels.CreateAliasOperation(
        create_alias=qmodels.CreateAlias(collection_name="kb_v2", alias_name="active")
    )
    assert op.create_alias.collection_name == "kb_v2"
    assert op.create_alias.alias_name == "active"


def test_delete_alias_operation_wraps_delete_alias(qmodels):
    op = qmodels.DeleteAliasOperation(delete_alias=qmodels.DeleteAlias(alias_name="old"))
    assert op.delete_alias.alias_name == "old"
