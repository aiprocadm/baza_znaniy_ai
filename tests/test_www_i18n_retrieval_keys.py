"""The end-user banner copy + wiring exist (guards against typos / missing keys)."""

from __future__ import annotations

import json
from pathlib import Path

WWW = Path(__file__).resolve().parents[1] / "data" / "www"

REQUIRED_KEYS = [
    "retrieval.banner.title",
    "retrieval.reason.hashing_embedder",
    "retrieval.reason.embedding_dim_mismatch",
    "retrieval.reason.search_truncated",
    "retrieval.reason.vector_backend_down",
]


def test_ru_json_has_retrieval_banner_keys():
    dictionary = json.loads((WWW / "i18n" / "ru.json").read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in dictionary]
    assert not missing, f"missing i18n keys: {missing}"


def test_index_html_wires_degradation_banner():
    html = (WWW / "index.html").read_text(encoding="utf-8")
    assert 'id="ask-degraded"' in html
    assert "renderDegradation" in html
