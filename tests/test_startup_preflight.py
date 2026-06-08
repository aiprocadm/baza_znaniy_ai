"""Preflight produces one readable status line and never raises."""

from __future__ import annotations

from app.services.startup_preflight import format_preflight


def test_format_includes_llm_embedder_mode():
    line = format_preflight(
        llm_name="gguf", llm_model="qwen2.5-3b", embedder_name="e5-small", mode="bundled"
    )
    assert "gguf" in line and "qwen2.5-3b" in line
    assert "e5-small" in line
    assert "bundled" in line.lower()


def test_format_handles_missing_llm():
    line = format_preflight(llm_name=None, llm_model=None, embedder_name="hash", mode="degraded")
    assert "extractive" in line.lower() or "none" in line.lower()
    assert "hash" in line
