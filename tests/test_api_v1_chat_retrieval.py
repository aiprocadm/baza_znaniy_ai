"""PR2b: the v1 ChatResponse carries an optional per-query retrieval report."""

from __future__ import annotations

from app.models import ChatResponse


def _base_kwargs() -> dict:
    return {
        "answer": "ok",
        "citations": [],
        "conversation_id": "c1",
        "citations_insufficient": False,
        "latency_ms": 1.0,
    }


def test_chat_response_retrieval_defaults_to_none():
    resp = ChatResponse(**_base_kwargs())
    assert resp.retrieval is None


def test_chat_response_coerces_retrieval_dict():
    resp = ChatResponse(
        **_base_kwargs(),
        retrieval={
            "degraded": True,
            "severity": "critical",
            "reasons": [
                {"reason": "vector_backend_down", "severity": "critical", "detail": "boom"}
            ],
        },
    )
    assert resp.retrieval is not None
    assert resp.retrieval.degraded is True
    assert resp.retrieval.severity == "critical"
    assert resp.retrieval.reasons[0].reason == "vector_backend_down"
    assert resp.retrieval.reasons[0].detail == "boom"
