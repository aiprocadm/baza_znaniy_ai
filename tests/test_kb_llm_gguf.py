"""Unit tests for the in-process GGUF eval provider adapter."""

from __future__ import annotations

from app.services.kb_llm import (
    GgufEvalProvider,
    _strip_trailing_citation_run,
    select_provider,
)

# Real runaway tails captured from the bundled Qwen2.5-3B on a RAG prompt.
_RUNAWAY_INCREMENT = (
    "Сотрудник работает удалённо не более трёх дней в неделю [3.1]. "
    "Согласование через корпоративный портал [3.2]. "
    "[2] [4] [5] [6] [7] [8] [9] [10] [11] [12]"
)
_RUNAWAY_CYCLE = (
    "Согласование через портал компании. "
    "[2] [1] [4] [5] [3] [4] [5] [1] [3] [2] [5] [4] [1] [3] [2"
)


def test_strip_removes_runaway_citation_tail() -> None:
    assert _strip_trailing_citation_run(_RUNAWAY_INCREMENT) == (
        "Сотрудник работает удалённо не более трёх дней в неделю [3.1]. "
        "Согласование через корпоративный портал [3.2]."
    )
    assert _strip_trailing_citation_run(_RUNAWAY_CYCLE) == ("Согласование через портал компании.")


def test_strip_preserves_legit_citations() -> None:
    # inline + single/double trailing citations (with terminal punctuation) stay
    assert (
        _strip_trailing_citation_run("Срок исковой давности — три года [196].")
        == "Срок исковой давности — три года [196]."
    )
    assert _strip_trailing_citation_run("См. [3.1] и [3.2].") == "См. [3.1] и [3.2]."
    assert _strip_trailing_citation_run("Без цитат вообще.") == "Без цитат вообще."


class _RunawayInner:
    def generate(self, prompt: str, *, context=None) -> str:
        return _RUNAWAY_INCREMENT


def test_gguf_provider_strips_runaway_tail() -> None:
    prov = GgufEvalProvider(model_path="/m.gguf", inner=_RunawayInner())
    resp = prov.generate("q", system="s")
    assert "[5] [6]" not in resp.text
    assert resp.text.endswith("[3.2].")


class _FakeInner:
    """Stand-in for app.llm.llama_cpp_provider.LlamaCppProvider."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, *, context=None) -> str:
        self.calls.append((prompt, dict(context or {})))
        return '{"faithfulness":5,"relevance":5,"completeness":5,"citation":5,"rationale":"ok"}'


def test_adapter_shapes_response_and_folds_system() -> None:
    inner = _FakeInner()
    prov = GgufEvalProvider(model_path="/models/qwen2.5-3b-instruct-q4_k_m.gguf", inner=inner)

    assert prov.name == "gguf"
    assert prov.model == "qwen2.5-3b-instruct-q4_k_m.gguf"

    resp = prov.generate("Вопрос?", system="Ты судья.", temperature=0.0, max_tokens=128)
    assert resp.provider == "gguf"
    assert resp.text.startswith("{") and "faithfulness" in resp.text

    sent_prompt, sent_ctx = inner.calls[0]
    assert "Ты судья." in sent_prompt and "Вопрос?" in sent_prompt
    assert sent_ctx["temperature"] == 0.0 and sent_ctx["max_tokens"] == 128


def test_select_provider_gguf_missing_model_returns_none(tmp_path) -> None:
    prov = select_provider(
        {"KB_LLM_PROVIDER": "gguf", "KB_LLM_GGUF_PATH": str(tmp_path / "absent.gguf")}
    )
    assert prov is None


def test_select_provider_gguf_present_does_not_load(tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF\x00\x00\x00")
    prov = select_provider({"KB_LLM_PROVIDER": "gguf", "KB_LLM_GGUF_PATH": str(model)})
    assert prov is not None and prov.name == "gguf"
