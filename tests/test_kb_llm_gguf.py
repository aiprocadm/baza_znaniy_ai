"""Unit tests for the in-process GGUF eval provider adapter."""

from __future__ import annotations

from app.services.kb_llm import GgufEvalProvider, select_provider


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


def test_default_max_tokens_used_when_caller_passes_none() -> None:
    inner = _FakeInner()
    prov = GgufEvalProvider(model_path="/m.gguf", inner=inner, default_max_tokens=90)

    prov.generate("Вопрос?")  # no explicit max_tokens

    _, sent_ctx = inner.calls[0]
    assert sent_ctx["max_tokens"] == 90


def test_kb_llm_max_tokens_env_sets_gguf_default(tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF\x00\x00\x00")
    prov = select_provider(
        {
            "KB_LLM_PROVIDER": "gguf",
            "KB_LLM_GGUF_PATH": str(model),
            "KB_LLM_MAX_TOKENS": "90",
        }
    )
    assert prov is not None and prov._default_max_tokens == 90


def test_invalid_kb_llm_max_tokens_falls_back_to_512(tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF\x00\x00\x00")
    prov = select_provider(
        {
            "KB_LLM_PROVIDER": "gguf",
            "KB_LLM_GGUF_PATH": str(model),
            "KB_LLM_MAX_TOKENS": "not-a-number",
        }
    )
    assert prov is not None and prov._default_max_tokens == 512


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
