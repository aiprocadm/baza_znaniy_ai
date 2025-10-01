from __future__ import annotations

from pathlib import Path
import warnings

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


EXPECTED_DEFAULTS = {
    "APP_ENV": "development",
    "APP_HOST": "0.0.0.0",
    "APP_PORT": "8000",
    "DATA_DIR": "./var/data",
    "DB_URL": "sqlite+aiosqlite:///./var/data/kb.sqlite",
    "MAX_UPLOAD_MB": "40",
    "UPLOAD_ALLOWED_EXTS": "pdf,docx,pptx,xlsx,txt,md",
    "VECTOR_BACKEND": "qdrant",
    "QDRANT_URL": "http://qdrant:6333",
    "QDRANT_COLLECTION": "kb_chunks",
    "VECTOR_EMBED_MODEL": "intfloat/multilingual-e5-small",
    "VECTOR_EMBED_DIMENSION": "384",
    "EMBED_BATCH_SIZE": "32",
    "RETRIEVE_TOPK": "10",
    "RERANK_ENABLED": "false",
    "RERANK_TOPK": "10",
    "LLM_PROVIDER": "ollama",
    "LLM_MODEL_NAME": "llama3.1:8b",
    "OLLAMA_MODEL": "llama3.1:8b",
    "OLLAMA_BASE_URL": "http://ollama:11434",
    "MAX_CONTEXT_TOKENS": "6000",
    "MAX_GENERATION_TOKENS": "1024",
}

EXPECTED_EXTENSION_SET = {"pdf", "docx", "pptx", "xlsx", "txt", "md"}


def test_env_example_contains_unique_keys() -> None:
    """The sample environment file should not contain duplicated keys."""

    lines = [line for line in ENV_EXAMPLE.read_text().splitlines() if line and not line.lstrip().startswith("#")]
    keys = [line.split("=", 1)[0].strip() for line in lines]
    assert len(keys) == len(set(keys)), "Duplicate keys found in .env.example"


def test_env_example_defaults_match_specification() -> None:
    """Loading the example env file should yield the documented defaults."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = dotenv_values(ENV_EXAMPLE)

    assert values, ".env.example should load into key/value pairs"
    assert not caught, ".env.example should not trigger warnings when parsed"

    for key, expected in EXPECTED_DEFAULTS.items():
        assert values.get(key) == expected, f"Unexpected default for {key}"

    extensions = {piece.strip() for piece in values["UPLOAD_ALLOWED_EXTS"].split(",") if piece.strip()}
    assert extensions == EXPECTED_EXTENSION_SET

    # Ensure reranking switches are present for clarity.
    assert values["RERANK_ENABLED"] in {"true", "false"}
    assert values["RERANK_TOPK"].isdigit()
