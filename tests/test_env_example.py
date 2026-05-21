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
    "QDRANT_URL": "",
    "QDRANT_COLLECTION": "kb_chunks",
    "VECTOR_EMBED_MODEL": "intfloat/multilingual-e5-small",
    "VECTOR_EMBED_DIMENSION": "384",
    "EMBED_BATCH_SIZE": "64",
    "RETRIEVE_TOPK": "10",
    "RERANK_ENABLED": "true",
    "RERANK_TOPK": "50",
    "LLM_PROVIDER": "llama-cpp",
    "LLM_MODEL_NAME": "kb-llama",
    "LLM_MODEL_PATH": "./models/model.gguf",
    "LLM_CTX": "4096",
    "LLM_THREADS": "4",
    "LLM_GPU_LAYERS": "0",
    "LLM_TEMPERATURE": "0.7",
    "LLM_TOP_P": "0.95",
    "LLM_TOP_K": "40",
    "LLM_MAX_TOKENS": "1024",
    "LORA_ADAPTER_PATH": "",
    "LORA_SCALING": "1.0",
    "DOCUMENT_PARSER_BACKEND": "auto",
    "DOCLING_ENABLED": "true",
    "LANGCHAIN_ENABLED": "false",
    "LANGCHAIN_MODE": "legacy",
    "LANGCHAIN_USE_HISTORY_AWARE": "false",
    "LANGCHAIN_RETURN_SOURCE_DOCS": "false",
    "LANGCHAIN_TRACING": "false",
    "LANGCHAIN_PROJECT": "kb-ai",
    "REDIS_URL": "",
    "RATE_LIMIT_BACKEND": "memory",
    "API_KEY_HASH_SALT": "kb-ai-salt",
    "BILLING_ENABLED": "false",
    "BILLING_PROVIDER": "none",
    "POSTGRES_DSN": "",
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
