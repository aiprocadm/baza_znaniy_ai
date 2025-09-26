import os

import httpx

OLLAMA = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = os.getenv("GEN_MODEL", "qwen2.5:3b-instruct")

        codex/split-existing-service-into-containers
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("GEN_MODEL","qwen2.5:3b-instruct")

        main

def ensure_model():
    try:
        with httpx.Client(timeout=60) as c:
            r = c.get(f"{OLLAMA}/api/tags")
            names = [m["name"] for m in r.json().get("models", [])]
        if MODEL in names:
            return
        with httpx.Client(timeout=None) as c:
            c.post(f"{OLLAMA}/api/pull", json={"name": MODEL})
    except Exception:
        pass


def generate(prompt: str) -> str:
    with httpx.Client(timeout=None) as c:
        r = c.post(
            f"{OLLAMA}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json().get("response", "")
