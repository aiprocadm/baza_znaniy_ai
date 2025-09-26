import os, httpx

OLLAMA = "http://127.0.0.1:11434"
MODEL = os.getenv("GEN_MODEL","qwen2.5:3b-instruct")

def ensure_model():
    try:
        with httpx.Client(timeout=60) as c:
            r = c.get(f"{OLLAMA}/api/tags")
            names = [m["name"] for m in r.json().get("models",[])]
        if MODEL in names:
            return
        with httpx.Client(timeout=None) as c:
            c.post(f"{OLLAMA}/api/pull", json={"name": MODEL})
    except Exception:
        pass

def generate(prompt: str) -> str:
    with httpx.Client(timeout=None) as c:
        r = c.post(f"{OLLAMA}/api/generate", json={"model":MODEL,"prompt":prompt,"stream":False})
        r.raise_for_status()
        return r.json().get("response","")
