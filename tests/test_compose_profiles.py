"""Each compose profile yields a coherent, non-contradictory env set."""

from __future__ import annotations

import pathlib

import yaml


def _services_env(path: str) -> dict:
    data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    svc = next(iter(data["services"].values()))
    env = svc.get("environment", {})
    if isinstance(env, list):
        env = dict(e.split("=", 1) for e in env)
    return env


def test_api_profile_disables_local_gguf():
    env = _services_env("compose.api.yml")
    assert str(env.get("KB_LLM_LOCAL_FALLBACK", "")).lower() in {"0", "false", "no", "off"}


def test_gpu_profile_requests_gpu_layers():
    env = _services_env("compose.gpu.yml")
    assert int(env.get("LLM_GPU_LAYERS", "0")) > 0
