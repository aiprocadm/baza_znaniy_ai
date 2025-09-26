"""Application settings and helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration loaded from the environment."""

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_reload: bool = Field(default=False, alias="APP_RELOAD")
    data_dir: Path = Field(default=Path("/data/storage"), alias="DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        populate_by_name = True


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    data = {k: v for k, v in os.environ.items() if k.startswith("APP_") or k in {"DATA_DIR", "LOG_LEVEL"}}
    return Settings.model_validate(data)
