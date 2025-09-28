"""Application settings and helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Type

from pydantic import BaseModel, Field, FieldInfo


class Settings(BaseModel):
    """Runtime configuration loaded from the environment."""

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_reload: bool = Field(default=False, alias="APP_RELOAD")
    data_dir: Path = Field(default=Path("/data/storage"), alias="DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        populate_by_name = True


def _build_field_metadata() -> Dict[str, Dict[str, Any]]:
    """Return settings field metadata keyed by attribute name."""

    metadata: Dict[str, Dict[str, Any]] = {}
    for name, annotation in Settings.__annotations__.items():
        field = getattr(Settings, name, None)
        alias = name.upper()
        if isinstance(field, FieldInfo) and field.metadata is not None:
            alias = field.metadata.get("alias", alias)
        metadata[name] = {"alias": alias, "annotation": annotation}
    return metadata


_FIELD_METADATA = _build_field_metadata()


def _coerce_value(value: str, annotation: Type[Any]) -> Any:
    """Convert environment strings to the expected field types."""

    if annotation is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(value)
    if annotation is Path:
        return Path(value)
    return value


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    overrides: Dict[str, Any] = {}
    for name, info in _FIELD_METADATA.items():
        alias = info["alias"]
        if alias in os.environ:
            overrides[name] = _coerce_value(os.environ[alias], info["annotation"])
    return Settings(**overrides)
