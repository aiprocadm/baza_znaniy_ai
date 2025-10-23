from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    """Application configuration loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    database_url: str = Field(default="sqlite:///./backend.db")
    s3_bucket: str = Field(default="documents")
    s3_region: str = Field(default="us-east-1")
    s3_endpoint_url: str | None = Field(default=None)
    s3_access_key: str = Field(default="test")
    s3_secret_key: str = Field(default="test")
    celery_broker_url: str = Field(default="memory://")
    celery_result_backend: str = Field(default="cache+memory://")
    celery_task_eager: bool = Field(default=False)
    json_max_bytes: int = Field(default=1024 * 1024)
    upload_max_bytes: int = Field(default=20 * 1024 * 1024)

    @field_validator("celery_task_eager", mode="before")
    @classmethod
    def _parse_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        value_str = str(value).strip().lower()
        return value_str in {"1", "true", "t", "yes", "on"}


def _load_settings() -> Settings:
    defaults = Settings.model_fields
    return Settings(
        database_url=os.getenv("BACKEND_DATABASE_URL", defaults["database_url"].default),
        s3_bucket=os.getenv("BACKEND_S3_BUCKET", defaults["s3_bucket"].default),
        s3_region=os.getenv("BACKEND_S3_REGION", defaults["s3_region"].default),
        s3_endpoint_url=os.getenv("BACKEND_S3_ENDPOINT_URL", defaults["s3_endpoint_url"].default),
        s3_access_key=os.getenv("BACKEND_S3_ACCESS_KEY", defaults["s3_access_key"].default),
        s3_secret_key=os.getenv("BACKEND_S3_SECRET_KEY", defaults["s3_secret_key"].default),
        celery_broker_url=os.getenv("BACKEND_CELERY_BROKER_URL", defaults["celery_broker_url"].default),
        celery_result_backend=os.getenv(
            "BACKEND_CELERY_RESULT_BACKEND", defaults["celery_result_backend"].default
        ),
        celery_task_eager=os.getenv("CELERY_TASK_ALWAYS_EAGER", "0"),
        json_max_bytes=os.getenv("BACKEND_JSON_MAX_BYTES", defaults["json_max_bytes"].default),
        upload_max_bytes=os.getenv("BACKEND_UPLOAD_MAX_BYTES", defaults["upload_max_bytes"].default),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return _load_settings()


def reset_settings_cache() -> None:
    """Reset cached settings (useful for tests)."""

    get_settings.cache_clear()
