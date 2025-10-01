"""Alembic configuration for KB.AI."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlmodel import SQLModel

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.models import file as models  # noqa: F401  # ensure models are imported
from app.models.file import get_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


target_metadata = SQLModel.metadata


def _coerce_sync_sqlite(url: str) -> str:
    if "+aiosqlite" in url:
        return url.replace("+aiosqlite", "")
    return url


def _resolve_url() -> str:
    env_url = os.getenv("DB_URL")
    if env_url:
        return _coerce_sync_sqlite(env_url)
    option_url = config.get_main_option("sqlalchemy.url")
    if option_url:
        return _coerce_sync_sqlite(option_url)
    engine = get_engine(create_schema=False)
    return _coerce_sync_sqlite(str(engine.url))


def run_migrations_offline() -> None:
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = get_engine(_resolve_url(), create_schema=False)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
