from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.app import create_api_app
from backend.app.db.utils import init_db


def create_app() -> FastAPI:
    init_db()
    return create_api_app()


app = create_app()

__all__ = ["create_app", "app"]
