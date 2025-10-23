from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.v1.router import api_router
from backend.app.db.utils import init_db


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="Document Generator")
    app.include_router(api_router)
    return app


app = create_app()

__all__ = ["create_app", "app"]
