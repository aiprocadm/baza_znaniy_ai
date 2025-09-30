"""Application entrypoint for running the FastAPI service with Uvicorn."""

from app.main import app  # noqa: F401

__all__ = ["app"]
