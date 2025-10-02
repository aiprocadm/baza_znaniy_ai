"""API package exposing the root router for inclusion in the app."""

__all__ = ["api_router"]


def __getattr__(name: str):  # pragma: no cover - trivial lazy import helper
    if name == "api_router":
        from .router import api_router

        return api_router
    raise AttributeError(name)
