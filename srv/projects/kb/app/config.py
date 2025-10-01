"""Compatibility shim for legacy service configuration access."""

from app.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
