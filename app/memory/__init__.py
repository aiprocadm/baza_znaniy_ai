"""Compatibility re-export for legacy DocumentMemory tests."""

from srv.projects.kb.app.memory import DocumentMemory  # type: ignore F401
from srv.projects.kb.app.models import DocumentCreate  # type: ignore F401

__all__ = ["DocumentMemory", "DocumentCreate"]
