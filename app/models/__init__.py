"""Compatibility re-export for shared document models."""

from srv.projects.kb.app.models import DocumentCreate  # type: ignore F401

__all__ = ["DocumentCreate"]
