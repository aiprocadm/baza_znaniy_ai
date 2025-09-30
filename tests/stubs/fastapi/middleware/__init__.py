"""Middleware namespace for the FastAPI compatibility layer."""

from .cors import CORSMiddleware

__all__ = ["CORSMiddleware"]
