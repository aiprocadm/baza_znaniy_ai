"""Minimal Starlette compatibility layer for the test suite."""

from __future__ import annotations

from .datastructures import MutableHeaders, UploadFile
from .requests import Request

__all__ = ["MutableHeaders", "Request", "UploadFile"]
