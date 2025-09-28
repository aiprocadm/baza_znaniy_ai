"""Compatibility helpers for third-party expectations of pydantic v1."""

ENCODERS_BY_TYPE: dict[type, object] = {}

__all__ = ["ENCODERS_BY_TYPE"]
