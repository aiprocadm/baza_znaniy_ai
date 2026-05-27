"""Minimal exceptions for the qdrant client stub."""


class UnexpectedResponse(Exception):
    """Raised when a collection is not found in the stub client."""

    pass
