"""Stub for fastapi.concurrency module."""

from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Stub implementation - just calls the function directly."""
    return func(*args, **kwargs)
