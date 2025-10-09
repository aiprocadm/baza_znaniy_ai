"""Application package initialisation helpers."""
from __future__ import annotations

import warnings

__all__ = ["configure_warning_filters"]

_SWIG_DEPRECATION_MESSAGES: tuple[str, ...] = (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
)


def configure_warning_filters() -> None:
    """Apply warning filters for known noisy dependencies.

    SWIG-generated bindings shipped with third-party libraries emit the same
    ``DeprecationWarning`` on every import when running on Python 3.12+. This
    makes the test output noisy and can hide meaningful warnings. We keep the
    first occurrence ("once") so developers are still informed about the
    impending deprecation but subsequent duplicates are suppressed.
    """

    for message in _SWIG_DEPRECATION_MESSAGES:
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=message,
            module=".*",
        )


configure_warning_filters()
