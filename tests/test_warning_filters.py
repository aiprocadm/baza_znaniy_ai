from __future__ import annotations

import importlib
import warnings


def test_swig_deprecation_warnings_are_suppressed() -> None:
    module = importlib.import_module("app")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        module.configure_warning_filters()

        for _ in range(3):
            warnings.warn(
                "builtin type SwigPyObject has no __module__ attribute",
                DeprecationWarning,
            )

        warnings.warn("a different warning", DeprecationWarning)

    assert [warning.message.args[0] for warning in caught] == [
        "a different warning",
    ]
