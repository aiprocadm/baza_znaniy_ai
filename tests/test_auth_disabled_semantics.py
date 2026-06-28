"""Pin the unified "is auth disabled via env" semantics.

Historically two helpers disagreed in an edge case:

* ``app.core.config._default_auth_disabled`` used an ``or``-chain that only
  inspected the *first non-empty* env value, so ``AUTH_DISABLED_FOR_TESTS=0``
  shadowed a later ``AUTH_DISABLED=1`` and returned ``False``.
* ``app.core.auth._env_auth_disabled`` treated *any* truthy key as a disable
  signal and returned ``True`` for the same environment.

The canonical rule is "auth is disabled if ANY recognised key is truthy"
(matching the looser auth.py check, which already governs every call site via
``settings.auth_disabled or _env_auth_disabled()``). These tests lock that in
and prove both helpers draw from one shared implementation.
"""

from __future__ import annotations

import pytest

from app.core import auth as auth_module
from app.core import config as config_module

_ALL_KEYS = (
    "AUTH_DISABLED_FOR_TESTS",
    "AUTH_DISABLED",
    "DISABLE_AUTH",
    "AUTH_DISABLE",
    "KB_DISABLE_AUTH",
)


@pytest.fixture(autouse=True)
def _clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a clean slate for all recognised keys."""

    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_single_source_of_truth() -> None:
    """auth.py must delegate to the callable defined in config, with no private copy.

    Object identity is avoided on purpose: other tests reload the config module,
    which rebinds ``config._env_auth_disabled`` to a fresh (behaviourally
    identical) object while auth keeps the original reference. ``__module__`` is
    stable across reloads and still proves the function originates in config.
    """

    assert auth_module._env_auth_disabled.__module__ == "app.core.config"
    # auth must not keep its own divergent copy of the key list / truthy set.
    assert not hasattr(auth_module, "_AUTH_DISABLED_ENV_KEYS")
    assert not hasattr(auth_module, "_TRUTHY_ENV_VALUES")


def test_key_list_and_truthy_set_live_in_config() -> None:
    """The key list and truthy set are defined once, in config."""

    assert tuple(config_module._AUTH_DISABLED_ENV_KEYS) == _ALL_KEYS
    assert config_module._TRUTHY_ENV_VALUES == {"1", "true", "yes", "on"}


def test_no_env_means_not_disabled() -> None:
    assert config_module._default_auth_disabled() is False
    assert auth_module._env_auth_disabled() is False


def test_zero_then_one_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """The historic divergence: a falsey first key, a truthy later key.

    Both helpers must now agree that auth is disabled.
    """

    monkeypatch.setenv("AUTH_DISABLED_FOR_TESTS", "0")
    monkeypatch.setenv("AUTH_DISABLED", "1")

    assert auth_module._env_auth_disabled() is True
    assert config_module._default_auth_disabled() is True


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_each_key_disables_independently(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(key, "1")

    assert auth_module._env_auth_disabled() is True
    assert config_module._default_auth_disabled() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_truthy_values_disable(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_DISABLED", value)

    assert auth_module._env_auth_disabled() is True
    assert config_module._default_auth_disabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_falsey_values_do_not_disable(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_DISABLED", value)

    assert auth_module._env_auth_disabled() is False
    assert config_module._default_auth_disabled() is False


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"AUTH_DISABLED_FOR_TESTS": "0", "AUTH_DISABLED": "1"}, True),
        ({"AUTH_DISABLED_FOR_TESTS": "1", "AUTH_DISABLED": "0"}, True),
        ({"AUTH_DISABLED": "0", "KB_DISABLE_AUTH": "yes"}, True),
        ({"AUTH_DISABLED": "0", "DISABLE_AUTH": "0"}, False),
    ],
)
def test_both_helpers_agree_across_matrix(
    env: dict[str, str], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert auth_module._env_auth_disabled() is expected
    assert config_module._default_auth_disabled() is expected


# ---------------------------------------------------------------------------
# Field-level authority: ``Settings.auth_disabled`` must, on its own, reflect
# the canonical env contract — not the settings shim's first-matching-alias
# pick (which previously diverged) nor pydantic's looser bool coercion.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def test_field_reflects_reverse_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falsey first alias + truthy later key: the field itself must read True.

    This is the residual the function-level fix alone did not close: the shim
    eager-override picks ``AUTH_DISABLED=0`` (first match) and would leave the
    field ``False`` despite ``AUTH_DISABLED_FOR_TESTS=1``.
    """

    monkeypatch.setenv("AUTH_DISABLED", "0")
    monkeypatch.setenv("AUTH_DISABLED_FOR_TESTS", "1")

    assert config_module.Settings().auth_disabled is True


def test_field_false_when_all_env_falsey(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_DISABLED", "0")

    assert config_module.Settings().auth_disabled is False


def test_field_false_when_no_env() -> None:
    assert config_module.Settings().auth_disabled is False


def test_field_uses_canonical_truthy_set_not_pydantic_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``"y"`` is truthy to pydantic but not to our canonical set.

    The field must follow the canonical set and stay ``False`` (auth stays on —
    the fail-closed direction).
    """

    monkeypatch.setenv("AUTH_DISABLED", "y")

    assert config_module.Settings().auth_disabled is False


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_field_true_for_each_key(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(key, "1")

    assert config_module.Settings().auth_disabled is True


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"AUTH_DISABLED": "1"},
        {"AUTH_DISABLED": "0"},
        {"AUTH_DISABLED": "0", "AUTH_DISABLED_FOR_TESTS": "1"},
        {"AUTH_DISABLED": "1", "AUTH_DISABLED_FOR_TESTS": "0"},
        {"KB_DISABLE_AUTH": "on"},
        {"AUTH_DISABLED": "y"},
    ],
)
def test_field_equals_env_helper(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert config_module.Settings().auth_disabled is config_module._env_auth_disabled()


def test_explicit_constructor_value_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``auth_disabled=`` arg wins when no env key is set."""

    assert config_module.Settings(auth_disabled=True).auth_disabled is True
