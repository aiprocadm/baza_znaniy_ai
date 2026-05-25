"""kb-cli entrypoint.

After ``pip install -e .`` (Sprint 1.6 wires the ``[project.scripts]``
entry), the ``kb-cli`` command is available on PATH and dispatches to
the subcommands in :mod:`scripts.cli`.

Until then, run with ``py -m scripts.kb_cli <subcommand>``.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="kb-cli",
    help="Operations CLI for KB.AI (backup, restore, reindex, health).",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root_callback() -> None:
    """Root callback — kept thin; subcommands attach via decorators below."""


def main() -> None:
    """Console-script entry point referenced from pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
