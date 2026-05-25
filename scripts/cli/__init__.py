"""kb-cli subcommands package.

Each subcommand lives in its own module (backup.py, restore.py,
reindex.py, health.py) and exposes a Typer ``app`` instance that
the top-level ``scripts/kb_cli.py`` mounts as a subcommand group.
"""
