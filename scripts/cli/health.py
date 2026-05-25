"""kb-cli health — query /api/kb/health and format for humans + cron."""

from __future__ import annotations

import httpx
import typer

health_app = typer.Typer(
    name="health",
    help="Query /api/kb/health and report status with cron-friendly exit codes.",
    add_completion=False,
)

DISK_FREE_WARN_THRESHOLD = 100 * 1024 * 1024  # 100 MB


@health_app.callback(invoke_without_command=True)
def health(
    base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--base-url",
        help="Base URL of the running kb_api server.",
    ),
    timeout: float = typer.Option(5.0, "--timeout", help="HTTP timeout in seconds."),
) -> None:
    """Health-check the running KB API. Exit 0=OK, 1=WARN, 2=FAIL."""

    url = f"{base_url.rstrip('/')}/api/kb/health"
    try:
        resp = httpx.get(url, timeout=timeout)
    except Exception as exc:
        typer.echo(f"FAIL: cannot reach {url}: {exc}", err=True)
        raise typer.Exit(code=2)

    if resp.status_code != 200:
        typer.echo(f"FAIL: HTTP {resp.status_code} from {url}", err=True)
        raise typer.Exit(code=2)

    try:
        data = resp.json()
    except ValueError:
        typer.echo("FAIL: response not JSON", err=True)
        raise typer.Exit(code=2)

    stats = data.get("kb_stats") or {}
    docs = stats.get("documents_count", 0)
    chunks = stats.get("chunks_count", 0)
    db_size = stats.get("db_size_bytes", 0)
    disk_free = stats.get("disk_free_bytes", 0)
    last_idx = stats.get("last_indexed_at") or "never"
    llm = (data.get("llm") or {}).get("selected") or "none"
    compliance = data.get("compliance_mode") or "off"

    summary = (
        f"documents={docs} chunks={chunks} db={db_size}B "
        f"disk_free={disk_free}B llm={llm} compliance={compliance} "
        f"last_indexed={last_idx}"
    )

    if data.get("status") != "ok":
        typer.echo(f"FAIL: {data.get('status')!r}: {summary}", err=True)
        raise typer.Exit(code=2)

    if disk_free and disk_free < DISK_FREE_WARN_THRESHOLD:
        typer.echo(f"WARN: low disk: {summary}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"OK: {summary}")
