from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_alembic_upgrade_and_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "smoke.db"
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{db_path}"

    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
    subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "-1"],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
