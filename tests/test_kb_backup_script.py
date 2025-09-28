import os
import subprocess
import tarfile
from pathlib import Path


def test_kb_backup_creates_expected_archive(tmp_path):
    project_root = tmp_path / "srv" / "projects" / "kb"
    backup_root = tmp_path / "srv" / "backups"
    log_file = tmp_path / "logs" / "kb_backup.log"

    # Prepare fake project tree
    (project_root / "data" / "db").mkdir(parents=True)
    (project_root / "data" / "storage").mkdir(parents=True)
    (project_root / "data" / "www").mkdir(parents=True)
    (project_root / "data").mkdir(exist_ok=True)
    (project_root / "data" / "ssl").mkdir(parents=True)

    (project_root / ".env").write_text("KEY=value\n")
    (project_root / "data" / "db" / "dummy.sql").write_text("SELECT 1;\n")
    (project_root / "data" / "storage" / "file.txt").write_text("data\n")
    (project_root / "data" / "www" / "index.html").write_text("<html></html>\n")
    (project_root / "data" / "nginx.conf").write_text("server {}\n")

    basic_user = "admin"
    htpasswd_path = project_root / "data" / "ssl" / basic_user
    htpasswd_path.write_text("admin:hashedpassword\n")

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project_root),
            "BACKUP_ROOT": str(backup_root),
            "LOG_FILE": str(log_file),
            "BASIC_USER": basic_user,
            "APP_PORT": "18000",
        }
    )

    script_path = Path("srv/projects/kb/data/scripts/kb_backup.sh").resolve()
    result = subprocess.run(
        ["bash", str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Backup created at" in result.stdout

    backup_dirs = list(backup_root.iterdir())
    assert len(backup_dirs) == 1
    dest_dir = backup_dirs[0]
    archive_path = dest_dir / "kb.tar.gz"
    assert archive_path.is_file()

    with tarfile.open(archive_path) as tar:
        members = {name.rstrip("/") for name in tar.getnames()}

    expected_paths = {
        ".env",
        "data/db",
        "data/storage",
        "data/www",
        "data/nginx.conf",
        f"data/ssl/{basic_user}",
    }
    assert expected_paths.issubset(members)

    assert log_file.is_file()
    log_content = log_file.read_text()
    assert "EXIT_CODE=0" in log_content
