from __future__ import annotations

import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts import download_model


class QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses standard logging."""

    def log_message(
        self, format: str, *args
    ) -> None:  # noqa: D401, ANN001 - signature mandated by base class
        """No-op override to keep test output clean."""
        return


@pytest.fixture()
def file_server(tmp_path: Path):
    root = tmp_path / "server"
    root.mkdir()

    def start_server() -> tuple[ThreadingHTTPServer, threading.Thread]:
        handler = partial(QuietHandler, directory=str(root))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    return root, start_server


def test_download_via_http_roundtrip(tmp_path: Path, file_server):
    root, start_server = file_server
    data = b"tiny-model"
    expected_sha = hashlib.sha256(data).hexdigest()
    (root / "model.gguf").write_bytes(data)

    server, thread = start_server()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/model.gguf"
        output = tmp_path / "models" / "model.gguf"
        result = download_model.download_via_http(
            url, output, expected_sha, max_retries=1, timeout=5
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert output.read_bytes() == data
    assert result.sha256 == expected_sha
    assert result.bytes_written == len(data)


def test_cli_populates_manifest_sha(tmp_path: Path, file_server):
    root, start_server = file_server
    data = b"another-model"
    (root / "model.gguf").write_bytes(data)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "default": {
                    "url": "http://placeholder",
                    "sha256": None,
                }
            }
        ),
        "utf-8",
    )

    output = tmp_path / "downloaded" / "model.gguf"

    server, thread = start_server()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/model.gguf"
        exit_code = download_model.main(
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--allow-missing-hash",
                "--target",
                "default",
                "--url",
                url,
                "--max-retries",
                "1",
                "--timeout",
                "5",
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert exit_code == 0
    assert output.read_bytes() == data
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["default"]["sha256"] == hashlib.sha256(data).hexdigest()
