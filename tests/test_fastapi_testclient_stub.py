from __future__ import annotations

from io import BytesIO

from tests.stubs.fastapi import FastAPI, TestClient, UploadFile


def test_testclient_post_handles_multiple_file_entries() -> None:
    app = FastAPI()

    @app.post("/upload")
    def upload(file: list[UploadFile]) -> dict[str, list[str]]:
        assert all(isinstance(item, UploadFile) for item in file)
        return {"filenames": [item.filename for item in file]}

    client = TestClient(app)

    response = client.post(
        "/upload",
        files={
            "file": [
                ("first.txt", BytesIO(b"first"), "text/plain"),
                ("second.txt", b"second", "text/plain"),
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {"filenames": ["first.txt", "second.txt"]}


