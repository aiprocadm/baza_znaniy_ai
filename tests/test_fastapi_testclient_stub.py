from __future__ import annotations

from io import BytesIO

import pytest

from tests.stubs.fastapi import FastAPI, TestClient, UploadFile
from tests.stubs.fastapi.uploads import (
    build_upload_file,
    coerce_uploads,
    ensure_list,
    normalise_file_entries,
)


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


def test_testclient_supports_additional_http_methods() -> None:
    app = FastAPI()

    @app.put("/items")
    def update_item() -> dict[str, str]:
        return {"method": "PUT"}

    @app.patch("/items")
    def patch_item() -> dict[str, str]:
        return {"method": "PATCH"}

    @app.delete("/items")
    def delete_item() -> tuple[dict[str, str], int]:
        return {"method": "DELETE"}, 202

    @app.options("/items")
    def options_item() -> dict[str, str]:
        return {"method": "OPTIONS"}

    @app.head("/items")
    def head_item() -> dict[str, str]:
        return {"method": "HEAD"}

    client = TestClient(app)

    assert client.put("/items").json()["method"] == "PUT"
    assert client.patch("/items").json()["method"] == "PATCH"
    response = client.delete("/items")
    assert response.status_code == 202
    assert response.json()["method"] == "DELETE"
    assert client.options("/items").json()["method"] == "OPTIONS"
    assert client.head("/items").status_code == 200


def test_router_includes_child_router_with_prefix() -> None:
    child = FastAPI()

    @child.get("/child")
    def child_route() -> dict[str, str]:
        return {"scope": "child"}

    parent = FastAPI()
    parent.include_router(child, prefix="/parent")

    client = TestClient(parent)

    assert client.get("/parent/child").json() == {"scope": "child"}


@pytest.mark.parametrize(
    "value,expected",
    [
        (("file.txt", b"payload"), [("file.txt", b"payload")]),
        ((["nested.txt", b"bytes"],), [["nested.txt", b"bytes"]]),
        ((("tuple.txt", b"data"),), [("tuple.txt", b"data")]),
    ],
)
def test_normalise_file_entries(value, expected):
    assert normalise_file_entries(value) == expected


def test_ensure_list_wraps_non_list_values() -> None:
    assert ensure_list(1) == [1]
    assert ensure_list([1, 2]) == [1, 2]


def test_coerce_uploads_builds_uploadfile_instances() -> None:
    uploads = coerce_uploads(("sample.txt", b"payload", "text/plain"))
    assert len(uploads) == 1
    upload = uploads[0]
    assert isinstance(upload, UploadFile)
    assert upload.filename == "sample.txt"
    assert upload.content_type == "text/plain"


def test_build_upload_file_preserves_uploadfile_instance() -> None:
    original = UploadFile(filename="existing.txt")
    assert build_upload_file(original) is original


