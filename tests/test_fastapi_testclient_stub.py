from __future__ import annotations

from io import BytesIO

from tests.stubs.fastapi import (
    BackgroundTasks,
    FastAPI,
    StreamingResponse,
    TestClient,
    UploadFile,
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


def test_background_tasks_execute_after_response() -> None:
    app = FastAPI()
    executed: list[str] = []

    @app.post("/task")
    def trigger(background_tasks: BackgroundTasks) -> dict[str, str]:
        background_tasks.add_task(executed.append, "done")
        return {"status": "scheduled"}

    client = TestClient(app)
    response = client.post("/task", json={})

    assert response.status_code == 200
    assert response.json() == {"status": "scheduled"}
    assert executed == ["done"]


def test_streaming_response_is_consumed() -> None:
    app = FastAPI()

    @app.get("/stream")
    def stream() -> StreamingResponse:
        async def generator():
            yield b"chunk1"
            yield "chunk2"

        return StreamingResponse(generator(), status_code=201)

    client = TestClient(app)
    response = client.get("/stream")

    assert response.status_code == 201
    assert response.content == b"chunk1chunk2"
    assert response.text == "chunk1chunk2"


