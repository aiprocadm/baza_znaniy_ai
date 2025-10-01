import asyncio
import pytest

starlette_datastructures = pytest.importorskip("starlette.datastructures")
StarletteUploadFile = starlette_datastructures.UploadFile

from app.api import routes as routes_module
from app.api import upload_utils


def test_coerce_upload_file_handles_tuple_with_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"example-bytes"

    # Ensure the helper constructs a real UploadFile from Starlette when available.
    monkeypatch.setattr(upload_utils, "UploadFile", StarletteUploadFile)
    monkeypatch.setattr(routes_module, "UploadFile", StarletteUploadFile)

    result = routes_module._coerce_upload_file([("document.txt", payload)])

    assert isinstance(result, StarletteUploadFile)
    assert result.filename == "document.txt"
    assert asyncio.run(result.read()) == payload
