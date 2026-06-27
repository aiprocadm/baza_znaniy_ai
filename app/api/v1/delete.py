from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.deps import get_data_dir, get_ingest_session
from app.models.user import UserRecord
from app.models import DeleteResponse
from app.models.file import FileRecord

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["files"])


def _is_within_data_dir(path: Path, data_dir: Path) -> bool:
    """Return whether ``path`` resolves inside ``data_dir``.

    ``record.path`` is persisted data; a tampered or buggy row must never let
    the delete handler unlink files outside the storage root.
    """

    try:
        return path.resolve().is_relative_to(data_dir.resolve())
    except (OSError, ValueError):  # pragma: no cover - defensive resolution guard
        return False


@router.delete("/file/{file_id}", response_model=DeleteResponse)
def delete_file(
    file_id: str,
    _: UserRecord = Depends(get_current_active_user),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
    data_dir: Path = Depends(get_data_dir),
) -> DeleteResponse:
    """Delete file metadata and remove the file from storage."""

    try:
        record_id = int(file_id)
    except (TypeError, ValueError):  # pragma: no cover - invalid identifier
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND") from None

    record = session.get(FileRecord, record_id)
    if record is None or record.tenant_id != tenant:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND")

    path = Path(record.path)
    if _is_within_data_dir(path, data_dir):
        try:
            path.unlink()
        except FileNotFoundError:  # pragma: no cover - best effort cleanup
            pass
    else:
        LOGGER.warning(
            "Refusing to unlink file outside DATA_DIR: %s (file_id=%s, tenant=%s)",
            path,
            record_id,
            tenant,
        )

    session.delete(record)
    session.commit()

    return DeleteResponse(ok=True, id=str(file_id))
