from __future__ import annotations

from celery import group
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.models import Document
from backend.app.schemas.packs import PackRunAcceptedResponse, PackRunRequest
from backend.app.tasks import generate_document_task

router = APIRouter(prefix="/packs", tags=["packs"])


@router.post("/run", status_code=status.HTTP_202_ACCEPTED, response_model=PackRunAcceptedResponse)
def run_pack(
    request: PackRunRequest, db: Session = Depends(get_db)
) -> PackRunAcceptedResponse:
    documents = db.execute(select(Document).order_by(Document.id.asc()).limit(1)).scalars().all()
    if not documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No source documents found")

    signatures = [
        generate_document_task.s(
            template_id=request.pack_id,
            context={},
            document_name=f"pack-{request.pack_id}",
        )
    ]

    task_group = group(signatures, app=generate_document_task.app)
    job = task_group.apply_async()
    batch_id = job.id or task_group.id
    status_url = f"/api/v1/tasks/batches/{batch_id}"

    return PackRunAcceptedResponse(batch_id=batch_id, status_url=status_url)


__all__ = ["router"]
