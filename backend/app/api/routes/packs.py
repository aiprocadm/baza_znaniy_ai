from __future__ import annotations

from celery import group
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.models import Pack, Template
from backend.app.schemas.packs import PackRunAcceptedResponse, PackRunRequest
from backend.app.tasks import generate_document_task

router = APIRouter(prefix="/packs", tags=["packs"])


@router.post("/run", status_code=status.HTTP_202_ACCEPTED, response_model=PackRunAcceptedResponse)
def run_pack(
    request: PackRunRequest, db: Session = Depends(get_db)
) -> PackRunAcceptedResponse:
    pack = db.get(Pack, request.pack_id)
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pack not found")

    items = list(pack.items)
    if not items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pack has no items")

    template_ids = {item.template_id for item in items}
    existing_template_ids = set(
        db.execute(select(Template.id).where(Template.id.in_(template_ids))).scalars().all()
    )
    missing_templates = sorted(template_ids - existing_template_ids)
    if missing_templates:
        missing = ", ".join(missing_templates)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Templates not found: {missing}",
        )

    signatures = [
        generate_document_task.s(
            template_id=item.template_id,
            context=item.context or {},
            document_name=item.document_name,
        )
        for item in items
    ]

    task_group = group(signatures, app=generate_document_task.app)
    job = task_group.apply_async()
    batch_id = job.id or task_group.id
    status_url = f"/api/v1/tasks/batches/{batch_id}"

    return PackRunAcceptedResponse(batch_id=batch_id, status_url=status_url)


__all__ = ["router"]
