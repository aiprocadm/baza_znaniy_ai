from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.models import Tenant
from backend.app.schemas.documents import DocGenerateAcceptedResponse, DocGenerateRequest
from backend.app.tasks import generate_document_task

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocGenerateAcceptedResponse,
)
def generate_document(request: DocGenerateRequest, db: Session = Depends(get_db)) -> DocGenerateAcceptedResponse:
    tenant = db.query(Tenant).filter(Tenant.status == "active").first()
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No active tenant configured")

    task = generate_document_task.delay(
        template_id=request.template_id,
        context=request.context,
        document_name=request.document_name,
    )
    status_url = f"/api/v1/tasks/{task.id}"
    return DocGenerateAcceptedResponse(task_id=task.id, status_url=status_url)
