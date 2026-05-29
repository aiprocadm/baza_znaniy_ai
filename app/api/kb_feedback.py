"""Live feedback collection endpoints for W4 (DPO post-training).

Mounted under ``/api/kb`` by the parent ``kb_mvp.router``. Protected
by ``require_api_key`` like the rest of the mutating MVP endpoints.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api.kb_auth import require_api_key
from app.services.kb_store import KnowledgeBaseStore, get_store

router = APIRouter(
    tags=["kb-feedback"],
    dependencies=[Depends(require_api_key)],
)


class FeedbackIn(BaseModel):
    rating: int = Field(..., description="1 = thumbs-up, -1 = thumbs-down")
    comment: Optional[str] = Field(default=None, max_length=2000)
    alternative_answer: Optional[str] = Field(default=None, max_length=4000)
    user_id: Optional[str] = Field(default=None, max_length=128)


class FeedbackOut(BaseModel):
    id: str
    created_at: str


@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackOut,
    status_code=201,
)
def post_feedback(
    body: FeedbackIn,
    message_id: int = Path(ge=1),
    store: KnowledgeBaseStore = Depends(get_store),
) -> FeedbackOut:
    if body.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be -1 or 1")

    with store._connect() as conn:
        row = conn.execute(
            "SELECT conversation_id FROM kb_messages WHERE id = ?",
            (int(message_id),),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="message not found")
    conversation_id = row[0]

    try:
        fid = store.store_feedback(
            conversation_id=conversation_id,
            message_id=int(message_id),
            user_id=body.user_id,
            rating=body.rating,
            comment=body.comment,
            alternative_answer=body.alternative_answer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with store._connect() as conn:
        ts_row = conn.execute("SELECT created_at FROM kb_feedback WHERE id = ?", (fid,)).fetchone()
    created_at = ts_row[0] if ts_row else ""
    return FeedbackOut(id=fid, created_at=created_at)


@router.get("/feedback/export")
def export_feedback(
    store: KnowledgeBaseStore = Depends(get_store),
) -> Response:
    lines: list[str] = []
    for pair in store.iter_feedback_pairs():
        lines.append(pair.to_jsonl_line())
    body = "".join(lines)
    headers = {"X-DPO-Pairs-Count": str(len(lines))}
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers=headers,
    )
