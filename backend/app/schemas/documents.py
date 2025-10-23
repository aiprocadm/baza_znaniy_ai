from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocGenerateRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    document_name: str | None = Field(None, max_length=255)
    context: dict[str, Any] = Field(default_factory=dict)


class DocGenerateAcceptedResponse(BaseModel):
    task_id: str
    status_url: str
