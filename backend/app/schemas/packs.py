from __future__ import annotations

from pydantic import BaseModel, Field


class PackRunRequest(BaseModel):
    pack_id: int = Field(..., ge=1)


class PackRunAcceptedResponse(BaseModel):
    batch_id: str
    status_url: str


__all__ = ["PackRunRequest", "PackRunAcceptedResponse"]
