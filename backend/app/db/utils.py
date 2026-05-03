from __future__ import annotations

from backend.app.db.base import Base
from backend.app.db.session import get_engine
import backend.app.models  # noqa: F401


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())
