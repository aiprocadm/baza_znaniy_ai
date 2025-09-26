import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://kb_ai:kb_ai@localhost:5432/kb_ai",
)

class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


def _create_engine():
    connect_args: dict[str, object] = {}
    return create_engine(DATABASE_URL, future=True, pool_pre_ping=True, connect_args=connect_args)


def _create_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)

engine = _create_engine()
SessionLocal = _create_session_factory(engine)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables."""
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
