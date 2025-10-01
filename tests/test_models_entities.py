from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.models import file as file_models
from app.models.entities import (
    JobRecord,
    JobStatus,
    SettingRecord,
    TenantRecord,
    TenantStatus,
    UserRecord,
    UserStatus,
)


@pytest.fixture
def sqlite_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "entities.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DB_URL", db_url)
    file_models.get_engine.cache_clear()
    yield db_url
    file_models.get_engine.cache_clear()


def test_entity_models_support_basic_crud(sqlite_db: str) -> None:
    engine = file_models.get_engine(sqlite_db)
    with Session(engine) as session:
        tenant = TenantRecord(
            tenant_id="tenant-1",
            name="Tenant One",
            status=TenantStatus.ACTIVE,
            storage_quota=1024,
            document_quota=10,
        )
        session.add(tenant)
        session.commit()
        session.refresh(tenant)

        user = UserRecord(
            tenant_id=tenant.tenant_id,
            external_id="user-1",
            email="user@example.com",
            role="admin",
        )
        session.add(user)

        job = JobRecord(
            tenant_id=tenant.tenant_id,
            job_type="ingest",
            status=JobStatus.PROCESSING,
            resource_id="file-1",
            payload={"action": "ingest"},
        )
        session.add(job)

        setting = SettingRecord(
            tenant_id=tenant.tenant_id,
            name="feature",
            value={"enabled": True},
            status="active",
        )
        session.add(setting)
        session.commit()

        session.refresh(user)
        session.refresh(job)
        session.refresh(setting)

        assert user.status == UserStatus.ACTIVE
        assert job.status == JobStatus.PROCESSING
        assert job.payload == {"action": "ingest"}
        assert setting.value == {"enabled": True}

        job.status = JobStatus.COMPLETED
        job.error = None
        job.payload = {"result": "ok"}
        session.add(job)
        session.commit()

    with Session(engine) as session:
        tenant_row = session.exec(select(TenantRecord)).one()
        assert tenant_row.document_quota == 10

        users = session.exec(select(UserRecord)).all()
        assert users and users[0].email == "user@example.com"

        jobs = session.exec(select(JobRecord)).all()
        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.COMPLETED
        assert jobs[0].payload == {"result": "ok"}

        settings = session.exec(select(SettingRecord)).all()
        assert settings and settings[0].status == "active"
