from __future__ import annotations

from io import BytesIO

from sqlmodel import select

from backend.app.db.session import session_scope
from backend.app.db.utils import init_db
from backend.app.models.pack import Pack, PackItem
from backend.app.models.template import Template


def _default_template_bytes() -> bytes:
    """Create a simple DOCX document used as the default template."""

    from docx import Document

    document = Document()
    document.add_paragraph("Hello {{ name }}!")
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def seed() -> None:
    """Populate the database with minimal data for local development."""

    init_db()

    with session_scope() as session:
        template = session.get(Template, "welcome")
        if template is None:
            template = Template(id="welcome", name="Welcome Template", content=_default_template_bytes())
            session.add(template)
            session.flush()

        pack = session.exec(select(Pack).where(Pack.name == "Sample Pack")).first()
        if pack is None:
            pack = Pack(name="Sample Pack")
            session.add(pack)
            session.flush()
            session.add(
                PackItem(
                    pack_id=pack.id,
                    template_id=template.id,
                    position=1,
                    document_name="Welcome Document",
                    context={"name": "World"},
                )
            )
            session.flush()

    print("Database seed completed.")


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    seed()
