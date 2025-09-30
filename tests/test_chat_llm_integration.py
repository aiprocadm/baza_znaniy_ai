from __future__ import annotations

from tests.stubs.fastapi import Request

from app.api.routes import chat
from app.core.app import create_app
from app.models.chat import ChatIn


class StubProvider:
    handles_citations = False

    def __init__(self) -> None:
        self.prompts: list[tuple[str, dict[str, object] | None]] = []

    def ensure_model(self) -> None:
        return None

    def generate(self, prompt: str, *, context: dict[str, object] | None = None) -> str:
        self.prompts.append((prompt, context))
        return "Ответ"


def test_chat_endpoint_uses_fallback_index(tmp_path):
    provider = StubProvider()
    app = create_app(provider)

    app.state.settings.data_dir = tmp_path
    app.state.chat_store = type(app.state.chat_store)(str(tmp_path / "chat.sqlite3"))
    app.state.fallback_index = [
        {"file": "doc.txt", "page": 1, "text": "пример", "score": 0.42},
    ]
    app.state.vector_store = None

    request = Request()
    request.app = app  # type: ignore[attr-defined]
    payload = chat(
        request,
        ChatIn(user_id="alice", conversation_id=None, message="Привет"),
    )

    assert payload["answer"].startswith("Ответ")
    assert payload["citations"]
    assert provider.prompts, "LLM provider should have been called"
