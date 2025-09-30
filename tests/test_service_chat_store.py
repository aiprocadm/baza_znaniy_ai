from importlib import util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "srv" / "projects" / "kb" / "app" / "store.py"
SPEC = util.spec_from_file_location("store", MODULE_PATH)
assert SPEC and SPEC.loader
store_module = util.module_from_spec(SPEC)
SPEC.loader.exec_module(store_module)
ChatStore = store_module.ChatStore
ConversationAccessError = store_module.ConversationAccessError


@pytest.fixture()
def store(tmp_path: Path) -> ChatStore:
    db_path = tmp_path / "chat" / "store.db"
    store = ChatStore(str(db_path), secret="top-secret")
    # ensure database file created in nested directory
    assert db_path.exists()
    return store


def test_ensure_conversation_access_control(store: ChatStore) -> None:
    user_id = "user-1"
    conversation_id = store.ensure_conversation(user_id, None)

    assert isinstance(conversation_id, str)
    assert conversation_id

    # should reuse existing conversation for same user
    reused_id = store.ensure_conversation(user_id, conversation_id)
    assert reused_id == conversation_id

    # ensure _secret was initialised safely
    assert store._secret == b"top-secret"

    # different user attempting to access the conversation should fail
    with pytest.raises(ConversationAccessError):
        store.ensure_conversation("user-2", conversation_id)


def test_message_workflow(store: ChatStore) -> None:
    user_id = "user-1"
    conversation_id = store.ensure_conversation(user_id, None)

    # No summary should be present initially
    assert store.get_summary(conversation_id) is None

    # No messages since summary at the beginning
    assert store.messages_since_summary(conversation_id) == 0

    # Recording exchanges should persist both user and assistant messages
    store.record_exchange(conversation_id, "hello", "hi there")
    store.record_exchange(conversation_id, "how are you?", "doing well")

    messages = store.get_recent_messages(conversation_id)
    assert messages == [
        ("user", "hello"),
        ("assistant", "hi there"),
        ("user", "how are you?"),
        ("assistant", "doing well"),
    ]

    assert store.messages_since_summary(conversation_id) == 4

    # Saving a summary should reset the counter and be retrievable
    store.save_summary(conversation_id, "summary text")
    assert store.get_summary(conversation_id) == "summary text"
    assert store.messages_since_summary(conversation_id) == 0

    # After summary, new messages should be tracked
    store.record_exchange(conversation_id, "new", "response")
    assert store.messages_since_summary(conversation_id) == 2

    # messages_since_summary should return 0 for unknown conversations
    assert store.messages_since_summary("missing") == 0

    recent_limited = store.get_recent_messages(conversation_id, limit=2)
    assert recent_limited == [
        ("user", "new"),
        ("assistant", "response"),
    ]

    # messages_since_summary should remain unaffected by read operations
    assert store.messages_since_summary(conversation_id) == 2

    # save a new summary to reset again
    store.save_summary(conversation_id, "updated summary")
    assert store.get_summary(conversation_id) == "updated summary"
    assert store.messages_since_summary(conversation_id) == 0


