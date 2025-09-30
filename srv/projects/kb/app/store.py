        codex/create-sqlmodel-models-for-files-and-pages
"""Compatibility wrapper exposing the chat store for legacy tests."""

from app.chat.store import ChatStore, ConversationAccessError

__all__ = ["ChatStore", "ConversationAccessError"]

"""Compatibility wrapper around :mod:`app.chat.store`."""

from app.chat.store import *  # noqa: F401,F403
        main
