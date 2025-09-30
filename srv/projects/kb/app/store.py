"""Compatibility wrapper exposing the chat store for legacy tests."""

from app.chat.store import ChatStore, ConversationAccessError

__all__ = ["ChatStore", "ConversationAccessError"]
