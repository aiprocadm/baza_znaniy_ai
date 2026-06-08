"""Persist + verify the embedder signature against the existing index.

Mismatch is a hard, loud failure (never silent, never auto-reindex). The storage
hooks (load/save) are injected so this stays pure and testable; the MVP store wires
them to its SQLite meta table.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

# Single source-of-truth for the kv_meta key used to store the embedder
# signature.  Both the load and save paths in the caller must reference this
# constant so the key lives in exactly one place.
SIGNATURE_KEY = "sig"


class _HasSignature(Protocol):
    name: str
    dimension: int


class EmbedderMismatchError(RuntimeError):
    """Raised when the active embedder disagrees with the indexed vectors."""


def signature_for(embedder: _HasSignature) -> str:
    return f"{embedder.name}:{int(embedder.dimension)}"


def verify_or_store(
    embedder: _HasSignature,
    *,
    load: Callable[[str], Optional[str]],
    save: Callable[[str], None],
) -> None:
    """Store the signature on a fresh index; raise on mismatch otherwise."""
    current = signature_for(embedder)
    stored = load(SIGNATURE_KEY)
    if stored is None:
        save(current)
        return
    if stored != current:
        raise EmbedderMismatchError(
            f"Embedder/index mismatch: index was built with {stored!r} but the active "
            f"embedder is {current!r}. The vectors are not comparable. Either run "
            f"`kb-cli reindex --embedder {embedder.name}` to rebuild the index, or set "
            f"KB_EMBEDDINGS_BACKEND to the original backend to keep the existing index."
        )


__all__ = [
    "SIGNATURE_KEY",
    "EmbedderMismatchError",
    "signature_for",
    "verify_or_store",
]
