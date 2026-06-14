"""MVP knowledge-base endpoints mounted under ``/api/kb``.

Auth-free contract for the simple frontend in ``data/www/index.html``.
The full multi-tenant API stays under ``/api/v1/*``. ``/ask`` prefers
:func:`app.services.kb_llm.select_provider`, falls back to
``state.llm_provider`` (legacy), then to an extractive answer.

This package is the split of the former single-file ``kb_mvp.py``. The
public import surface (``router`` plus the helpers/models/prompt that
tests and ``app/eval`` import) is re-exported here so
``from app.api.kb_mvp import X`` keeps working unchanged.
"""

from __future__ import annotations

from .common import router, public, protected

# Importing the endpoint modules registers their routes on the shared
# ``public`` / ``protected`` routers via decorator side-effects.
from . import health, documents, search, chat  # noqa: F401,E402

# Wire sub-routers into the top-level router (same order/paths as before).
router.include_router(public)
router.include_router(protected)

# W4 — live feedback collection endpoints
from app.api.kb_feedback import router as kb_feedback_router  # noqa: E402

router.include_router(kb_feedback_router)

# ---- Re-export the public import surface (back-compat with the old module) ----
from .common import (  # noqa: E402,F401
    LOGGER,
    MAX_UPLOAD_BYTES,
    SUPPORTED_UPLOAD_EXT,
    _conversation_to_out,
    _decode_text,
    _doc_to_out,
    _extension_for,
    _format_history,
    _hit_to_out,
    _message_to_out,
    _parse_file_bytes,
    _parse_file_bytes_with_pages,
    _resolve_data_dir,
    _resolve_kb_files_dir,
    _sources_payload_to_hit_out,
    _store_for,
)
from .rag import (  # noqa: E402,F401
    _RAG_SYSTEM_PROMPT,
    _build_rag_prompt,
    _extractive_answer,
    _format_context,
    _generate_answer,
    _retrieve_with_rerank,
)
from .health import health, providers  # noqa: E402,F401,F811
from .documents import (  # noqa: E402,F401
    create_document,
    delete_document,
    get_document,
    get_document_file,
    list_documents,
    upload_document,
)
from .search import search_documents  # noqa: E402,F401
from .chat import (  # noqa: E402,F401
    ask,
    ask_stream,
    create_conversation,
    delete_conversation,
    get_conversation_detail,
    list_conversations,
    rename_conversation,
    _sse_event,
    _stream_extractive,
    _stream_legacy,
)
from .schemas import (  # noqa: E402,F401
    AskRequest,
    AskResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationRename,
    DocumentCreate,
    DocumentListItem,
    DocumentOut,
    HitOut,
    MessageOut,
    RerankInfo,
    RetrievalReasonOut,
    RetrievalReportOut,
    SearchRequest,
    SearchResponse,
)

__all__ = ["router"]
