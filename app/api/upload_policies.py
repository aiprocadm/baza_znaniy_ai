"""Shared upload content-type policies used across API versions."""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_CONTENT_TYPES_BY_EXTENSION: dict[str, set[str]] = {
    "pdf": {"application/pdf"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    "pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    },
    "xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "txt": {"text/plain"},
    "md": {"text/markdown", "text/plain"},
}
GENERIC_ALLOWED_CONTENT_TYPES = {"application/octet-stream"}


@dataclass(slots=True)
class UploadContentTypeEvaluation:
    """Outcome of validating an upload's declared content-type."""

    content_type: str
    requires_logging: bool = False


class UploadContentTypeError(ValueError):
    """Raised when an upload declares an unsupported content-type."""

    def __init__(self, detail: str = "UPLOAD_INVALID_TYPE") -> None:
        super().__init__(detail)
        self.detail = detail


def normalise_content_type(content_type: str | None) -> str:
    """Return a lower-cased MIME type without parameters."""

    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def evaluate_content_type(extension: str, content_type: str | None) -> UploadContentTypeEvaluation:
    """Validate ``content_type`` for *extension*.

    The returned :class:`UploadContentTypeEvaluation` contains the normalised
    content-type and whether the caller should log an informational message.
    ``UploadContentTypeError`` is raised when the declared type is not allowed.
    """

    normalised = normalise_content_type(content_type)
    allowed = ALLOWED_CONTENT_TYPES_BY_EXTENSION.get(extension)
    if allowed:
        if normalised not in allowed:
            raise UploadContentTypeError()
        return UploadContentTypeEvaluation(content_type=normalised)
    if not normalised:
        raise UploadContentTypeError()
    requires_logging = normalised not in GENERIC_ALLOWED_CONTENT_TYPES
    return UploadContentTypeEvaluation(
        content_type=normalised,
        requires_logging=requires_logging,
    )

