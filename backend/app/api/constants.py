TRACE_HEADER = "X-Trace-Id"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_UPLOAD_MIME_TYPES = {
    "application/msword",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/csv",
    "text/markdown",
    "text/plain",
}

__all__ = ["TRACE_HEADER", "MAX_UPLOAD_SIZE_BYTES", "ALLOWED_UPLOAD_MIME_TYPES"]
