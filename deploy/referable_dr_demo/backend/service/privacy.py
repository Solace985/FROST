"""privacy.py -- memory-only privacy helpers and safe logging for FROST.

Default behaviour is memory-only: uploads are never persisted, filenames are
never logged, and no patient identifiers are accepted or recorded. This module
centralises the *permitted* local log fields and provides a safe logging helper
that structurally cannot include image bytes, filenames, sample IDs, patient
IDs, raw embeddings, or image-linked scores.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("frost.request")

# The ONLY fields permitted in a local request log line.
PERMITTED_LOG_FIELDS = frozenset({
    "event",
    "timestamp",
    "app_version",
    "bundle_version",
    "success",
    "total_ms",
    "error_category",
    "parity_status",
})

# Fields that must NEVER appear in a log record (defensive allow-list enforcement).
FORBIDDEN_LOG_SUBSTRINGS = (
    "filename",
    "file_name",
    "sample_id",
    "patient",
    "image_bytes",
    "embedding",
    "score",
)


def safe_request_log(
    *,
    success: bool,
    app_version: str,
    bundle_version: str,
    total_ms: float | None = None,
    error_category: str | None = None,
    parity_status: str | None = None,
) -> dict[str, Any]:
    """Build and emit a privacy-safe request log record.

    Only permitted aggregate fields are logged. The score, any identifier, the
    filename, and the raw bytes are never passed in and never logged.
    """
    record: dict[str, Any] = {
        "event": "predict",
        "app_version": app_version,
        "bundle_version": bundle_version,
        "success": success,
    }
    if total_ms is not None:
        record["total_ms"] = round(float(total_ms), 3)
    if error_category is not None:
        record["error_category"] = error_category
    if parity_status is not None:
        record["parity_status"] = parity_status

    _assert_permitted(record)
    logger.info("request %s", record)
    return record


def _assert_permitted(record: dict[str, Any]) -> None:
    """Fail loudly if a record key is outside the permitted allow-list."""
    extra = set(record) - PERMITTED_LOG_FIELDS
    if extra:
        raise ValueError(
            f"privacy violation: attempted to log non-permitted fields {sorted(extra)}"
        )
    for key in record:
        low = key.lower()
        for bad in FORBIDDEN_LOG_SUBSTRINGS:
            if bad in low:
                raise ValueError(
                    f"privacy violation: log field {key!r} matches forbidden token {bad!r}"
                )
