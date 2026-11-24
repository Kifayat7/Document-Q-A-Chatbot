"""Small, dependency-free helpers shared across modules."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def user_key(user_id: str) -> str:
    """Return a stable, filesystem/collection-safe key for a user id.

    The raw id (which may be an e-mail address) is never written to disk.
    """
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes (used for duplicate detection)."""
    return hashlib.sha256(data).hexdigest()


def safe_display_name(name: str | None, default: str = "document.pdf") -> str:
    """Strip any path components and control characters from an uploaded name."""
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = _CONTROL_CHARS.sub("", base).strip()
    return base[:150] or default


def human_size(num_bytes: int) -> str:
    """Format a byte count, e.g. 1536 -> '1.5 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
