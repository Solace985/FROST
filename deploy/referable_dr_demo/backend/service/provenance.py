from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
DEMO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = DEMO_ROOT / ".local"


def ensure_src_importable() -> None:
    """Insert ``<repo>/src`` onto sys.path so ``import retina_screen`` works.

    Idempotent and harmless if ``retina_screen`` is already installed (editable).
    """
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


_CHUNK = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    """Return the uppercase hex SHA-256 of a file (streamed; constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    """Return the uppercase hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest().upper()


def sha256_text(text: str) -> str:
    """Return the uppercase hex SHA-256 of a UTF-8 string."""
    return sha256_bytes(text.encode("utf-8"))


def task_ordering_hash(task_names: Sequence[str]) -> str:
    """Stable hash of the ordered task list (order-sensitive)."""
    serialized = json.dumps(list(task_names), ensure_ascii=True)
    return sha256_text(serialized)


def python_version() -> str:
    return sys.version.split()[0]


def torch_version() -> str:
    try:
        import torch  # noqa: PLC0415

        return str(torch.__version__)
    except Exception:  # pragma: no cover - torch always present in this env
        return "unknown"


def timm_version() -> str:
    try:
        import timm  # noqa: PLC0415

        return str(timm.__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def git_commit() -> str:
    """Return the current git commit hash, or '' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git always present here
        pass
    return ""


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp (timezone-aware)."""
    return datetime.now(tz=timezone.utc).isoformat()
