from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy.referable_dr_demo.hf_space import weights_source  # noqa: E402

weights_source.resolve()

from deploy.referable_dr_demo.backend.app import app  # noqa: E402,F401

__all__ = ["app"]
