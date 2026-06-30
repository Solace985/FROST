"""Shared fixtures for FROST app-local tests.

Heavy fixtures (bundle, service, FastAPI client) try to assemble the real
artifacts and ``pytest.skip`` when they are unavailable, so the committed suite
passes in any environment and runs fully where the private checkpoints exist.

No BRSET images are embedded in committed tests; synthetic images are generated
in-process.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Make local artifacts discoverable via the canonical defaults if present.
_default_brset = REPO_ROOT / "data" / "brset"
if "RETINA_SCREEN_BRSET_ROOT" not in os.environ and _default_brset.exists():
    os.environ["RETINA_SCREEN_BRSET_ROOT"] = str(_default_brset)
_default_backbone = REPO_ROOT / "models" / "retfoundgreen_statedict.pth"
if "RETFOUND_GREEN_CHECKPOINT" not in os.environ and _default_backbone.exists():
    os.environ["RETFOUND_GREEN_CHECKPOINT"] = str(_default_backbone)

from deploy.referable_dr_demo.backend.service import bundle as bundle_mod  # noqa: E402


def make_png_bytes(w: int = 420, h: int = 380, fmt: str = "PNG") -> bytes:
    """Deterministic synthetic RGB image bytes (not a BRSET image)."""
    from PIL import Image  # noqa: PLC0415

    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    g = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    b = (((xx // 12) + (yy // 12)) % 2).astype(np.uint8) * 90 + 40
    arr = np.stack([r, g, b.astype(np.uint8)], axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    return make_png_bytes()


@pytest.fixture(scope="session")
def built_bundle():
    """Build + persist + validate the deployment bundle, or skip."""
    try:
        manifest = bundle_mod.build_bundle(validate_backbone_forward=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native-392 artifacts unavailable: {exc}")
    out = bundle_mod.default_bundle_path()
    bundle_mod.write_bundle(manifest, out)
    return bundle_mod.validate_bundle_against_artifacts(manifest)


@pytest.fixture(scope="session")
def service(built_bundle):
    from deploy.referable_dr_demo.backend.service import inference as inf  # noqa: PLC0415

    try:
        return inf.init_service(built_bundle)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"inference service unavailable: {exc}")


@pytest.fixture(scope="session")
def client(built_bundle):
    """FastAPI TestClient with full startup gates, or skip."""
    pytest.importorskip("fastapi")
    from deploy.referable_dr_demo.analysis import derive_threshold  # noqa: PLC0415
    from deploy.referable_dr_demo.backend.service import threshold_policy  # noqa: PLC0415

    if not threshold_policy.default_threshold_path().exists():
        try:
            derive_threshold.derive(write=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"operating point unavailable: {exc}")

    from fastapi.testclient import TestClient  # noqa: PLC0415

    from deploy.referable_dr_demo.backend import app as app_mod  # noqa: PLC0415

    with TestClient(app_mod.app) as c:
        yield c
