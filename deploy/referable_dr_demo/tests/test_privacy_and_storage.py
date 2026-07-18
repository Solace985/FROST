from __future__ import annotations

from pathlib import Path

import pytest

from deploy.referable_dr_demo.backend.service import privacy

DEMO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NETWORK_TOKENS = (
    "import requests",
    "requests.get",
    "requests.post",
    "import urllib",
    "urllib.request",
    "import http.client",
    "import socket",
    "import aiohttp",
    "import httpx",
    "from httpx",
)


def _non_test_sources() -> list[Path]:
    files: list[Path] = []
    for p in DEMO_ROOT.rglob("*.py"):
        parts = p.relative_to(DEMO_ROOT).parts
        if parts and parts[0] == "tests":
            continue
        if "__pycache__" in parts:
            continue
        files.append(p)
    return files


def test_no_upload_persistence(client, png_bytes):
    """18. Uploads are processed in memory; nothing is persisted to disk."""
    with pytest.raises(ValueError):
        privacy._assert_permitted({"filename": "x"})

    resp = client.post(
        "/predict", files={"image": ("synthetic.png", png_bytes, "image/png")}
    )
    assert resp.status_code in (200, 503)

    assert not (DEMO_ROOT / "uploads").exists()
    rasters = [
        p
        for p in DEMO_ROOT.rglob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and "frontend" not in p.relative_to(DEMO_ROOT).parts
    ]
    assert rasters == [], f"unexpected persisted image files: {rasters}"


def test_no_external_network_dependency():
    """20. No demonstrator source performs outbound network access."""
    with pytest.raises(ValueError):
        privacy._assert_permitted({"score": 0.5})

    hits: list[tuple[str, str]] = []
    for f in _non_test_sources():
        text = f.read_text(encoding="utf-8")
        for tok in FORBIDDEN_NETWORK_TOKENS:
            if tok in text:
                hits.append((f.name, tok))
    assert not hits, f"forbidden network usage found: {hits}"
