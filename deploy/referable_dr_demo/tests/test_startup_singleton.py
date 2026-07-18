from __future__ import annotations

import json

import pytest


def test_model_is_loaded_once_per_process(built_bundle):
    """21. The inference service is a process-wide singleton (loaded once)."""
    from deploy.referable_dr_demo.backend.service import inference as inf

    s1 = inf.init_service(built_bundle)
    s2 = inf.init_service(built_bundle)
    assert s1 is s2
    assert s1.load_count == 1


def test_health_does_not_expose_private_paths(client):
    """25. /health returns only safe aggregate status, no private paths."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "status",
        "bundle_version",
        "model_loaded",
        "parity_status",
        "threshold_status",
        "network_access_required",
    }
    blob = json.dumps(body)
    for leak in (
        "C:\\",
        "/Users/",
        "retfoundgreen",
        "model_checkpoint",
        ".pth",
        ".pt",
        "runs/",
        "runs\\",
        "models/",
        "data/brset",
        "cache/",
    ):
        assert leak not in blob, f"/health leaked {leak!r}"
    assert body["network_access_required"] is False


def test_predict_fails_closed_when_parity_or_threshold_invalid(client, png_bytes):
    """26. /predict returns 503 fail-closed when the server is not ready."""
    from deploy.referable_dr_demo.backend import app as app_mod

    original = app_mod.STATE.ready
    try:
        app_mod.STATE.ready = False
        resp = client.post(
            "/predict", files={"image": ("synthetic.png", png_bytes, "image/png")}
        )
        assert resp.status_code == 503
        assert resp.json()["category"] == "server_not_ready"
    finally:
        app_mod.STATE.ready = original
