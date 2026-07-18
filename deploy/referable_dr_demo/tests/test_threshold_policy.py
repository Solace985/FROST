from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.referable_dr_demo.backend.service import preprocessing_parity
from deploy.referable_dr_demo.backend.service import threshold_policy


class _StubBundle:
    """Minimal stand-in exposing the attributes threshold_policy reads."""

    backbone_sha256 = "A" * 64
    head_sha256 = "B" * 64
    preprocessing_hash = preprocessing_parity.preprocessing_hash()
    task_ordering_hash = "C" * 64
    manifest = {
        "model_family": "retfound_green",
        "native_protocol": "native_392_avg_pool",
    }


def _valid_op(bundle: _StubBundle, **overrides) -> dict:
    op = {
        "threshold": 0.0444,
        "derivation_split": "validation",
        "target_sensitivity": 0.95,
        "validation_sensitivity": 0.95,
        "validation_specificity": 0.96,
        "validation_positive_count": 100,
        "validation_negative_count": 2343,
        "heldout_test_sensitivity": 0.97,
        "heldout_test_specificity": 0.95,
        "test_positive_count": 73,
        "test_negative_count": 1550,
        "backbone_checkpoint_sha256": bundle.backbone_sha256,
        "head_checkpoint_sha256": bundle.head_sha256,
        "preprocessing_hash": bundle.preprocessing_hash,
        "task_ordering_hash": bundle.task_ordering_hash,
        "model_family": bundle.manifest["model_family"],
        "native_protocol": bundle.manifest["native_protocol"],
        "threshold_selection_algorithm": "largest T with val sensitivity >= 0.95",
    }
    op.update(overrides)
    return op


def _write(tmp_path: Path, op: dict) -> Path:
    p = tmp_path / "operating_point.local.json"
    p.write_text(json.dumps(op), encoding="utf-8")
    return p


def test_threshold_selection_uses_validation_only(tmp_path):
    """12. A validation-derived, provenance-matched artifact loads and decides."""
    bundle = _StubBundle()
    p = _write(tmp_path, _valid_op(bundle))
    op = threshold_policy.load_operating_point(bundle, path=p)
    assert op.derivation_split == "validation"
    assert op.decide(0.9) == threshold_policy.REFERABLE
    assert op.decide(0.0) == threshold_policy.NOT_REFERABLE

    real = threshold_policy.default_threshold_path()
    if real.exists():
        m = json.loads(real.read_text(encoding="utf-8"))
        assert m["derivation_split"] == "validation"


def test_threshold_manifest_invalidates_on_bundle_change(tmp_path):
    """13. Any drift in a bound hash invalidates the operating point."""
    bundle = _StubBundle()
    for field in (
        "backbone_checkpoint_sha256",
        "head_checkpoint_sha256",
        "preprocessing_hash",
        "task_ordering_hash",
        "model_family",
        "native_protocol",
    ):
        p = _write(tmp_path, _valid_op(bundle, **{field: "CHANGED"}))
        with pytest.raises(threshold_policy.ThresholdInvalidError):
            threshold_policy.load_operating_point(bundle, path=p)


def test_no_test_paths_allowed_in_threshold_selection(tmp_path):
    """14. A threshold derived from the test split is refused."""
    bundle = _StubBundle()
    p = _write(tmp_path, _valid_op(bundle, derivation_split="test"))
    with pytest.raises(threshold_policy.ThresholdInvalidError):
        threshold_policy.load_operating_point(bundle, path=p)


def test_no_reliability_split_used_for_threshold_selection(tmp_path):
    """15. A threshold derived from the reliability split is refused."""
    bundle = _StubBundle()
    p = _write(tmp_path, _valid_op(bundle, derivation_split="reliability"))
    with pytest.raises(threshold_policy.ThresholdInvalidError):
        threshold_policy.load_operating_point(bundle, path=p)
