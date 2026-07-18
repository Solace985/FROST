from __future__ import annotations

from pathlib import Path

import pytest

from deploy.referable_dr_demo.backend.service import bundle as bundle_mod
from deploy.referable_dr_demo.backend.service import preprocessing_parity


def test_bundle_requires_exact_native392_artifacts(built_bundle):
    """1. The validated bundle describes the exact native-392 MT deployment cell."""
    m = built_bundle.manifest
    assert m["backbone_identifier"] == "retfound_green_native392"
    assert m["model_family"] == "retfound_green"
    assert m["native_protocol"] == "native_392_avg_pool"
    assert m["head_type"] == "MultiTaskHead"
    assert m["native_input_size"] == 392
    assert m["native_pooling"] == "average"
    assert m["expected_embedding_dim"] == 384
    assert m["dr_grade_task_key"] == "dr_grade"
    assert m["dr_grade_task_index"] == 0
    assert m["model_task_ordering"][0] == "dr_grade"


def test_bundle_fails_without_checkpoint(monkeypatch, tmp_path):
    """2. Discovery fails closed when no backbone checkpoint can be resolved."""
    monkeypatch.delenv(bundle_mod.ENV_BACKBONE_CKPT, raising=False)
    monkeypatch.delenv(bundle_mod.ENV_CANONICAL_BACKBONE_CKPT, raising=False)
    monkeypatch.delenv(bundle_mod.ENV_HEAD_CKPT, raising=False)
    monkeypatch.setattr(bundle_mod, "_DEFAULT_BACKBONE_CKPT", tmp_path / "absent_backbone.pth")
    monkeypatch.setattr(bundle_mod, "_DEFAULT_HEAD_CKPT", tmp_path / "absent_head.pt")
    with pytest.raises(bundle_mod.BundleValidationError):
        bundle_mod.discover_artifacts()


def test_bundle_fails_on_hash_mismatch(tmp_path):
    """3. A manifest whose recorded SHA-256 no longer matches the file is refused."""
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"backbone-bytes")
    head = tmp_path / "head.pt"
    head.write_bytes(b"head-bytes")
    manifest = {
        "expected_embedding_dim": 384,
        "native_input_size": 392,
        "native_pooling": "average",
        "head_type": "MultiTaskHead",
        "model_family": "retfound_green",
        "native_protocol": "native_392_avg_pool",
        "dr_grade_output_shape": 5,
        "preprocessing_hash": preprocessing_parity.preprocessing_hash(),
        "backbone_checkpoint_path": str(backbone),
        "head_checkpoint_path": str(head),
        "backbone_checkpoint_sha256": "DEADBEEF" * 8,
        "head_checkpoint_sha256": "FEEDFACE" * 8,
        "model_task_ordering": ["dr_grade"],
        "task_ordering_hash": "x",
    }
    with pytest.raises(bundle_mod.BundleValidationError):
        bundle_mod.validate_bundle_against_artifacts(manifest)


def test_native_input_size_is_392(built_bundle):
    """4. Native input size is exactly 392 (constant + live manifest)."""
    assert bundle_mod.NATIVE_INPUT_SIZE == 392
    assert preprocessing_parity.EXPECTED_IMAGE_SIZE == 392
    assert built_bundle.manifest["native_input_size"] == 392


def test_native_pooling_is_average(built_bundle):
    """5. Native pooling is average pooling (timm global_pool='avg')."""
    assert bundle_mod.NATIVE_POOLING == "average"
    assert bundle_mod.BACKBONE_GLOBAL_POOL == "avg"
    assert built_bundle.manifest["native_pooling"] == "average"
    assert built_bundle.backbone_config_kwargs()["global_pool"] == "avg"


def test_expected_embedding_dim_is_384(built_bundle):
    """6. Expected embedding dimension is exactly 384."""
    assert bundle_mod.EXPECTED_EMBEDDING_DIM == 384
    assert built_bundle.embedding_dim == 384
    assert built_bundle.manifest["expected_embedding_dim"] == 384
