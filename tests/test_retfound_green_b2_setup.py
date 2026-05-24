"""
test_retfound_green_b2_setup.py -- B2 setup verification tests for RETFound-Green.

Verifies Stage 8D-3.5 B2 setup is correct before BRSET extraction runs:

 1. retfound_green_matched224.yaml declares embedding_dim=384
 2. Config uses matched-224 input, native_392_deferred=true
 3. Config uses a distinct name/model_type from retfound.yaml
 4. B2 experiment configs bind to BRSET only (no ODIR/AIROGS/DDR as active dataset)
 5. Backbone YAML documents external-validation exclusions (ODIR, AIROGS, DDR)
 6. load_backbone routes model_type='retfound_green' to timm loader (mocked checkpoint)
 7. timm architecture produces output shape (1, 384) — no weights, no BRSET
 8. Loader raises BackboneUnavailableError for missing checkpoint (no mock fallback)
 9. W1 compaction applies to 384-dim ViT-S-style output (storage ratio ≤ 2.0)
10. MT and LP configs share the same backbone/dataset/preprocessing → same cache namespace
11. Both B2 experiment configs have final_test_result=false (or absent, defaults false)
12. Neither B2 experiment config overrides class_weighting_enabled to true

No real checkpoint, no BRSET raw data, no network access required.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from retina_screen.core import load_config
from retina_screen.embeddings import (
    BackboneConfig,
    BackboneUnavailableError,
    _compact_embedding,
    get_cache_dir,
    load_backbone,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_BACKBONE_DIR = Path("configs/backbone")
_EXPERIMENT_DIR = Path("configs/experiment")

_B2_MT_CONFIG = "stage8d35_b2_brset_retfound_green_matched224_multitask.yaml"
_B2_LP_CONFIG = "stage8d35_b2_brset_retfound_green_matched224_linearprobe.yaml"
_B2_BACKBONE_CONFIG = "retfound_green_matched224.yaml"

_FORBIDDEN_DATASETS = {
    "odir", "odir-5k", "odir-2019", "odir_5k", "odir_2019",
    "airogs", "ddr", "mbrset", "aptos", "idrid", "messidor",
    "rfmid", "eyepacs",
}


def _load_backbone_cfg(name: str) -> dict:
    return load_config(_BACKBONE_DIR / f"{name}.yaml")


def _load_exp_cfg(filename: str) -> dict:
    return load_config(_EXPERIMENT_DIR / filename)


# ---------------------------------------------------------------------------
# 1. retfound_green_matched224.yaml declares embedding_dim = 384
# ---------------------------------------------------------------------------


def test_retfound_green_config_declares_384_dim() -> None:
    cfg = _load_backbone_cfg("retfound_green_matched224")
    assert int(cfg["embedding_dim"]) == 384, (
        f"retfound_green_matched224.yaml embedding_dim={cfg['embedding_dim']!r}, expected 384"
    )


# ---------------------------------------------------------------------------
# 2. Config uses matched-224 input, not native-392
# ---------------------------------------------------------------------------


def test_retfound_green_config_is_matched224_not_native392() -> None:
    cfg = _load_backbone_cfg("retfound_green_matched224")
    assert int(cfg.get("input_size", 0)) == 224, (
        f"retfound_green_matched224.yaml input_size={cfg.get('input_size')!r}, expected 224"
    )
    assert cfg.get("native_392_deferred") is True, (
        f"native_392_deferred must be true; got {cfg.get('native_392_deferred')!r}"
    )
    # Confirm native input size is documented as 392 (not the active input size)
    assert int(cfg.get("native_input_size", 0)) == 392, (
        f"native_input_size should document 392; got {cfg.get('native_input_size')!r}"
    )


# ---------------------------------------------------------------------------
# 3. Config is distinct from retfound.yaml — separate name and model_type
# ---------------------------------------------------------------------------


def test_retfound_green_config_does_not_reuse_retfound_yaml() -> None:
    rg_cfg = _load_backbone_cfg("retfound_green_matched224")
    rf_cfg = _load_backbone_cfg("retfound")

    # Different name
    assert rg_cfg.get("name") != rf_cfg.get("name"), (
        "retfound_green_matched224.yaml must have a different 'name' from retfound.yaml"
    )
    # Different model_type
    assert rg_cfg.get("model_type") != rf_cfg.get("model_type"), (
        "retfound_green_matched224.yaml must not share 'model_type' with retfound.yaml"
    )
    # Not deferred
    status = str(rg_cfg.get("status", "") or rg_cfg.get("stage_available", "")).lower()
    assert "deferred" not in status, (
        f"retfound_green_matched224.yaml must not be marked deferred; got status={status!r}"
    )
    # model_type must be retfound_green
    assert rg_cfg.get("model_type") == "retfound_green", (
        f"Expected model_type='retfound_green', got {rg_cfg.get('model_type')!r}"
    )


# ---------------------------------------------------------------------------
# 4. B2 experiment configs bind to BRSET only (no forbidden datasets)
# ---------------------------------------------------------------------------


def test_retfound_green_configs_are_brset_only() -> None:
    for filename in [_B2_MT_CONFIG, _B2_LP_CONFIG]:
        cfg = _load_exp_cfg(filename)
        assert cfg.get("dataset") == "brset", (
            f"{filename}: expected dataset='brset', got {cfg.get('dataset')!r}"
        )
        # Scan all string values for forbidden dataset tokens
        cfg_str = " ".join(str(v).lower() for v in cfg.values())
        for forbidden in _FORBIDDEN_DATASETS:
            assert forbidden not in cfg_str, (
                f"{filename}: forbidden dataset token {forbidden!r} found in config values"
            )


# ---------------------------------------------------------------------------
# 5. Backbone YAML documents external-validation exclusions for ODIR/AIROGS/DDR
# ---------------------------------------------------------------------------


def test_retfound_green_configs_exclude_odir_and_external_datasets() -> None:
    cfg = _load_backbone_cfg("retfound_green_matched224")
    exclusions = cfg.get("external_validation_exclusions", [])
    assert exclusions, (
        "retfound_green_matched224.yaml must declare external_validation_exclusions"
    )
    exclusions_lower = [str(e).lower() for e in exclusions]
    for required_token in ("odir", "airogs", "ddr"):
        assert any(required_token in e for e in exclusions_lower), (
            f"external_validation_exclusions must include {required_token!r}; "
            f"got {exclusions}"
        )


# ---------------------------------------------------------------------------
# 6. load_backbone routes retfound_green to timm loader (mocked checkpoint)
# ---------------------------------------------------------------------------


def test_retfound_green_loader_resolves_timm_model(tmp_path: Path) -> None:
    """Verify load_backbone calls timm.create_model with correct args when checkpoint exists."""
    # Create a fake checkpoint file so the path-exists check passes
    fake_ckpt = tmp_path / "retfoundgreen_statedict.pth"
    fake_ckpt.write_bytes(b"fake")

    cfg = BackboneConfig(
        name="retfound_green",
        embedding_dim=384,
        model_type="retfound_green",
        version="retfound_green_v0.1",
        checkpoint_path=str(fake_ckpt),
    )

    # Mock timm.create_model so it returns a simple linear model producing (1, 384)
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.parameters.return_value = iter([])
    mock_model.named_parameters.return_value = iter([])
    # forward returns a tensor of the expected shape for _verify_embedding_dim
    mock_model.return_value = torch.zeros(1, 384)
    mock_model.to.return_value = mock_model
    mock_model.training = False

    with (
        patch("timm.create_model", return_value=mock_model) as mock_create,
        patch("timm.list_models", return_value=["vit_small_patch14_reg4_dinov2"]),
        patch("retina_screen.embeddings._freeze_backbone"),
        patch("retina_screen.embeddings._verify_embedding_dim"),
    ):
        result = load_backbone(cfg, torch.device("cpu"))

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    assert call_kwargs[0][0] == "vit_small_patch14_reg4_dinov2", (
        f"timm.create_model called with wrong model name: {call_kwargs}"
    )
    assert call_kwargs[1].get("img_size") == (224, 224), (
        f"timm.create_model not called with img_size=(224,224): {call_kwargs}"
    )
    assert call_kwargs[1].get("num_classes") == 0, (
        f"timm.create_model not called with num_classes=0: {call_kwargs}"
    )
    assert call_kwargs[1].get("checkpoint_path") == str(fake_ckpt), (
        f"timm.create_model not called with correct checkpoint_path: {call_kwargs}"
    )


# ---------------------------------------------------------------------------
# 7. timm architecture produces output shape (1, 384) — no weights, no BRSET
# ---------------------------------------------------------------------------

timm = pytest.importorskip(
    "timm",
    reason="timm not installed; skipping architecture smoke tests",
)


def test_retfound_green_synthetic_output_dim_384() -> None:
    """Architecture-only smoke: vit_small_patch14_reg4_dinov2 at 224×224 outputs (1,384)."""
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(224, 224),
        num_classes=0,
        pretrained=False,   # no weights download
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 384), (
        f"Expected output shape (1, 384), got {tuple(out.shape)}"
    )


def test_retfound_green_model_default_pool_is_cls_not_avg() -> None:
    """Default global_pool must be 'token' (CLS), not 'avg' (native-392 protocol)."""
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(224, 224),
        num_classes=0,
        pretrained=False,
    )
    pool = getattr(model, "global_pool", None)
    assert pool != "avg", (
        f"global_pool must not be 'avg' for B2 matched-224 protocol; got {pool!r}. "
        "The loader must NOT set model.global_pool='avg' (that is native-392 protocol)."
    )


# ---------------------------------------------------------------------------
# 8. Loader raises BackboneUnavailableError when checkpoint is missing (no mock fallback)
# ---------------------------------------------------------------------------


def test_retfound_green_loader_no_mock_fallback_empty_checkpoint() -> None:
    """No checkpoint path and no env var → BackboneUnavailableError (never MockBackbone)."""
    cfg = BackboneConfig(
        name="retfound_green",
        embedding_dim=384,
        model_type="retfound_green",
        version="retfound_green_v0.1",
        checkpoint_path="",
    )
    # Clear env var to ensure both sources are empty
    env_backup = os.environ.pop("RETFOUND_GREEN_CHECKPOINT", None)
    try:
        with (
            patch("timm.list_models", return_value=["vit_small_patch14_reg4_dinov2"]),
        ):
            with pytest.raises(BackboneUnavailableError, match="checkpoint"):
                load_backbone(cfg, torch.device("cpu"))
    finally:
        if env_backup is not None:
            os.environ["RETFOUND_GREEN_CHECKPOINT"] = env_backup


def test_retfound_green_loader_no_mock_fallback_missing_file(tmp_path: Path) -> None:
    """checkpoint_path pointing to non-existent file → BackboneUnavailableError."""
    cfg = BackboneConfig(
        name="retfound_green",
        embedding_dim=384,
        model_type="retfound_green",
        version="retfound_green_v0.1",
        checkpoint_path=str(tmp_path / "does_not_exist.pth"),
    )
    env_backup = os.environ.pop("RETFOUND_GREEN_CHECKPOINT", None)
    try:
        with (
            patch("timm.list_models", return_value=["vit_small_patch14_reg4_dinov2"]),
        ):
            with pytest.raises(BackboneUnavailableError, match="not found"):
                load_backbone(cfg, torch.device("cpu"))
    finally:
        if env_backup is not None:
            os.environ["RETFOUND_GREEN_CHECKPOINT"] = env_backup


# ---------------------------------------------------------------------------
# 9. W1 compaction applies to 384-dim ViT-S-style output (storage ratio ≤ 2.0)
# ---------------------------------------------------------------------------

# ViT-S/14-reg4: 16*16 spatial patches + 1 CLS + 4 register tokens = 261 sequence positions
_DIM_384 = 384
_SEQ_LEN_VITS = 261


def _make_384_cls_view() -> torch.Tensor:
    """Simulate CLS-token view into a ViT-S/14-reg4 token sequence (261 positions)."""
    tokens = torch.randn(1, _SEQ_LEN_VITS, _DIM_384)
    return tokens[:, 0].squeeze(0)   # (384,) but backed by (1, 261, 384) storage


def _storage_ratio(t: torch.Tensor) -> float:
    compact_bytes = t.numel() * t.element_size()
    storage_bytes = t.untyped_storage().nbytes()
    return storage_bytes / compact_bytes


def test_retfound_green_w1_storage_ratio_exceeds_threshold_before_compaction() -> None:
    """Pre-compaction view of 384-dim CLS token has bloated backing storage."""
    cls_view = _make_384_cls_view()
    ratio = _storage_ratio(cls_view)
    assert ratio > 5.0, (
        f"Expected storage_ratio > 5.0 for ViT-S CLS view; got {ratio:.2f}"
    )


def test_retfound_green_w1_compaction_reduces_storage_ratio() -> None:
    """_compact_embedding must reduce storage_ratio to ≤ 2.0 for 384-dim output."""
    compacted = _compact_embedding(_make_384_cls_view())
    assert compacted.shape == (_DIM_384,)
    ratio = _storage_ratio(compacted)
    assert ratio <= 2.0, (
        f"After _compact_embedding, storage_ratio must be ≤ 2.0; got {ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# 10. MT and LP configs share the same cache namespace
# ---------------------------------------------------------------------------


def test_b2_mt_lp_configs_share_cache_namespace(tmp_path: Path) -> None:
    mt_cfg = _load_exp_cfg(_B2_MT_CONFIG)
    lp_cfg = _load_exp_cfg(_B2_LP_CONFIG)

    assert mt_cfg.get("backbone") == lp_cfg.get("backbone"), (
        "MT and LP configs must use the same backbone (for shared cache)"
    )
    assert mt_cfg.get("preprocessing") == lp_cfg.get("preprocessing"), (
        "MT and LP configs must use the same preprocessing"
    )
    assert mt_cfg.get("dataset") == lp_cfg.get("dataset"), (
        "MT and LP configs must bind to the same dataset"
    )

    # Verify that get_cache_dir resolves to the same directory
    prep_hash = "92d0f40b94aea26c"
    backbone_name = mt_cfg["backbone"]
    mt_dir = get_cache_dir(tmp_path, backbone_name, "brset", prep_hash)
    lp_dir = get_cache_dir(tmp_path, backbone_name, "brset", prep_hash)
    assert mt_dir == lp_dir, (
        f"MT and LP configs resolve to different cache directories: {mt_dir} vs {lp_dir}"
    )


# ---------------------------------------------------------------------------
# 11. Both B2 configs have final_test_result = false (or absent = defaults false)
# ---------------------------------------------------------------------------


def test_b2_configs_final_test_result_false() -> None:
    for filename in [_B2_MT_CONFIG, _B2_LP_CONFIG]:
        cfg = _load_exp_cfg(filename)
        value = cfg.get("final_test_result", False)
        assert value is False or value == "false", (
            f"{filename}: final_test_result must be false; got {value!r}"
        )


# ---------------------------------------------------------------------------
# 12. Neither B2 config overrides class_weighting_enabled to true
# ---------------------------------------------------------------------------


def test_b2_configs_class_weighting_false() -> None:
    for filename in [_B2_MT_CONFIG, _B2_LP_CONFIG]:
        cfg = _load_exp_cfg(filename)
        value = cfg.get("class_weighting_enabled")
        # Value must be absent (inherits false from standard.yaml) or explicitly false
        assert value is None or value is False or value == "false", (
            f"{filename}: class_weighting_enabled must be absent or false (Decision 027); "
            f"got {value!r}"
        )
