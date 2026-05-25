"""
test_retfound_green_native392_setup.py -- Setup verification for RETFound-Green native-392 comparator.

Verifies Stage 8D-3.5 B2 native-392 setup is correct before BRSET extraction runs:

 1. retfound_green_native392.yaml declares embedding_dim=384
 2. Native backbone config declares input_size=392 (not 224)
 3. Native backbone config declares global_pool='avg' (not 'token')
 4. Native backbone config uses original checkpoint route (NOT interpolated _224)
 5. Native backbone config documents external-validation exclusions (ODIR, AIROGS, DDR)
 6. Native backbone config is a protocol comparator, not a main matrix replacement
 7. Native preprocessing config has image_size=392 (distinct from default_224)
 8. Native preprocessing hash differs from default_224 preprocessing hash
 9. Native experiment configs are BRSET-only (no ODIR/AIROGS/DDR as active dataset)
10. Native experiment configs do not override class_weighting_enabled to true
11. Native experiment configs have final_test_result=false
12. MT and LP native configs share the same cache namespace
13. Matched-224 config remains unchanged (input_size=224, native_392_deferred=true)
14. loader instantiates native-392 model with img_size=(392,392) and global_pool='avg' (mocked)
15. BackboneConfig input_size and global_pool fields default to 224 and 'token' for matched-224

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
    get_cache_dir,
    load_backbone,
)
from retina_screen.preprocessing import PreprocessingConfig, get_preprocessing_hash

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_BACKBONE_DIR = Path("configs/backbone")
_EXPERIMENT_DIR = Path("configs/experiment")
_PREP_DIR = Path("configs/preprocessing")

_NATIVE_BACKBONE_CONFIG = "retfound_green_native392.yaml"
_NATIVE_MT_CONFIG = "stage8d35_b2_brset_retfound_green_native392_multitask.yaml"
_NATIVE_LP_CONFIG = "stage8d35_b2_brset_retfound_green_native392_linearprobe.yaml"
_MATCHED224_BACKBONE_CONFIG = "retfound_green_matched224.yaml"

_FORBIDDEN_DATASETS = {
    "odir", "odir-5k", "odir-2019", "odir_5k", "odir_2019",
    "airogs", "ddr", "mbrset", "aptos", "idrid", "messidor",
    "rfmid", "eyepacs",
}


def _load_backbone_cfg(name: str) -> dict:
    return load_config(_BACKBONE_DIR / f"{name}.yaml")


def _load_exp_cfg(filename: str) -> dict:
    return load_config(_EXPERIMENT_DIR / filename)


def _load_prep_cfg(name: str) -> dict:
    return load_config(_PREP_DIR / f"{name}.yaml")


def _make_prep_config(raw: dict) -> PreprocessingConfig:
    return PreprocessingConfig(
        image_size=int(raw.get("image_size", 224)),
        mean=tuple(raw.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(raw.get("std", [0.229, 0.224, 0.225])),
        use_clahe=bool(raw.get("use_clahe", False)),
        use_graham=bool(raw.get("use_graham", False)),
        interpolation=str(raw.get("interpolation", "bilinear")),
        random_hflip_p=float(raw.get("random_hflip_p", 0.0)),
        random_rotation_deg=float(raw.get("random_rotation_deg", 0.0)),
        color_jitter=bool(raw.get("color_jitter", False)),
    )


# ---------------------------------------------------------------------------
# 1. Native backbone config declares embedding_dim=384
# ---------------------------------------------------------------------------


def test_native392_config_declares_384_dim() -> None:
    cfg = _load_backbone_cfg("retfound_green_native392")
    assert int(cfg["embedding_dim"]) == 384, (
        f"retfound_green_native392.yaml embedding_dim={cfg['embedding_dim']!r}, expected 384"
    )


# ---------------------------------------------------------------------------
# 2. Native backbone config declares input_size=392
# ---------------------------------------------------------------------------


def test_native392_config_declares_input_size_392() -> None:
    cfg = _load_backbone_cfg("retfound_green_native392")
    assert int(cfg.get("input_size", 0)) == 392, (
        f"retfound_green_native392.yaml input_size={cfg.get('input_size')!r}, expected 392"
    )


# ---------------------------------------------------------------------------
# 3. Native backbone config declares global_pool='avg'
# ---------------------------------------------------------------------------


def test_native392_config_declares_avg_pooling() -> None:
    cfg = _load_backbone_cfg("retfound_green_native392")
    gpool = cfg.get("global_pool", "")
    assert gpool == "avg", (
        f"retfound_green_native392.yaml global_pool={gpool!r}, expected 'avg' (native protocol)"
    )


# ---------------------------------------------------------------------------
# 4. Native backbone config uses original checkpoint route (NOT interpolated _224)
# ---------------------------------------------------------------------------


def test_native392_config_references_original_checkpoint_not_224() -> None:
    cfg = _load_backbone_cfg("retfound_green_native392")
    checkpoint_path = str(cfg.get("checkpoint_path", "") or "")
    # If a hardcoded path is set, it must not point to the interpolated _224 checkpoint
    assert "statedict_224" not in checkpoint_path.lower(), (
        f"retfound_green_native392.yaml checkpoint_path must not reference "
        f"retfoundgreen_statedict_224.pth (the PE-interpolated matched-224 checkpoint); "
        f"got {checkpoint_path!r}"
    )
    # Native config sha256 field should match original checkpoint, not 224 checkpoint
    sha_field = str(cfg.get("checkpoint_sha256_original", "") or "")
    if sha_field:
        assert sha_field.upper() == "431DE5DBC1BEBBB32F60E2C0BCF8DAA4F8BCBF06F7CB1E1DC97EC589713942E1", (
            f"checkpoint_sha256_original does not match expected original checkpoint SHA256; "
            f"got {sha_field!r}"
        )


# ---------------------------------------------------------------------------
# 5. Native backbone config documents external-validation exclusions
# ---------------------------------------------------------------------------


def test_native392_config_excludes_odir_and_external_datasets() -> None:
    cfg = _load_backbone_cfg("retfound_green_native392")
    exclusions = cfg.get("external_validation_exclusions", [])
    assert exclusions, (
        "retfound_green_native392.yaml must declare external_validation_exclusions"
    )
    exclusions_lower = [str(e).lower() for e in exclusions]
    for required_token in ("odir", "airogs", "ddr"):
        assert any(required_token in e for e in exclusions_lower), (
            f"external_validation_exclusions must include {required_token!r}; "
            f"got {exclusions}"
        )


# ---------------------------------------------------------------------------
# 6. Native config is a comparator, not a main matrix replacement
# ---------------------------------------------------------------------------


def test_native392_config_is_comparator_not_main_matrix() -> None:
    cfg = _load_backbone_cfg("retfound_green_native392")
    # Must not claim to replace the matched-224 main matrix row
    assert cfg.get("main_matrix_replacement", True) is False, (
        "retfound_green_native392.yaml must declare main_matrix_replacement=false"
    )
    assert cfg.get("native_comparator", False) is True, (
        "retfound_green_native392.yaml must declare native_comparator=true"
    )


# ---------------------------------------------------------------------------
# 7. Native preprocessing config has image_size=392
# ---------------------------------------------------------------------------


def test_native392_preprocessing_config_has_392_image_size() -> None:
    cfg = _load_prep_cfg("retfound_green_native392")
    assert int(cfg.get("image_size", 0)) == 392, (
        f"retfound_green_native392.yaml preprocessing image_size={cfg.get('image_size')!r}, "
        "expected 392"
    )


# ---------------------------------------------------------------------------
# 8. Native preprocessing hash differs from default_224
# ---------------------------------------------------------------------------


def test_native392_preprocessing_hash_differs_from_default224() -> None:
    native_raw = _load_prep_cfg("retfound_green_native392")
    default_raw = _load_prep_cfg("default_224")
    native_cfg = _make_prep_config(native_raw)
    default_cfg = _make_prep_config(default_raw)
    native_hash = get_preprocessing_hash(native_cfg)
    default_hash = get_preprocessing_hash(default_cfg)
    assert native_hash != default_hash, (
        f"retfound_green_native392 preprocessing hash must differ from default_224 "
        f"to produce a distinct cache namespace; "
        f"native={native_hash!r}, default224={default_hash!r}"
    )
    # Also confirm 92d0f40b94aea26c is NOT the native hash (that's the matched-224 hash)
    assert native_hash != "92d0f40b94aea26c", (
        f"Native-392 preprocessing hash must not be 92d0f40b94aea26c "
        f"(that is the matched-224 RETFound-Green cache hash)"
    )


# ---------------------------------------------------------------------------
# 9. Native experiment configs are BRSET-only
# ---------------------------------------------------------------------------


def test_native392_experiment_configs_are_brset_only() -> None:
    for filename in [_NATIVE_MT_CONFIG, _NATIVE_LP_CONFIG]:
        cfg = _load_exp_cfg(filename)
        assert cfg.get("dataset") == "brset", (
            f"{filename}: expected dataset='brset', got {cfg.get('dataset')!r}"
        )
        cfg_str = " ".join(str(v).lower() for v in cfg.values())
        for forbidden in _FORBIDDEN_DATASETS:
            assert forbidden not in cfg_str, (
                f"{filename}: forbidden dataset token {forbidden!r} found in config values"
            )


# ---------------------------------------------------------------------------
# 10. Native experiment configs do not override class_weighting_enabled
# ---------------------------------------------------------------------------


def test_native392_configs_class_weighting_false() -> None:
    for filename in [_NATIVE_MT_CONFIG, _NATIVE_LP_CONFIG]:
        cfg = _load_exp_cfg(filename)
        value = cfg.get("class_weighting_enabled")
        assert value is None or value is False or value == "false", (
            f"{filename}: class_weighting_enabled must be absent or false (Decision 027); "
            f"got {value!r}"
        )


# ---------------------------------------------------------------------------
# 11. Native experiment configs have final_test_result=false
# ---------------------------------------------------------------------------


def test_native392_configs_final_test_result_false() -> None:
    for filename in [_NATIVE_MT_CONFIG, _NATIVE_LP_CONFIG]:
        cfg = _load_exp_cfg(filename)
        value = cfg.get("final_test_result", False)
        assert value is False or value == "false", (
            f"{filename}: final_test_result must be false; got {value!r}"
        )


# ---------------------------------------------------------------------------
# 12. MT and LP native configs share the same cache namespace
# ---------------------------------------------------------------------------


def test_native392_mt_lp_configs_share_cache_namespace(tmp_path: Path) -> None:
    mt_cfg = _load_exp_cfg(_NATIVE_MT_CONFIG)
    lp_cfg = _load_exp_cfg(_NATIVE_LP_CONFIG)

    assert mt_cfg.get("backbone") == lp_cfg.get("backbone"), (
        "Native-392 MT and LP configs must use the same backbone (for shared cache)"
    )
    assert mt_cfg.get("preprocessing") == lp_cfg.get("preprocessing"), (
        "Native-392 MT and LP configs must use the same preprocessing"
    )
    assert mt_cfg.get("dataset") == lp_cfg.get("dataset"), (
        "Native-392 MT and LP configs must bind to the same dataset"
    )
    # Confirm they resolve to the same cache dir (different from matched-224)
    native_raw = _load_prep_cfg("retfound_green_native392")
    native_prep = _make_prep_config(native_raw)
    native_hash = get_preprocessing_hash(native_prep)
    backbone_name = mt_cfg["backbone"]
    mt_dir = get_cache_dir(tmp_path, backbone_name, "brset", native_hash)
    lp_dir = get_cache_dir(tmp_path, backbone_name, "brset", native_hash)
    assert mt_dir == lp_dir, (
        f"MT and LP native configs resolve to different cache dirs: {mt_dir} vs {lp_dir}"
    )
    # Confirm native cache dir is separate from matched-224 cache dir
    matched224_dir = get_cache_dir(tmp_path, "retfound_green", "brset", "92d0f40b94aea26c")
    assert mt_dir != matched224_dir, (
        "Native-392 and matched-224 RETFound-Green must not share a cache directory"
    )


# ---------------------------------------------------------------------------
# 13. Matched-224 config remains unchanged
# ---------------------------------------------------------------------------


def test_matched224_config_unchanged() -> None:
    cfg = _load_backbone_cfg("retfound_green_matched224")
    assert int(cfg.get("input_size", 0)) == 224, (
        f"retfound_green_matched224.yaml input_size={cfg.get('input_size')!r} changed; expected 224"
    )
    assert cfg.get("native_392_deferred") is True, (
        f"retfound_green_matched224.yaml native_392_deferred changed; must remain true"
    )
    # global_pool not set in matched-224 config → BackboneConfig defaults to 'token'
    assert cfg.get("global_pool") is None or cfg.get("global_pool") == "token", (
        f"retfound_green_matched224.yaml global_pool changed unexpectedly: "
        f"{cfg.get('global_pool')!r}"
    )


# ---------------------------------------------------------------------------
# 14. loader instantiates native-392 model with img_size=(392,392) and global_pool='avg'
# ---------------------------------------------------------------------------


def test_native392_loader_resolves_correct_timm_args(tmp_path: Path) -> None:
    """Verify load_backbone calls timm.create_model with native-392 args when config sets them.

    For global_pool='avg' (native-392), the loader:
      1. Calls torch.load() to get the state dict.
      2. Remaps norm.weight/bias -> fc_norm.weight/bias (timm avg-pool key rename).
      3. Calls timm.create_model WITHOUT checkpoint_path.
      4. Calls model.load_state_dict() with the remapped state dict.
    """
    fake_ckpt = tmp_path / "retfoundgreen_statedict.pth"
    fake_ckpt.write_bytes(b"fake")  # path must exist; torch.load is mocked below

    cfg = BackboneConfig(
        name="retfound_green_native392",
        embedding_dim=384,
        model_type="retfound_green",
        version="retfound_green_v0.1_native392",
        checkpoint_path=str(fake_ckpt),
        input_size=392,
        global_pool="avg",
    )

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.parameters.return_value = iter([])
    mock_model.named_parameters.return_value = iter([])
    mock_model.return_value = torch.zeros(1, 384)
    mock_model.to.return_value = mock_model
    mock_model.training = False
    mock_model.load_state_dict.return_value = ([], [])  # success (no missing/unexpected keys)

    # Fake state dict with norm.weight/bias — these must be remapped to fc_norm.*
    fake_state = {
        "norm.weight": torch.zeros(384),
        "norm.bias": torch.zeros(384),
        "pos_embed": torch.zeros(1, 784, 384),
    }

    with (
        patch("timm.create_model", return_value=mock_model) as mock_create,
        patch("timm.list_models", return_value=["vit_small_patch14_reg4_dinov2"]),
        patch("retina_screen.embeddings._freeze_backbone"),
        patch("retina_screen.embeddings._verify_embedding_dim"),
        patch("torch.load", return_value=fake_state),  # mock state-dict load for avg-pool path
    ):
        result = load_backbone(cfg, torch.device("cpu"))

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    # Must use 392×392 input, not 224×224
    assert call_kwargs[1].get("img_size") == (392, 392), (
        f"timm.create_model must be called with img_size=(392,392) for native protocol; "
        f"got {call_kwargs}"
    )
    # Must use avg pooling, not CLS token
    assert call_kwargs[1].get("global_pool") == "avg", (
        f"timm.create_model must be called with global_pool='avg' for native protocol; "
        f"got {call_kwargs}"
    )
    # Must still use num_classes=0
    assert call_kwargs[1].get("num_classes") == 0, (
        f"timm.create_model must be called with num_classes=0; got {call_kwargs}"
    )
    # For avg-pool path: checkpoint_path must NOT be passed to timm.create_model
    # (state dict is loaded manually and injected via load_state_dict)
    assert "checkpoint_path" not in call_kwargs[1], (
        f"For avg-pool native protocol, timm.create_model must NOT receive checkpoint_path; "
        f"checkpoint is loaded manually and injected via load_state_dict. got {call_kwargs}"
    )
    # Verify load_state_dict was called with fc_norm remapping
    mock_model.load_state_dict.assert_called_once()
    state_passed = mock_model.load_state_dict.call_args[0][0]
    assert "fc_norm.weight" in state_passed, (
        "load_state_dict must receive fc_norm.weight (remapped from norm.weight for avg-pool)"
    )
    assert "fc_norm.bias" in state_passed, (
        "load_state_dict must receive fc_norm.bias (remapped from norm.bias for avg-pool)"
    )
    assert "norm.weight" not in state_passed, (
        "norm.weight must be remapped to fc_norm.weight before load_state_dict"
    )
    assert "norm.bias" not in state_passed, (
        "norm.bias must be remapped to fc_norm.bias before load_state_dict"
    )


# ---------------------------------------------------------------------------
# 15. BackboneConfig defaults to 224 / 'token' for matched-224 compatibility
# ---------------------------------------------------------------------------


def test_backbone_config_defaults_preserve_matched224_behavior() -> None:
    """BackboneConfig with no input_size/global_pool must default to 224 / 'token'."""
    cfg = BackboneConfig(
        name="retfound_green",
        embedding_dim=384,
        model_type="retfound_green",
        version="retfound_green_v0.1",
        checkpoint_path="",
    )
    assert cfg.input_size == 224, (
        f"BackboneConfig default input_size must be 224; got {cfg.input_size!r}"
    )
    assert cfg.global_pool == "token", (
        f"BackboneConfig default global_pool must be 'token' (CLS); got {cfg.global_pool!r}"
    )


# ---------------------------------------------------------------------------
# Architecture smoke (timm required)
# ---------------------------------------------------------------------------

timm = pytest.importorskip(
    "timm",
    reason="timm not installed; skipping native-392 architecture smoke tests",
)


def test_native392_architecture_produces_384_dim_at_392() -> None:
    """Architecture-only smoke: vit_small_patch14_reg4_dinov2 at 392×392, avg-pool → (1,384)."""
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(392, 392),
        num_classes=0,
        global_pool="avg",
        pretrained=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 392, 392))
    assert tuple(out.shape) == (1, 384), (
        f"Expected native-392 output shape (1, 384), got {tuple(out.shape)}"
    )


def test_native392_avg_pool_differs_from_cls_token() -> None:
    """With the same random weights, avg-pool output must differ from CLS-token output."""
    torch.manual_seed(99)
    model_cls = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(392, 392),
        num_classes=0,
        global_pool="token",
        pretrained=False,
    )
    model_avg = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(392, 392),
        num_classes=0,
        global_pool="avg",
        pretrained=False,
    )
    model_cls.eval()
    model_avg.eval()
    x = torch.randn(1, 3, 392, 392)
    with torch.no_grad():
        out_cls = model_cls(x)
        out_avg = model_avg(x)
    assert not torch.allclose(out_cls, out_avg), (
        "avg-pool output must differ from CLS-token output for same input"
    )
