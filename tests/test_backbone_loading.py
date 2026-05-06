"""
test_backbone_loading.py -- Tier 6 tests for Stage 8A real backbone loading.

Verifies:
- backbone configs contain required fields
- expected embedding dimensions in configs
- RETFound config marks as deferred/unavailable
- unknown model_type raises BackboneUnavailableError (no silent mock fallback)
- _freeze_backbone freezes all parameters
- dim mismatch raises BackboneDimensionError
- ResNet-50/ConvNeXt extraction path does not produce 1000-class logits (weights=None)
- backbone cache namespaces are distinct (mock vs real)

Unit tests use mocking or weights=None to avoid repeated large downloads.
Guarded integration tests skip if torchvision is not importable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from retina_screen.embeddings import (
    BackboneConfig,
    BackboneDimensionError,
    BackboneUnavailableError,
    MockBackbone,
    _freeze_backbone,
    _verify_embedding_dim,
    get_cache_dir,
    load_backbone,
)
from retina_screen.core import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BACKBONE_DIR = Path("configs/backbone")

REQUIRED_FIELDS = {"name", "model_type", "embedding_dim"}


def _load_backbone_cfg(name: str) -> dict:
    return load_config(_BACKBONE_DIR / f"{name}.yaml")


# ---------------------------------------------------------------------------
# 1–4. Config field validation (no model loading)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone_name", [
    "resnet50", "convnext_base", "dinov2_base", "dinov2_large",
])
def test_backbone_config_has_required_fields(backbone_name: str) -> None:
    cfg = _load_backbone_cfg(backbone_name)
    missing = REQUIRED_FIELDS - set(cfg.keys())
    assert not missing, (
        f"configs/backbone/{backbone_name}.yaml is missing required fields: {missing}"
    )


# ---------------------------------------------------------------------------
# 5–8. Expected embedding dimensions in configs
# ---------------------------------------------------------------------------


def test_resnet50_config_expects_embedding_dim_2048() -> None:
    cfg = _load_backbone_cfg("resnet50")
    assert int(cfg["embedding_dim"]) == 2048, (
        f"resnet50.yaml embedding_dim={cfg['embedding_dim']}, expected 2048"
    )


def test_convnext_base_config_expects_embedding_dim_1024() -> None:
    cfg = _load_backbone_cfg("convnext_base")
    assert int(cfg["embedding_dim"]) == 1024, (
        f"convnext_base.yaml embedding_dim={cfg['embedding_dim']}, expected 1024"
    )


def test_dinov2_base_config_expects_embedding_dim_768() -> None:
    cfg = _load_backbone_cfg("dinov2_base")
    assert int(cfg["embedding_dim"]) == 768, (
        f"dinov2_base.yaml embedding_dim={cfg['embedding_dim']}, expected 768"
    )


def test_dinov2_large_config_expects_embedding_dim_1024() -> None:
    cfg = _load_backbone_cfg("dinov2_large")
    assert int(cfg["embedding_dim"]) == 1024, (
        f"dinov2_large.yaml embedding_dim={cfg['embedding_dim']}, expected 1024"
    )


# ---------------------------------------------------------------------------
# 9. RETFound config marks as deferred/unavailable
# ---------------------------------------------------------------------------


def test_retfound_config_marks_as_deferred() -> None:
    cfg = _load_backbone_cfg("retfound")
    # Must not be model_type=mock (that would be silent fallback)
    assert cfg.get("model_type") != "mock", (
        "retfound.yaml model_type must not be 'mock' — "
        "RETFound is deferred, not silently replaced by mock."
    )
    # Must indicate deferred/unavailable status
    status = str(cfg.get("status", "") or cfg.get("stage_available", "")).lower()
    assert "deferred" in status or "unavailable" in status, (
        f"retfound.yaml does not clearly mark as deferred/unavailable. "
        f"Got status={cfg.get('status')!r}, stage_available={cfg.get('stage_available')!r}"
    )


# ---------------------------------------------------------------------------
# 10. Unknown model_type raises BackboneUnavailableError (no silent mock fallback)
# ---------------------------------------------------------------------------


def test_unknown_model_type_raises_backbone_unavailable() -> None:
    """Unrecognised model_type must raise BackboneUnavailableError, not return MockBackbone."""
    cfg = BackboneConfig(
        name="hypothetical", embedding_dim=512, model_type="totally_unknown_model_xyz"
    )
    with pytest.raises(BackboneUnavailableError):
        load_backbone(cfg, torch.device("cpu"))


def test_unknown_model_type_does_not_return_mock() -> None:
    """Verify the returned object is not a MockBackbone on unknown model_type."""
    cfg = BackboneConfig(
        name="hypothetical", embedding_dim=512, model_type="not_a_real_backbone"
    )
    try:
        result = load_backbone(cfg, torch.device("cpu"))
        # If it somehow didn't raise, it must not be a MockBackbone
        assert not isinstance(result, MockBackbone), (
            "load_backbone silently returned MockBackbone for an unknown model_type"
        )
    except (BackboneUnavailableError, ValueError, RuntimeError):
        pass  # expected


def test_retfound_model_type_raises_backbone_unavailable() -> None:
    """RETFound model_type must raise BackboneUnavailableError, not silently use mock."""
    cfg = BackboneConfig(name="retfound", embedding_dim=1024, model_type="retfound")
    with pytest.raises(BackboneUnavailableError):
        load_backbone(cfg, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 11. _freeze_backbone freezes all parameters
# ---------------------------------------------------------------------------


def test_freeze_backbone_sets_requires_grad_false() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    # Verify parameters start as trainable
    for param in model.parameters():
        param.requires_grad_(True)
    assert any(p.requires_grad for p in model.parameters()), (
        "Test setup: expected trainable params before freezing"
    )
    _freeze_backbone(model)
    for name, param in model.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name!r} still has requires_grad=True after _freeze_backbone"
        )


def test_freeze_backbone_sets_eval_mode() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
    model.train()
    assert model.training, "Test setup: model should be in train mode"
    _freeze_backbone(model)
    assert not model.training, "_freeze_backbone did not set model to eval mode"


# ---------------------------------------------------------------------------
# 12. _verify_embedding_dim raises BackboneDimensionError on mismatch
# ---------------------------------------------------------------------------


def test_verify_embedding_dim_raises_on_wrong_dim() -> None:
    # Model that outputs 512 but config says 1024
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 512))
    model.eval()
    cfg = BackboneConfig(name="test", embedding_dim=1024, model_type="mock")
    with pytest.raises(BackboneDimensionError):
        _verify_embedding_dim(model, cfg, torch.device("cpu"))


def test_verify_embedding_dim_passes_on_correct_dim() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2048))
    model.eval()
    cfg = BackboneConfig(name="test", embedding_dim=2048, model_type="mock")
    _verify_embedding_dim(model, cfg, torch.device("cpu"))  # must not raise


# ---------------------------------------------------------------------------
# 13. Mock and real backbone cache namespaces are distinct
# ---------------------------------------------------------------------------


def test_mock_and_real_backbone_cache_namespaces_are_separate(tmp_path: Path) -> None:
    prep_hash = "abcd1234abcd1234"
    mock_dir = get_cache_dir(tmp_path, "mock", "odir", prep_hash)
    resnet_dir = get_cache_dir(tmp_path, "resnet50", "odir", prep_hash)
    dinov2_dir = get_cache_dir(tmp_path, "dinov2_large", "odir", prep_hash)

    assert mock_dir != resnet_dir, "mock and resnet50 share the same cache directory"
    assert mock_dir != dinov2_dir, "mock and dinov2_large share the same cache directory"
    assert resnet_dir != dinov2_dir, "resnet50 and dinov2_large share the same cache directory"

    # Verify backbone name appears in path
    assert "mock" in mock_dir.parts
    assert "resnet50" in resnet_dir.parts
    assert "dinov2_large" in dinov2_dir.parts


# ---------------------------------------------------------------------------
# 14–17. Guarded structural tests — weights=None (no download, no pretrained)
# ---------------------------------------------------------------------------

torchvision = pytest.importorskip(
    "torchvision",
    reason="torchvision not installed; skipping structural backbone tests",
)


def test_resnet50_fc_removal_does_not_produce_imagenet_logits() -> None:
    """ResNet-50 with fc=Identity must output 2048-dim, not 1000-dim ImageNet logits."""
    from torchvision.models import resnet50  # noqa: PLC0415
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 2048), (
        f"Expected (1, 2048) after fc removal, got {tuple(out.shape)}. "
        f"Ensure fc is replaced with Identity, not kept as Linear(2048, 1000)."
    )
    assert out.shape[1] != 1000, "ResNet-50 is outputting 1000-class ImageNet logits"


def test_convnext_base_classifier_removal_does_not_produce_imagenet_logits() -> None:
    """ConvNeXt-Base with classifier[-1]=Identity must output 1024-dim, not 1000-dim logits."""
    from torchvision.models import convnext_base  # noqa: PLC0415
    model = convnext_base(weights=None)
    model.classifier[-1] = nn.Identity()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 1024), (
        f"Expected (1, 1024) after classifier[-1] removal, got {tuple(out.shape)}. "
        f"Ensure classifier[-1] (Linear) is replaced with Identity."
    )
    assert out.shape[1] != 1000, "ConvNeXt-Base is outputting 1000-class ImageNet logits"


def test_resnet50_all_frozen_after_freeze(tmp_path: Path) -> None:
    from torchvision.models import resnet50  # noqa: PLC0415
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    _freeze_backbone(model)
    for name, param in model.named_parameters():
        assert not param.requires_grad, (
            f"ResNet-50 parameter {name!r} has requires_grad=True after _freeze_backbone"
        )


def test_convnext_base_all_frozen_after_freeze() -> None:
    from torchvision.models import convnext_base  # noqa: PLC0415
    model = convnext_base(weights=None)
    model.classifier[-1] = nn.Identity()
    _freeze_backbone(model)
    for name, param in model.named_parameters():
        assert not param.requires_grad, (
            f"ConvNeXt-Base parameter {name!r} has requires_grad=True after _freeze_backbone"
        )
