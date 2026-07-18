from __future__ import annotations

import os
import pathlib

import pytest
import torch
import yaml

from retina_screen.embeddings import _compact_embedding

REPO = pathlib.Path(__file__).resolve().parent.parent
BACKBONE_CFG = REPO / "configs/backbone/dinov3_large.yaml"
MT_CFG = REPO / "configs/experiment/stage8d35_b3_brset_dinov3_large_multitask.yaml"
LP_CFG = REPO / "configs/experiment/stage8d35_b3_brset_dinov3_large_linearprobe.yaml"




class TestDINOv3BackboneConfig:
    def setup_method(self):
        self.cfg = yaml.safe_load(BACKBONE_CFG.read_text())

    def test_embedding_dim(self):
        assert self.cfg["embedding_dim"] == 1024

    def test_model_type(self):
        assert self.cfg["model_type"] == "dinov3"

    def test_input_size(self):
        assert self.cfg["input_size"] == 224

    def test_patch_size(self):
        assert self.cfg["patch_size"] == 16

    def test_num_register_tokens(self):
        assert self.cfg["num_register_tokens"] == 4

    def test_expected_tokens_at_224(self):
        assert self.cfg["expected_tokens_at_224"] == 201

    def test_global_pool_token(self):
        assert self.cfg["global_pool"] == "token"

    def test_frozen(self):
        assert self.cfg["frozen"] is True

    def test_no_hardcoded_path(self):
        assert self.cfg.get("checkpoint_path", "") == ""

    def test_has_env_var_field(self):
        assert "checkpoint_env_var" in self.cfg

    def test_preprocessing_is_default_224(self):
        assert self.cfg.get("preprocessing", "default_224") == "default_224"




class TestDINOv3MTConfig:
    def setup_method(self):
        self.cfg = yaml.safe_load(MT_CFG.read_text())

    def test_dataset_is_brset(self):
        assert self.cfg["dataset"] == "brset"

    def test_backbone_is_dinov3_large(self):
        assert self.cfg["backbone"] == "dinov3_large"

    def test_head_type_multitask(self):
        assert self.cfg["head_type"] == "multitask"

    def test_seed_42(self):
        assert self.cfg["seed"] == 42

    def test_embedding_dim_1024(self):
        assert self.cfg["embedding_dim"] == 1024

    def test_final_test_result_false(self):
        assert self.cfg["final_test_result"] is False

    def test_fast_dev_run_false(self):
        assert self.cfg["fast_dev_run"] is False

    def test_full_dataset_run_true(self):
        assert self.cfg["full_dataset_run"] is True

    def test_preprocessing_default_224(self):
        assert self.cfg["preprocessing"] == "default_224"


class TestDINOv3LPConfig:
    def setup_method(self):
        self.cfg = yaml.safe_load(LP_CFG.read_text())

    def test_head_type_linear_probe(self):
        assert self.cfg["head_type"] == "linear_probe"

    def test_same_backbone(self):
        assert self.cfg["backbone"] == "dinov3_large"

    def test_final_test_result_false(self):
        assert self.cfg["final_test_result"] is False

    def test_same_preprocessing(self):
        assert self.cfg["preprocessing"] == "default_224"


class TestDINOv3MTLPCacheNamespaceConsistency:
    def test_same_backbone_and_preprocessing(self):
        mt = yaml.safe_load(MT_CFG.read_text())
        lp = yaml.safe_load(LP_CFG.read_text())
        assert mt["backbone"] == lp["backbone"] == "dinov3_large"
        assert mt["preprocessing"] == lp["preprocessing"] == "default_224"




class TestDINOv3LoaderFailsLoudly:
    """Verifies loader raises BackboneUnavailableError without fallback.
    Requires no checkpoint file — tests the empty-path error path only."""

    def test_missing_checkpoint_raises(self, monkeypatch):
        from retina_screen.embeddings import (
            BackboneConfig,
            BackboneUnavailableError,
            load_backbone,
        )

        monkeypatch.delenv("RETINA_SCREEN_DINOV3_VITL16_WEIGHTS", raising=False)
        cfg = BackboneConfig(
            name="dinov3_large",
            embedding_dim=1024,
            model_type="dinov3",
            version="lvd1689m",
            checkpoint_path="",
            input_size=224,
            global_pool="token",
        )
        with pytest.raises(BackboneUnavailableError):
            load_backbone(cfg, torch.device("cpu"))

    def test_nonexistent_path_raises(self, tmp_path):
        from retina_screen.embeddings import (
            BackboneConfig,
            BackboneUnavailableError,
            load_backbone,
        )

        nonexistent = str(tmp_path / "nonexistent.pth")
        cfg = BackboneConfig(
            name="dinov3_large",
            embedding_dim=1024,
            model_type="dinov3",
            version="lvd1689m",
            checkpoint_path=nonexistent,
            input_size=224,
            global_pool="token",
        )
        with pytest.raises(BackboneUnavailableError):
            load_backbone(cfg, torch.device("cpu"))




class TestDINOv3W1Compaction:
    """Verify _compact_embedding handles DINOv3's 201-token / 1024-dim output.

    DINOv3 with CLS-token pooling returns the CLS embedding as a view into
    the full 201×1024 token-sequence backing storage.  _compact_embedding must
    reduce storage to (1024,) with W1 ratio = 1.0.
    """

    _DIM = 1024
    _SEQ_LEN = 201

    def _make_cls_view(self) -> torch.Tensor:
        tokens = torch.randn(1, self._SEQ_LEN, self._DIM)
        return tokens[:, 0].squeeze(0)

    def _storage_ratio(self, t: torch.Tensor) -> float:
        return t.untyped_storage().nbytes() / (t.numel() * t.element_size())

    def test_view_storage_bloated_before_compaction(self):
        cls_view = self._make_cls_view()
        ratio = self._storage_ratio(cls_view)
        assert ratio > 10.0, f"Expected ratio>10 (simulates bug); got {ratio:.2f}"

    def test_compact_embedding_reduces_to_flat(self):
        cls_view = self._make_cls_view()
        compact = _compact_embedding(cls_view)
        assert compact.shape == (self._DIM,)
        assert compact.dtype == torch.float32

    def test_compact_embedding_w1_ratio_is_1(self):
        cls_view = self._make_cls_view()
        compact = _compact_embedding(cls_view)
        ratio = self._storage_ratio(compact)
        assert ratio <= 2.0, f"Expected W1 ratio<=2.0; got {ratio:.2f}"

    def test_compact_preserves_values(self):
        cls_view = self._make_cls_view()
        compact = _compact_embedding(cls_view)
        assert torch.allclose(cls_view, compact)

    def test_compact_storage_size_bytes(self):
        cls_view = self._make_cls_view()
        compact = _compact_embedding(cls_view)
        storage_bytes = compact.untyped_storage().nbytes()
        expected_bytes = self._DIM * 4
        assert storage_bytes <= expected_bytes * 2
