from __future__ import annotations

import pathlib
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO / "configs" / "experiment"

BACKBONES = ["dinov2_large", "dinov3_large"]
VARIANTS = ["singletask_linearprobe_pertask", "singletask_mlp_pertask"]
TASKS = [
    "dr_grade", "macular_edema", "hypertensive_retinopathy",
    "amd", "drusen", "other_ocular", "diabetes",
]


def _f1_configs():
    """Yield (fname, cfg_dict) for all 28 F1 experiment configs."""
    for backbone in BACKBONES:
        for variant in VARIANTS:
            for task in TASKS:
                fname = f"stage8d35_f1_brset_{backbone}_{variant}_{task}.yaml"
                p = CONFIGS_DIR / fname
                yield fname, yaml.safe_load(p.read_text(encoding="utf-8"))


class TestF1ConfigsLoad:
    """All 28 configs must exist and parse without error."""

    @pytest.mark.parametrize("backbone,variant,task", [
        (b, v, t) for b in BACKBONES for v in VARIANTS for t in TASKS
    ])
    def test_config_exists_and_loads(self, backbone, variant, task):
        fname = f"stage8d35_f1_brset_{backbone}_{variant}_{task}.yaml"
        p = CONFIGS_DIR / fname
        assert p.exists(), f"Config not found: {p}"
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(cfg, dict), f"Config is not a dict: {fname}"


class TestF1ConfigFields:
    """Each config must have correct field values matching its filename."""

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_selected_task_present(self, fname, cfg):
        assert "selected_task" in cfg, f"selected_task missing in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_selected_task_is_single_string(self, fname, cfg):
        st = cfg.get("selected_task")
        assert isinstance(st, str), f"selected_task must be a string in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_selected_task_matches_filename(self, fname, cfg):
        st = cfg.get("selected_task", "")
        assert fname.endswith(f"_{st}.yaml"), (
            f"selected_task={st!r} does not match filename task in {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_backbone_matches_filename(self, fname, cfg):
        bb = cfg.get("backbone", "")
        assert f"brset_{bb}_" in fname, (
            f"backbone={bb!r} does not appear in filename {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_head_type_matches_filename_variant(self, fname, cfg):
        ht = cfg.get("head_type", "")
        assert f"_{ht}_" in fname, (
            f"head_type={ht!r} does not appear as variant in filename {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_head_type_is_valid_f1_variant(self, fname, cfg):
        ht = cfg.get("head_type", "")
        assert ht in VARIANTS, (
            f"head_type={ht!r} is not a valid F1 variant in {fname}. "
            f"Expected one of: {VARIANTS}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_dataset_is_brset(self, fname, cfg):
        assert cfg.get("dataset") == "brset", f"dataset != 'brset' in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_embedding_dim_1024(self, fname, cfg):
        assert cfg.get("embedding_dim") == 1024, f"embedding_dim != 1024 in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_final_test_result_false(self, fname, cfg):
        assert cfg.get("final_test_result") is False, (
            f"final_test_result must be false in {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_fast_dev_run_false(self, fname, cfg):
        assert cfg.get("fast_dev_run") is False, f"fast_dev_run must be false in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_full_dataset_run_true(self, fname, cfg):
        assert cfg.get("full_dataset_run") is True, (
            f"full_dataset_run must be true in {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_class_weighting_disabled(self, fname, cfg):
        assert cfg.get("class_weighting_enabled") is False, (
            f"class_weighting_enabled must be false in {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_seed_42(self, fname, cfg):
        assert cfg.get("seed") == 42, f"seed must be 42 in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_preprocessing_default_224(self, fname, cfg):
        assert cfg.get("preprocessing") == "default_224", (
            f"preprocessing must be default_224 in {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_task_config_is_brset_default(self, fname, cfg):
        assert cfg.get("task_config") == "configs/tasks/brset_default.yaml", (
            f"task_config must be brset_default.yaml in {fname}"
        )

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_stage_is_f1(self, fname, cfg):
        assert cfg.get("stage") == "8D-3.5-F1", f"stage must be '8D-3.5-F1' in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_no_focal_loss_field(self, fname, cfg):
        for banned in ("focal_loss", "focal_alpha", "focal_gamma"):
            assert banned not in cfg, f"{banned!r} must not appear in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_no_coral_corn_field(self, fname, cfg):
        for banned in ("coral", "corn", "ordinal_regression"):
            assert banned not in cfg, f"{banned!r} must not appear in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_no_gradnorm_field(self, fname, cfg):
        assert "gradnorm" not in cfg, f"gradnorm must not appear in {fname}"

    @pytest.mark.parametrize("fname,cfg", list(_f1_configs()))
    def test_selected_task_is_known_brset_task(self, fname, cfg):
        st = cfg.get("selected_task", "")
        assert st in TASKS, (
            f"selected_task={st!r} is not a known BRSET task in {fname}. "
            f"Known tasks: {TASKS}"
        )


class TestF1ConfigTotals:
    """Aggregate checks: correct number of configs, no duplicates."""

    def test_total_count_is_28(self):
        configs = list(_f1_configs())
        assert len(configs) == 28, f"Expected 28 F1 configs, found {len(configs)}"

    def test_each_backbone_variant_has_7_tasks(self):
        for backbone in BACKBONES:
            for variant in VARIANTS:
                configs = [
                    (fn, c) for fn, c in _f1_configs()
                    if f"_{backbone}_{variant}_" in fn
                ]
                tasks_found = [c.get("selected_task") for _, c in configs]
                assert sorted(tasks_found) == sorted(TASKS), (
                    f"backbone={backbone}, variant={variant}: "
                    f"expected tasks {sorted(TASKS)}, got {sorted(tasks_found)}"
                )

    def test_no_duplicate_filenames(self):
        fnames = [fn for fn, _ in _f1_configs()]
        assert len(fnames) == len(set(fnames)), "Duplicate F1 config filenames detected"


class TestBuildHeadF1Routing:
    """build_head() must correctly route F1 head types without new model classes."""

    def test_singletask_linearprobe_routes_to_linear_probe_head(self):
        from retina_screen.model import build_head, LinearProbeHead
        model = build_head(
            embedding_dim=1024,
            task_names=["dr_grade"],
            head_type="singletask_linearprobe_pertask",
        )
        assert isinstance(model, LinearProbeHead), (
            f"Expected LinearProbeHead, got {type(model).__name__}"
        )

    def test_singletask_mlp_routes_to_multitask_head(self):
        from retina_screen.model import build_head, MultiTaskHead
        model = build_head(
            embedding_dim=1024,
            task_names=["macular_edema"],
            head_type="singletask_mlp_pertask",
        )
        assert isinstance(model, MultiTaskHead), (
            f"Expected MultiTaskHead, got {type(model).__name__}"
        )

    def test_singletask_lp_binary_output_dim(self):
        """Binary task: output must be scalar (1) linear layer."""
        from retina_screen.model import build_head
        import torch
        model = build_head(1024, ["macular_edema"], head_type="singletask_linearprobe_pertask")
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1024))
        assert "macular_edema" in out
        assert out["macular_edema"].shape == (4,), (
            f"Expected (4,) scalar logit for binary task, got {out['macular_edema'].shape}"
        )

    def test_singletask_lp_ordinal_output_dim(self):
        """dr_grade (ORDINAL, 5 classes): output must be (B, 5)."""
        from retina_screen.model import build_head
        import torch
        model = build_head(1024, ["dr_grade"], head_type="singletask_linearprobe_pertask")
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1024))
        assert "dr_grade" in out
        assert out["dr_grade"].shape == (4, 5), (
            f"Expected (4, 5) for dr_grade ORDINAL, got {out['dr_grade'].shape}"
        )

    def test_singletask_mlp_binary_output_dim(self):
        """Binary task with MLP: output must be scalar (B,)."""
        from retina_screen.model import build_head
        import torch
        model = build_head(1024, ["diabetes"], head_type="singletask_mlp_pertask")
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1024))
        assert "diabetes" in out
        assert out["diabetes"].shape == (4,), (
            f"Expected (4,) scalar logit for binary task, got {out['diabetes'].shape}"
        )

    def test_singletask_mlp_ordinal_output_dim(self):
        """dr_grade with MLP: output must be (B, 5)."""
        from retina_screen.model import build_head
        import torch
        model = build_head(1024, ["dr_grade"], head_type="singletask_mlp_pertask")
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1024))
        assert "dr_grade" in out
        assert out["dr_grade"].shape == (4, 5), (
            f"Expected (4, 5) for dr_grade ORDINAL, got {out['dr_grade'].shape}"
        )

    def test_existing_linear_probe_still_routes_correctly(self):
        """Existing locked head_type='linear_probe' must still work."""
        from retina_screen.model import build_head, LinearProbeHead
        model = build_head(1024, ["macular_edema", "amd"], head_type="linear_probe")
        assert isinstance(model, LinearProbeHead)

    def test_existing_multitask_still_routes_correctly(self):
        """Existing locked head_type='multitask' must still work."""
        from retina_screen.model import build_head, MultiTaskHead
        model = build_head(1024, ["macular_edema", "amd"], head_type="multitask")
        assert isinstance(model, MultiTaskHead)

    def test_f1_lp_single_task_has_only_one_head(self):
        """F1 LP model with one task must have exactly one entry in task_heads."""
        from retina_screen.model import build_head
        model = build_head(1024, ["amd"], head_type="singletask_linearprobe_pertask")
        assert len(model.task_heads) == 1
        assert "amd" in model.task_heads

    def test_f1_mlp_single_task_has_only_one_head(self):
        """F1 MLP model with one task must have exactly one entry in task_heads."""
        from retina_screen.model import build_head
        model = build_head(1024, ["drusen"], head_type="singletask_mlp_pertask")
        assert len(model.task_heads) == 1
        assert "drusen" in model.task_heads
