from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def evaluate_mod():
    spec = importlib.util.spec_from_file_location(
        "_scripts_05_evaluate",
        _PROJECT_ROOT / "scripts" / "05_evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_checkpoint(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_checkpoint.pt").write_bytes(b"")


def test_train_preferred_over_fast_dev_run(evaluate_mod, tmp_path, monkeypatch) -> None:
    """train run must be returned when both train and fast_dev_run exist."""
    fast_dir = tmp_path / "runs" / "fast_dev_run" / "brset_20260507_old"
    train_dir = tmp_path / "runs" / "train" / "brset_20260511_new"
    _make_checkpoint(fast_dir)
    _make_checkpoint(train_dir)

    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")

    assert result is not None, "_latest_run_dir returned None"
    assert "train" in str(result), (
        f"Expected 'train' run to be preferred, got: {result}"
    )
    assert "fast_dev_run" not in str(result), (
        f"fast_dev_run must not shadow train run, got: {result}"
    )


def test_fast_dev_run_fallback_when_no_train(evaluate_mod, tmp_path, monkeypatch) -> None:
    """fast_dev_run must be returned when no train run exists."""
    fast_dir = tmp_path / "runs" / "fast_dev_run" / "brset_20260507_smoke"
    _make_checkpoint(fast_dir)

    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")

    assert result is not None, "_latest_run_dir returned None when fast_dev_run should be fallback"
    assert "fast_dev_run" in str(result), (
        f"Expected fast_dev_run fallback, got: {result}"
    )


def test_none_when_no_run_exists(evaluate_mod, tmp_path, monkeypatch) -> None:
    """None must be returned when neither train nor fast_dev_run has a checkpoint."""
    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")
    assert result is None


def test_latest_train_run_selected_by_name(evaluate_mod, tmp_path, monkeypatch) -> None:
    """When multiple train runs exist, the lexicographically latest name is selected."""
    for name in ["brset_20260510_early", "brset_20260511_latest", "brset_20260509_oldest"]:
        _make_checkpoint(tmp_path / "runs" / "train" / name)

    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")

    assert result is not None
    assert result.name == "brset_20260511_latest", (
        f"Expected latest-named run, got: {result.name}"
    )


def test_dataset_prefix_filter(evaluate_mod, tmp_path, monkeypatch) -> None:
    """Runs for a different dataset must not be returned."""
    _make_checkpoint(tmp_path / "runs" / "train" / "odir_20260511_run")
    monkeypatch.chdir(tmp_path)
    result = evaluate_mod._latest_run_dir("brset")
    assert result is None, f"Expected None for brset query when only odir run exists, got: {result}"




def _make_full_run_dir(run_dir: Path, head_type: str | None = None) -> None:
    """Create a minimal full/internal run directory for testing."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_checkpoint.pt").write_bytes(b"")
    resolved: dict = {
        "dataset": "brset",
        "backbone": "resnet50",
        "task_config": "configs/tasks/brset_default.yaml",
        "fast_dev_run": False,
        "rehearsal": False,
        "run_mode": "stage8d2_brset_full_resnet50_multitask",
    }
    if head_type is not None:
        resolved["head_type"] = head_type
    import yaml  # noqa: PLC0415
    (run_dir / "resolved_config.yaml").write_text(
        yaml.dump(resolved), encoding="utf-8"
    )


def _full_eval_cfg(head_type: str | None = None) -> dict:
    """Minimal full/internal eval config dict."""
    cfg: dict = {
        "dataset": "brset",
        "backbone": "resnet50",
        "task_config": "configs/tasks/brset_default.yaml",
        "full_dataset_run": True,
    }
    if head_type is not None:
        cfg["head_type"] = head_type
    return cfg


def test_head_type_mismatch_linear_probe_eval_multitask_run(evaluate_mod, tmp_path) -> None:
    """linear_probe eval config + multitask run → hard fail (prevents mislabeled matrix results)."""
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_multitask"
    _make_full_run_dir(run_dir, head_type="multitask")
    eval_cfg = _full_eval_cfg(head_type="linear_probe")
    with pytest.raises(SystemExit) as exc_info:
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)
    assert exc_info.value.code != 0, "Must exit with non-zero code on head_type mismatch"


def test_head_type_mismatch_multitask_eval_linear_probe_run(evaluate_mod, tmp_path) -> None:
    """multitask eval config + linear_probe run → hard fail (reverse direction)."""
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_linear"
    _make_full_run_dir(run_dir, head_type="linear_probe")
    eval_cfg = _full_eval_cfg(head_type="multitask")
    with pytest.raises(SystemExit) as exc_info:
        evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)
    assert exc_info.value.code != 0, "Must exit with non-zero code on head_type mismatch"


def test_head_type_match_linear_probe_both(evaluate_mod, tmp_path) -> None:
    """linear_probe eval + linear_probe run → passes."""
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_lp"
    _make_full_run_dir(run_dir, head_type="linear_probe")
    eval_cfg = _full_eval_cfg(head_type="linear_probe")
    evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_head_type_match_multitask_both(evaluate_mod, tmp_path) -> None:
    """multitask eval + multitask run → passes."""
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_mt"
    _make_full_run_dir(run_dir, head_type="multitask")
    eval_cfg = _full_eval_cfg(head_type="multitask")
    evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_head_type_both_missing_normalizes_to_multitask(evaluate_mod, tmp_path) -> None:
    """Both eval config and run resolved_config missing head_type → normalizes to multitask, passes.

    This is the backward-compatibility path for Stage 8D-2 artifacts.
    """
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_legacy"
    _make_full_run_dir(run_dir, head_type=None)
    eval_cfg = _full_eval_cfg(head_type=None)
    evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_head_type_eval_missing_run_multitask_passes(evaluate_mod, tmp_path) -> None:
    """Eval config missing head_type + run head_type=multitask → passes (both normalize to multitask)."""
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_mt2"
    _make_full_run_dir(run_dir, head_type="multitask")
    eval_cfg = _full_eval_cfg(head_type=None)
    evaluate_mod._validate_run_dir_for_eval(run_dir, eval_cfg)


def test_head_type_eval_config_path_in_error(evaluate_mod, tmp_path) -> None:
    """eval_config_path must appear in the mismatch error message."""
    run_dir = tmp_path / "runs" / "train" / "brset_20260511_msg"
    _make_full_run_dir(run_dir, head_type="multitask")
    eval_cfg = _full_eval_cfg(head_type="linear_probe")

    import io  # noqa: PLC0415
    import logging  # noqa: PLC0415
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(handler)
    try:
        with pytest.raises(SystemExit):
            evaluate_mod._validate_run_dir_for_eval(
                run_dir, eval_cfg,
                eval_config_path="configs/experiment/stage8d3b_brset_resnet50_full_linear_probe.yaml",
            )
    finally:
        logging.getLogger().removeHandler(handler)

    log_output = log_capture.getvalue()
    assert "stage8d3b" in log_output or "linear_probe" in log_output, (
        "Error message must mention eval config path or head_type details"
    )
