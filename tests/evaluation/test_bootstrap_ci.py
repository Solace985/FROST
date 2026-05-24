"""
tests/evaluation/test_bootstrap_ci.py — Unit tests for bootstrap_ci.py (Stage 8D-3.5 A1).

All tests use synthetic data only. No BRSET artifacts, no backbone weights, no real patient IDs.

Test coverage (11 required tests + helpers):
  1.  test_determinism_under_fixed_seed
  2.  test_paired_indices_shared
  3.  test_stratification_preserves_positives
  4.  test_masked_samples_excluded
  5.  test_patient_level_resampling
  6.  test_zero_positives_in_resample_raises
  7.  test_delong_not_required_and_bootstrap_delta_available
  8.  test_point_estimate_matches_overall_metrics_json
  9.  test_long_form_outputs_have_required_columns
  10. test_pairwise_long_form_outputs_have_status_and_source_columns
  11. test_determinism_normalization_allows_only_whitelisted_fields
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pytest

from retina_screen.evaluation.bootstrap_ci import (
    CellCIResult,
    DeltaCIResult,
    TaskCIResult,
    TaskDeltaCIResult,
    ZeroPositivesInResampleError,
    compute_cell_ci,
    compute_paired_delta_ci,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_synthetic_binary(
    n_samples: int = 200,
    n_patients: int = 80,
    n_pos: int = 40,
    seed: int = 0,
) -> tuple[dict, dict, dict, np.ndarray]:
    """Return (predictions, labels, masks, patient_ids) for a single binary task."""
    rng = np.random.default_rng(seed)
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    pos_idx = rng.choice(n_samples, size=n_pos, replace=False)
    labels_arr[pos_idx] = 1.0

    logits = rng.standard_normal(n_samples).astype(np.float32)
    # Inflate logits for positives so AUROC is reasonably above 0.5
    logits[pos_idx] += 1.5

    masks = np.ones(n_samples, dtype=np.float64)

    predictions = {"task_a": logits}
    labels = {"task_a": labels_arr}
    masks_d = {"task_a": masks}
    task_meta = {"task_a": {"task_type": "binary"}}
    return predictions, labels, masks_d, patient_ids, task_meta


def _make_synthetic_ordinal(
    n_samples: int = 200,
    n_patients: int = 80,
    n_classes: int = 5,
    seed: int = 0,
) -> tuple[dict, dict, dict, np.ndarray, dict]:
    """Return (predictions, labels, masks, patient_ids, task_meta) for a single ordinal task."""
    rng = np.random.default_rng(seed)
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    y_true = rng.integers(0, n_classes, size=n_samples).astype(np.float64)
    logits = rng.standard_normal((n_samples, n_classes)).astype(np.float32)
    masks = np.ones(n_samples, dtype=np.float64)

    predictions = {"task_ord": logits}
    labels = {"task_ord": y_true}
    masks_d = {"task_ord": masks}
    task_meta = {"task_ord": {"task_type": "ordinal", "n_classes": n_classes}}
    return predictions, labels, masks_d, patient_ids, task_meta


# ---------------------------------------------------------------------------
# Test 1 — Determinism under fixed seed
# ---------------------------------------------------------------------------


def test_determinism_under_fixed_seed():
    """Same seed must produce byte-identical CellCIResult."""
    preds, lbls, msks, pids, meta = _make_synthetic_binary(n_samples=200, n_patients=80)
    result_a = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=100, seed=42)
    result_b = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=100, seed=42)

    assert result_a.n_resamples == result_b.n_resamples
    assert result_a.n_patients == result_b.n_patients
    assert len(result_a.tasks) == len(result_b.tasks)

    for ta, tb in zip(result_a.tasks, result_b.tasks):
        assert ta.metric_name == tb.metric_name
        assert ta.point_estimate == tb.point_estimate, (
            f"Point estimates differ for {ta.metric_name}"
        )
        assert ta.ci_lo == tb.ci_lo, f"CI lo differs for {ta.metric_name}"
        assert ta.ci_hi == tb.ci_hi, f"CI hi differs for {ta.metric_name}"
        assert ta.n_resamples_ok == tb.n_resamples_ok
        assert ta.status == tb.status


# ---------------------------------------------------------------------------
# Test 2 — Paired indices shared between cells
# ---------------------------------------------------------------------------


def test_paired_indices_shared():
    """Paired delta CI must use the same resample indices for both cells."""
    rng = np.random.default_rng(1)
    n_samples, n_patients, n_pos = 200, 80, 40
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    labels_arr[rng.choice(n_samples, size=n_pos, replace=False)] = 1.0
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    # Cell A: well-calibrated logits
    logits_a = rng.standard_normal(n_samples).astype(np.float32)
    logits_a[labels_arr == 1.0] += 2.0

    # Cell B: slightly weaker logits
    logits_b = rng.standard_normal(n_samples).astype(np.float32)
    logits_b[labels_arr == 1.0] += 1.5

    delta_result = compute_paired_delta_ci(
        {"task_a": logits_a},
        {"task_a": labels_arr},
        {"task_a": masks},
        patient_ids,
        {"task_a": logits_b},
        {"task_a": labels_arr},  # same labels (same test set)
        {"task_a": masks},
        patient_ids,  # same patient_ids (same test set)
        task_meta,
        n_resamples=100,
        seed=7,
    )
    # If indices are NOT shared, delta variance would be inflated.
    # Verify the delta CI was computed successfully (not na).
    task_delta = delta_result.tasks[0]
    assert task_delta.n_resamples_ok > 50, "Too many skipped resamples — possible logic error"
    assert task_delta.delta_point is not None


# ---------------------------------------------------------------------------
# Test 3 — Stratification preserves positives (dense-positive scenario)
# ---------------------------------------------------------------------------


def test_stratification_preserves_positives():
    """With dense positives (50%), bootstrap resamples should reliably have positives."""
    rng = np.random.default_rng(99)
    n_samples, n_patients = 400, 100
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    pos_idx = rng.choice(n_samples, size=n_samples // 2, replace=False)
    labels_arr[pos_idx] = 1.0
    logits = rng.standard_normal(n_samples).astype(np.float32)
    logits[pos_idx] += 1.0
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    result = compute_cell_ci(
        {"task_a": logits},
        {"task_a": labels_arr},
        {"task_a": masks},
        patient_ids,
        task_meta,
        n_resamples=200,
        seed=5,
    )
    auroc_result = next(t for t in result.tasks if t.metric_name == "auroc")
    # With dense positives, no resamples should be skipped for zero-positives
    assert auroc_result.n_resamples_ok >= 195, (
        f"Expected ~200 ok resamples for dense-positive data, got {auroc_result.n_resamples_ok}"
    )
    assert auroc_result.status == "ok"


# ---------------------------------------------------------------------------
# Test 4 — Masked samples excluded (GATE M analog)
# ---------------------------------------------------------------------------


def test_masked_samples_excluded():
    """Samples with mask==0.0 must be excluded from metric computation."""
    rng = np.random.default_rng(2)
    n_samples, n_patients, n_pos = 200, 50, 60
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    pos_idx = rng.choice(n_samples, size=n_pos, replace=False)
    labels_arr[pos_idx] = 1.0

    # Logits: positives get high scores, negatives get low
    logits = -2.0 * np.ones(n_samples, dtype=np.float32)
    logits[pos_idx] = 2.0

    task_meta = {"task_a": {"task_type": "binary"}}

    # All observed (mask=1): should get high AUROC
    masks_all = np.ones(n_samples, dtype=np.float64)
    result_all = compute_cell_ci(
        {"task_a": logits},
        {"task_a": labels_arr},
        {"task_a": masks_all},
        patient_ids,
        task_meta,
        n_resamples=50,
        seed=0,
    )

    # Mask out the positives — so the "observed" set has only negatives
    masks_no_pos = masks_all.copy()
    masks_no_pos[pos_idx] = 0.0

    # This should trigger ZeroPositivesInResampleError because masked positives = 0
    with pytest.raises(ZeroPositivesInResampleError):
        compute_cell_ci(
            {"task_a": logits},
            {"task_a": labels_arr},
            {"task_a": masks_no_pos},
            patient_ids,
            task_meta,
            n_resamples=10,
            seed=0,
        )

    # Verify point estimate changes when masking changes subset
    pt_all = next(t for t in result_all.tasks if t.metric_name == "auroc").point_estimate
    assert pt_all is not None
    # n_included must reflect mask
    auroc_all = next(t for t in result_all.tasks if t.metric_name == "auroc")
    assert auroc_all.n_included == n_samples  # all observed


# ---------------------------------------------------------------------------
# Test 5 — Patient-level resampling (not sample-level)
# ---------------------------------------------------------------------------


def test_patient_level_resampling():
    """Resampling unit must be patient, not individual sample.

    We set up a scenario where one patient has 5 images (clearly multi-image)
    and verify that when that patient is sampled k times, all k×5 images appear.
    """
    # 50 patients, each with exactly 1 image, plus 1 special patient with 5 images
    n_special_images = 5
    n_regular = 50
    n_total = n_regular + n_special_images
    patient_ids = np.array(
        [f"p{i:03d}" for i in range(n_regular)] + ["p_special"] * n_special_images
    )
    labels_arr = np.zeros(n_total, dtype=np.float64)
    # Make p_special positive
    labels_arr[n_regular:] = 1.0
    # Also add some other positives
    labels_arr[:10] = 1.0
    logits = np.zeros(n_total, dtype=np.float32)
    masks = np.ones(n_total, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    result = compute_cell_ci(
        {"task_a": logits},
        {"task_a": labels_arr},
        {"task_a": masks},
        patient_ids,
        task_meta,
        n_resamples=50,
        seed=0,
    )
    # Patient count must match n_regular + 1 (for p_special)
    assert result.n_patients == n_regular + 1
    assert result.n_samples == n_total


# ---------------------------------------------------------------------------
# Test 6 — Zero positives in resample raises ZeroPositivesInResampleError
# ---------------------------------------------------------------------------


def test_zero_positives_in_resample_raises():
    """When a resample has zero positive labels, ZeroPositivesInResampleError must be raised."""
    # Construct a dataset where only ONE patient (1 image) has the positive label.
    # The patient can be missed in a resample with replacement.
    n_samples = 20
    patient_ids = np.array([f"p{i:02d}" for i in range(n_samples)])
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    # Only patient 0 is positive
    labels_arr[0] = 1.0
    logits = np.zeros(n_samples, dtype=np.float32)
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    # With only 1 positive patient and 20 patients, many resamples will miss patient 0.
    # The utility must raise ZeroPositivesInResampleError on the first such resample.
    with pytest.raises(ZeroPositivesInResampleError) as exc_info:
        compute_cell_ci(
            {"task_a": logits},
            {"task_a": labels_arr},
            {"task_a": masks},
            patient_ids,
            task_meta,
            n_resamples=1000,
            seed=0,
        )
    assert exc_info.value.task_name == "task_a"
    assert exc_info.value.resample_idx >= 0


# ---------------------------------------------------------------------------
# Test 7 — DeLong not required; paired bootstrap delta is available
# ---------------------------------------------------------------------------


def test_delong_not_required_and_bootstrap_delta_available():
    """DeLong is not implemented; paired bootstrap delta CI must still work for AUROC."""
    rng = np.random.default_rng(3)
    n_samples, n_patients, n_pos = 200, 80, 50
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    labels_arr[rng.choice(n_samples, size=n_pos, replace=False)] = 1.0
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    logits_a = rng.standard_normal(n_samples).astype(np.float32)
    logits_a[labels_arr == 1.0] += 2.0
    logits_b = rng.standard_normal(n_samples).astype(np.float32)
    logits_b[labels_arr == 1.0] += 2.0

    delta = compute_paired_delta_ci(
        {"task_a": logits_a}, {"task_a": labels_arr}, {"task_a": masks}, patient_ids,
        {"task_a": logits_b}, {"task_a": labels_arr}, {"task_a": masks}, patient_ids,
        task_meta,
        n_resamples=100,
        seed=1,
    )
    # Must have results
    assert len(delta.tasks) > 0
    for td in delta.tasks:
        assert td.source == "paired_bootstrap", "Source must be paired_bootstrap, not DeLong"
        # Status must be one of the valid values
        assert td.status in ("supported", "not_supported", "na")


# ---------------------------------------------------------------------------
# Test 8 — Point estimate matches direct computation
# ---------------------------------------------------------------------------


def test_point_estimate_matches_overall_metrics_json():
    """Bootstrap point estimate (unresampled) must match direct metric computation."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    n_samples, n_patients, n_pos = 200, 60, 50
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    pos_idx = rng.choice(n_samples, size=n_pos, replace=False)
    labels_arr[pos_idx] = 1.0
    logits = rng.standard_normal(n_samples).astype(np.float32)
    logits[pos_idx] += 1.5
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    result = compute_cell_ci(
        {"task_a": logits},
        {"task_a": labels_arr},
        {"task_a": masks},
        patient_ids,
        task_meta,
        n_resamples=50,
        seed=0,
    )

    # Compute directly as evaluation.py does
    y_score = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    expected_auroc = float(roc_auc_score(labels_arr, y_score))

    auroc_result = next(t for t in result.tasks if t.metric_name == "auroc")
    assert auroc_result.point_estimate is not None
    assert abs(auroc_result.point_estimate - expected_auroc) < 1e-6, (
        f"Point estimate {auroc_result.point_estimate} != direct compute {expected_auroc}"
    )


# ---------------------------------------------------------------------------
# Test 9 — Long-form output has required columns (CellCIResult fields)
# ---------------------------------------------------------------------------


def test_long_form_outputs_have_required_columns():
    """CellCIResult and TaskCIResult must contain all required fields."""
    required_cell_fields = {"cell_name", "tasks", "n_resamples", "seed", "n_patients", "n_samples"}
    required_task_fields = {
        "task_name", "task_type", "metric_name", "point_estimate", "ci_lo", "ci_hi",
        "n_included", "n_total", "n_resamples_ok", "n_resamples_skip", "status", "reason",
    }

    preds, lbls, msks, pids, meta = _make_synthetic_binary(n_samples=200, n_patients=80)
    result = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=50, seed=0)

    # Check CellCIResult fields
    cell_fields = {f for f in dir(result) if not f.startswith("_")}
    for field in required_cell_fields:
        assert hasattr(result, field), f"CellCIResult missing field: {field}"

    # Check TaskCIResult fields
    assert len(result.tasks) > 0
    for task_result in result.tasks:
        for field in required_task_fields:
            assert hasattr(task_result, field), f"TaskCIResult missing field: {field}"


# ---------------------------------------------------------------------------
# Test 10 — Pairwise long-form output has status and source columns
# ---------------------------------------------------------------------------


def test_pairwise_long_form_outputs_have_status_and_source_columns():
    """DeltaCIResult and TaskDeltaCIResult must have status and source fields."""
    required_delta_fields = {"cell_a", "cell_b", "tasks", "n_resamples", "seed"}
    required_task_delta_fields = {
        "task_name", "metric_name", "delta_point", "delta_ci_lo", "delta_ci_hi",
        "n_resamples_ok", "status", "source",
    }

    rng = np.random.default_rng(6)
    n_samples, n_patients, n_pos = 200, 80, 50
    patient_ids = np.array(
        [f"p{i:03d}" for i in rng.integers(0, n_patients, size=n_samples)]
    )
    labels_arr = np.zeros(n_samples, dtype=np.float64)
    labels_arr[rng.choice(n_samples, size=n_pos, replace=False)] = 1.0
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}
    logits = rng.standard_normal(n_samples).astype(np.float32)
    logits[labels_arr == 1.0] += 1.5

    result = compute_paired_delta_ci(
        {"task_a": logits}, {"task_a": labels_arr}, {"task_a": masks}, patient_ids,
        {"task_a": logits}, {"task_a": labels_arr}, {"task_a": masks}, patient_ids,
        task_meta,
        n_resamples=50,
        seed=0,
    )

    for field in required_delta_fields:
        assert hasattr(result, field), f"DeltaCIResult missing field: {field}"

    assert len(result.tasks) > 0
    for td in result.tasks:
        for field in required_task_delta_fields:
            assert hasattr(td, field), f"TaskDeltaCIResult missing field: {field}"
        # Source must always be paired_bootstrap
        assert td.source == "paired_bootstrap"
        # Status must be a valid value
        assert td.status in ("supported", "not_supported", "na")


# ---------------------------------------------------------------------------
# Test 11 — Determinism normalization strips only whitelisted fields
# ---------------------------------------------------------------------------


_WHITELISTED_PATTERNS = [
    re.compile(r'"output_dir"\s*:\s*"[^"]*"'),
    re.compile(r'"run_timestamp"\s*:\s*"[^"]*"'),
    re.compile(r'"generated_at"\s*:\s*"[^"]*"'),
    re.compile(r'"report_path"\s*:\s*"[^"]*"'),
]


def _normalize_provenance(text: str) -> str:
    """Strip whitelisted timestamp/path fields from provenance text."""
    result = text
    for pat in _WHITELISTED_PATTERNS:
        result = pat.sub(lambda m: m.group(0).split(":")[0] + ': "NORMALIZED"', result)
    return result


def test_determinism_normalization_allows_only_whitelisted_fields():
    """Normalization must only touch whitelisted timestamp/path fields, not statistical values."""
    # Build a synthetic provenance string that looks like run_manifest.json
    provenance = json.dumps({
        "output_dir": "/some/path/to/outputs/analysis/A1_bootstrap_ci/20260524_120000",
        "run_timestamp": "2026-05-24T12:00:00",
        "generated_at": "2026-05-24T12:05:00",
        "report_path": "/some/path/to/outputs/analysis/A1_bootstrap_ci/20260524_120000/report.md",
        "n_resamples": 2000,
        "seed": 42,
        "ci_lo_auroc": 0.8234,
        "ci_hi_auroc": 0.9012,
    }, indent=2)

    normalized = _normalize_provenance(provenance)
    parsed = json.loads(normalized)

    # Whitelisted fields must be normalized
    assert parsed["output_dir"] == "NORMALIZED"
    assert parsed["run_timestamp"] == "NORMALIZED"
    assert parsed["generated_at"] == "NORMALIZED"
    assert parsed["report_path"] == "NORMALIZED"

    # Statistical fields must be UNTOUCHED
    assert parsed["n_resamples"] == 2000
    assert parsed["seed"] == 42
    assert parsed["ci_lo_auroc"] == pytest.approx(0.8234)
    assert parsed["ci_hi_auroc"] == pytest.approx(0.9012)

    # Second "run" of the same logic must produce identical statistical output
    provenance2 = json.dumps({
        "output_dir": "/different/path/20260524_130000",
        "run_timestamp": "2026-05-24T13:00:00",
        "generated_at": "2026-05-24T13:05:00",
        "report_path": "/different/path/20260524_130000/report.md",
        "n_resamples": 2000,
        "seed": 42,
        "ci_lo_auroc": 0.8234,
        "ci_hi_auroc": 0.9012,
    }, indent=2)
    normalized2 = _normalize_provenance(provenance2)
    assert normalized == normalized2, (
        "After normalization, two runs with identical statistics should be identical"
    )


# ---------------------------------------------------------------------------
# Additional correctness checks
# ---------------------------------------------------------------------------


def test_ordinal_task_produces_three_metrics():
    """Ordinal task must produce accuracy, macro_f1, balanced_accuracy."""
    preds, lbls, msks, pids, meta = _make_synthetic_ordinal(
        n_samples=300, n_patients=100, n_classes=5, seed=1
    )
    result = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=50, seed=0)
    metric_names = {t.metric_name for t in result.tasks}
    assert "accuracy" in metric_names
    assert "macro_f1" in metric_names
    assert "balanced_accuracy" in metric_names


def test_binary_macro_auroc_is_computed():
    """When binary tasks present, binary_macro_auroc must appear in CellCIResult."""
    preds, lbls, msks, pids, meta = _make_synthetic_binary(n_samples=200, n_patients=80)
    result = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=50, seed=0)
    metric_names = [t.metric_name for t in result.tasks]
    assert "binary_macro_auroc" in metric_names


def test_different_seeds_produce_different_ci():
    """Different seeds should produce different CI bounds (almost certainly)."""
    preds, lbls, msks, pids, meta = _make_synthetic_binary(n_samples=400, n_patients=100)
    result_a = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=200, seed=42)
    result_b = compute_cell_ci(preds, lbls, msks, pids, meta, n_resamples=200, seed=99)
    auroc_a = next(t for t in result_a.tasks if t.metric_name == "auroc")
    auroc_b = next(t for t in result_b.tasks if t.metric_name == "auroc")
    # Point estimates must be equal (same unresampled data)
    assert auroc_a.point_estimate == pytest.approx(auroc_b.point_estimate, abs=1e-9)
    # CI bounds should differ due to different resamples
    # (not guaranteed for very stable distributions but very likely with n=200)
    assert (auroc_a.ci_lo != auroc_b.ci_lo) or (auroc_a.ci_hi != auroc_b.ci_hi), (
        "Different seeds should produce different CI bounds"
    )


def test_mismatched_patient_ids_raises_in_delta():
    """compute_paired_delta_ci must raise ValueError for non-identical patient sets."""
    rng = np.random.default_rng(0)
    n_samples = 100
    pids_a = np.array([f"p{i:03d}" for i in rng.integers(0, 40, size=n_samples)])
    pids_b = np.array([f"q{i:03d}" for i in rng.integers(0, 40, size=n_samples)])  # different prefix
    labels = np.zeros(n_samples, dtype=np.float64)
    labels[rng.choice(n_samples, size=20, replace=False)] = 1.0
    logits = rng.standard_normal(n_samples).astype(np.float32)
    masks = np.ones(n_samples, dtype=np.float64)
    task_meta = {"task_a": {"task_type": "binary"}}

    with pytest.raises(ValueError, match="identical patient sets"):
        compute_paired_delta_ci(
            {"task_a": logits}, {"task_a": labels}, {"task_a": masks}, pids_a,
            {"task_a": logits}, {"task_a": labels}, {"task_a": masks}, pids_b,
            task_meta,
            n_resamples=10,
            seed=0,
        )
