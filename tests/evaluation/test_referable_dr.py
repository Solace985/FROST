"""
test_referable_dr.py -- Unit tests for Stage 8D-3.5 C1 referable-DR utility.

All tests use synthetic data only. No BRSET data is accessed.
Deterministic via np.random.default_rng(0).
"""

from __future__ import annotations

import numpy as np
import pytest

from retina_screen.evaluation.referable_dr import (
    REFERABLE_DR_TASK_METADATA,
    ZeroPositivesInResampleError,
    _prob_to_logit,
    compute_referable_dr_bootstrap_ci,
    compute_referable_dr_label,
    compute_referable_dr_pair_delta,
    compute_referable_dr_score,
    make_referable_dr_from_dr_grade_logits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(n: int, seed: int = 0) -> np.ndarray:
    """Random (N, 5) logit array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 5)).astype(np.float32)


def _make_synthetic_cell(
    n: int = 200,
    n_referable: int = 30,
    seed: int = 0,
) -> dict:
    """Create a synthetic cell with n_referable positive samples."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, 5)).astype(np.float32)
    labels = np.zeros(n, dtype=np.float64)
    pos_idx = rng.choice(n, size=n_referable, replace=False)
    labels[pos_idx] = rng.choice([2, 3, 4], size=n_referable)
    mask = np.ones(n, dtype=np.float64)
    # Assign 1 patient per 2 samples (approx)
    patient_ids = np.repeat(np.arange(n // 2), 2)[:n]
    return {
        "logits": logits,
        "labels": labels,
        "mask": mask,
        "patient_ids": patient_ids,
    }


# ---------------------------------------------------------------------------
# 1. Label mapping tests
# ---------------------------------------------------------------------------


def test_labels_0_1_map_to_nonreferable():
    """Grades 0 and 1 must map to binary label 0 (nonreferable)."""
    labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    result = compute_referable_dr_label(labels)
    assert np.all(result == 0.0), f"Expected all 0.0, got {result}"


def test_labels_2_3_4_map_to_referable():
    """Grades 2, 3, and 4 must map to binary label 1 (referable)."""
    labels = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    result = compute_referable_dr_label(labels)
    assert np.all(result == 1.0), f"Expected all 1.0, got {result}"


def test_missing_label_remains_missing():
    """Missing label sentinel (-1.0) must pass through unchanged."""
    labels = np.array([-1.0, 0.0, 2.0, -1.0], dtype=np.float64)
    result = compute_referable_dr_label(labels)
    assert result[0] == -1.0
    assert result[3] == -1.0
    assert result[1] == 0.0
    assert result[2] == 1.0


def test_mixed_labels_correct_mapping():
    """End-to-end: all five grades + missing map correctly."""
    labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0, -1.0], dtype=np.float64)
    expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    result = compute_referable_dr_label(labels)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 2. Score computation tests
# ---------------------------------------------------------------------------


def test_score_equals_softmax_classes_234():
    """Score must equal softmax(logits)[:,2:5].sum(axis=1) computed independently."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((50, 5)).astype(np.float64)

    # Reference implementation
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    expected = probs[:, 2:].sum(axis=1)

    result = compute_referable_dr_score(logits)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_softmax_numerical_stability():
    """Large logit values must not produce NaN or Inf."""
    logits = np.array([[1000.0, -1000.0, 500.0, 200.0, 100.0]], dtype=np.float64)
    result = compute_referable_dr_score(logits)
    assert np.isfinite(result).all(), f"Non-finite value: {result}"
    assert 0.0 <= float(result[0]) <= 1.0


def test_score_sums_to_complement_of_nonreferable():
    """P(referable) + P(nonreferable) must sum to 1.0 for each sample."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((20, 5)).astype(np.float32)

    arr = logits.astype(np.float64)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    referable_score = compute_referable_dr_score(logits)
    nonreferable_score = probs[:, :2].sum(axis=1)
    np.testing.assert_allclose(referable_score + nonreferable_score, 1.0, atol=1e-12)


def test_shape_mismatch_raises():
    """Non-(N,5) logit input must raise ValueError."""
    logits_bad = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        compute_referable_dr_score(logits_bad)


# ---------------------------------------------------------------------------
# 3. Mask exclusion tests
# ---------------------------------------------------------------------------


def test_mask_excludes_samples():
    """Samples with mask=0.0 must not appear in n_valid."""
    rng = np.random.default_rng(0)
    n = 100
    n_masked = 20
    logits = _make_logits(n)
    labels = rng.choice([0, 1, 2, 3, 4], size=n).astype(np.float64)
    mask = np.ones(n, dtype=np.float64)
    masked_idx = rng.choice(n, size=n_masked, replace=False)
    mask[masked_idx] = 0.0

    result = make_referable_dr_from_dr_grade_logits(logits, labels, mask)
    assert result["n_valid"] == n - n_masked


def test_all_missing_raises():
    """All mask=0.0 must raise ValueError before computing AUROC."""
    n = 50
    logits = _make_logits(n)
    labels = np.zeros(n, dtype=np.float64)
    mask = np.zeros(n, dtype=np.float64)  # all missing
    pids = np.arange(n, dtype=np.int64)

    with pytest.raises(ValueError):
        compute_referable_dr_bootstrap_ci(logits, labels, mask, pids, cell_name="test_all_masked")


# ---------------------------------------------------------------------------
# 4. Bootstrap determinism tests
# ---------------------------------------------------------------------------


def test_auroc_recomputation_deterministic():
    """Same inputs + same seed must yield identical AUROC CI."""
    cell = _make_synthetic_cell(n=200, n_referable=30, seed=1)
    result1 = compute_referable_dr_bootstrap_ci(
        cell["logits"], cell["labels"], cell["mask"], cell["patient_ids"],
        cell_name="det_test", n_resamples=200, seed=7,
    )
    result2 = compute_referable_dr_bootstrap_ci(
        cell["logits"], cell["labels"], cell["mask"], cell["patient_ids"],
        cell_name="det_test", n_resamples=200, seed=7,
    )
    auroc1 = next(t for t in result1.tasks if t.metric_name == "auroc")
    auroc2 = next(t for t in result2.tasks if t.metric_name == "auroc")
    assert auroc1.ci_lo == auroc2.ci_lo
    assert auroc1.ci_hi == auroc2.ci_hi
    assert auroc1.point_estimate == auroc2.point_estimate


def test_auprc_recomputation_deterministic():
    """Same inputs + same seed must yield identical AUPRC CI."""
    cell = _make_synthetic_cell(n=200, n_referable=30, seed=2)
    result1 = compute_referable_dr_bootstrap_ci(
        cell["logits"], cell["labels"], cell["mask"], cell["patient_ids"],
        cell_name="det_test_auprc", n_resamples=200, seed=7,
    )
    result2 = compute_referable_dr_bootstrap_ci(
        cell["logits"], cell["labels"], cell["mask"], cell["patient_ids"],
        cell_name="det_test_auprc", n_resamples=200, seed=7,
    )
    auprc1 = next((t for t in result1.tasks if t.metric_name == "auprc"), None)
    auprc2 = next((t for t in result2.tasks if t.metric_name == "auprc"), None)
    assert auprc1 is not None, "AUPRC result missing (sparse=True should force AUPRC)"
    assert auprc2 is not None
    assert auprc1.ci_lo == auprc2.ci_lo
    assert auprc1.ci_hi == auprc2.ci_hi


def test_patient_level_bootstrap_deterministic():
    """Fixed seed must produce identical CI across independent calls."""
    cell = _make_synthetic_cell(n=400, n_referable=40, seed=3)
    r1 = compute_referable_dr_bootstrap_ci(
        cell["logits"], cell["labels"], cell["mask"], cell["patient_ids"],
        cell_name="plbt", n_resamples=100, seed=42,
    )
    r2 = compute_referable_dr_bootstrap_ci(
        cell["logits"], cell["labels"], cell["mask"], cell["patient_ids"],
        cell_name="plbt", n_resamples=100, seed=42,
    )
    t1 = {t.metric_name: (t.ci_lo, t.ci_hi) for t in r1.tasks}
    t2 = {t.metric_name: (t.ci_lo, t.ci_hi) for t in r2.tasks}
    assert t1 == t2


# ---------------------------------------------------------------------------
# 5. Paired delta tests
# ---------------------------------------------------------------------------


def test_paired_bootstrap_is_deterministic():
    """Paired delta with same seed gives identical results."""
    ca = _make_synthetic_cell(n=200, n_referable=25, seed=10)
    cb = _make_synthetic_cell(n=200, n_referable=25, seed=11)

    d1 = compute_referable_dr_pair_delta(
        ca["logits"], ca["labels"], ca["mask"], ca["patient_ids"],
        cb["logits"], cb["labels"], cb["mask"], cb["patient_ids"],
        cell_name_a="A", cell_name_b="B", n_resamples=100, seed=42,
    )
    d2 = compute_referable_dr_pair_delta(
        ca["logits"], ca["labels"], ca["mask"], ca["patient_ids"],
        cb["logits"], cb["labels"], cb["mask"], cb["patient_ids"],
        cell_name_a="A", cell_name_b="B", n_resamples=100, seed=42,
    )
    td1 = {t.metric_name: (t.delta_ci_lo, t.delta_ci_hi) for t in d1.tasks}
    td2 = {t.metric_name: (t.delta_ci_lo, t.delta_ci_hi) for t in d2.tasks}
    assert td1 == td2


def test_sample_alignment_mismatch_raises():
    """compute_paired_delta_ci raises ValueError for mismatched patient sets."""
    rng = np.random.default_rng(0)
    n = 100

    ca = _make_synthetic_cell(n=n, n_referable=15, seed=5)
    cb = _make_synthetic_cell(n=n, n_referable=15, seed=6)

    # Different patient IDs for cell B
    cb["patient_ids"] = cb["patient_ids"] + 10000  # offset to create mismatch

    with pytest.raises(ValueError, match="patient"):
        compute_referable_dr_pair_delta(
            ca["logits"], ca["labels"], ca["mask"], ca["patient_ids"],
            cb["logits"], cb["labels"], cb["mask"], cb["patient_ids"],
            cell_name_a="A_mismatch", cell_name_b="B_mismatch",
            n_resamples=50, seed=0,
        )


# ---------------------------------------------------------------------------
# 6. Score invariance and anti-argmax tests
# ---------------------------------------------------------------------------


def test_no_use_of_predicted_argmax():
    """Score must be P(class ≥ 2), not argmax-based — ordering must differ."""
    # Construct logits where argmax = class 0 (nonreferable)
    # but P(class ≥ 2) is high (classes 2,3,4 summed > 0.9)
    #
    # logits: [10, 0, 5, 5, 5] → softmax ~ [0.999, ~0, ~0.0003, ~0.0003, ~0.0003]
    # No — flip: make argmax=0 but ensure 2+3+4 sum > 0.5
    # Actually: [0, 0, 5, 5, 5] → softmax: class0≈0.015, class1≈0.015, classes2-4≈0.97
    # argmax = 2 (still referable). Let's use: [5, 0, 3, 3, 3]
    # softmax: class0 ~ exp(5)/Z, classes2-4 ~ exp(3)/Z
    # P(class 0) = exp(5)/(exp(5)+exp(0)+3*exp(3)) ≈ 148/(148+1+60) ≈ 0.71
    # P(referable) ≈ 0.29
    # argmax = 0 (nonreferable) but P(referable) = 0.29 (could flip with different samples)

    # Test: ensure score = sum of softmax classes 2,3,4 (not just indicator of argmax >= 2)
    logits = np.array([[5.0, 0.0, 3.0, 3.0, 3.0]], dtype=np.float64)
    arr = logits - logits.max(axis=1, keepdims=True)  # = [[0, -5, -2, -2, -2]]
    exp_s = np.exp(arr)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)

    argmax_pred = int(np.argmax(logits[0]))
    expected_score = float(probs[0, 2:].sum())
    actual_score = float(compute_referable_dr_score(logits)[0])

    # Verify argmax is 0 (nonreferable) but score correctly reflects soft probability
    assert argmax_pred == 0, "Argmax should be class 0 in this test case"
    np.testing.assert_allclose(actual_score, expected_score, atol=1e-10)

    # Verify the score is NOT 0.0 (it would be if we used argmax >= 2 indicator)
    assert actual_score > 0.01, (
        f"Score={actual_score} suspiciously close to 0 — "
        "may be using argmax indicator instead of softmax probability."
    )


# ---------------------------------------------------------------------------
# 7. Single-class / degenerate case tests
# ---------------------------------------------------------------------------


def test_single_class_result_not_fake_auroc():
    """All-nonreferable labels must raise an error, not return AUROC=0.5."""
    rng = np.random.default_rng(0)
    n = 100
    logits = _make_logits(n)
    # All labels are grade 0 (nonreferable)
    labels = np.zeros(n, dtype=np.float64)
    mask = np.ones(n, dtype=np.float64)
    pids = np.arange(n, dtype=np.int64)

    # Should raise ValueError (pre-check) or ZeroPositivesInResampleError (bootstrap)
    with pytest.raises((ValueError, ZeroPositivesInResampleError)):
        compute_referable_dr_bootstrap_ci(
            logits, labels, mask, pids,
            cell_name="all_nonreferable_test",
            n_resamples=50,
            seed=0,
        )


# ---------------------------------------------------------------------------
# 8. prob_to_logit round-trip test
# ---------------------------------------------------------------------------


def test_prob_to_logit_round_trip():
    """sigmoid(logit(p)) ≈ p (rank-invariant equivalence)."""
    probs = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99], dtype=np.float64)
    logits = _prob_to_logit(probs)
    recovered = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(recovered, probs, atol=1e-6)


def test_prob_to_logit_monotone():
    """logit transform must be strictly monotone (preserve rank ordering)."""
    probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    logits = _prob_to_logit(probs)
    diffs = np.diff(logits)
    assert np.all(diffs > 0), f"Logit transform not monotone: {logits}"


# ---------------------------------------------------------------------------
# 9. REFERABLE_DR_TASK_METADATA validation
# ---------------------------------------------------------------------------


def test_task_metadata_correct_structure():
    """REFERABLE_DR_TASK_METADATA must have correct task_type and sparse=True."""
    assert "referable_dr" in REFERABLE_DR_TASK_METADATA
    meta = REFERABLE_DR_TASK_METADATA["referable_dr"]
    assert meta["task_type"] == "binary", "task_type must be 'binary'"
    assert meta.get("sparse") is True, "sparse must be True to force AUPRC computation"


# ---------------------------------------------------------------------------
# 10. Output files list test
# ---------------------------------------------------------------------------


def test_compatibility_output_files_produced():
    """REQUIRED_OUTPUT_FILES must contain exactly 13 files including all aliases."""
    from scripts.analysis.run_c1_referable_dr import REQUIRED_OUTPUT_FILES

    assert len(REQUIRED_OUTPUT_FILES) == 13, (
        f"Expected 13 output files, got {len(REQUIRED_OUTPUT_FILES)}"
    )

    canonical = {
        "c1_referable_dr_report.md",
        "c1_referable_dr_manifest.json",
        "referable_dr_cell_metrics.csv",
        "referable_dr_cell_metrics.json",
        "referable_dr_pairwise_deltas.csv",
        "referable_dr_pairwise_deltas.json",
        "dr_grade_5class_limitation_table.csv",
        "dr_grade_5class_limitation_table.json",
        "input_artifact_checksums.json",
    }
    aliases = {
        "c1_results.csv",
        "c1_results.json",
        "c1_pair_deltas.csv",
        "c1_report.md",
    }
    expected = canonical | aliases
    assert set(REQUIRED_OUTPUT_FILES) == expected, (
        f"Output file mismatch.\nExpected: {sorted(expected)}\nGot: {sorted(REQUIRED_OUTPUT_FILES)}"
    )
