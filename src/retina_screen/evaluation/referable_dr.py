"""
referable_dr.py -- Referable-DR binary endpoint recast from dr_grade ordinal predictions.

Stage 8D-3.5 C1 implementation.

Design
------
Referable-DR is the clinically meaningful binary endpoint derived from the 5-class
dr_grade ordinal task. Grade ≥ 2 (moderate DR or worse) is defined as referable.

This module provides:
  1. Pure array transformations (no I/O, no dataset name coupling)
  2. Thin wrappers over bootstrap_ci.compute_cell_ci / compute_paired_delta_ci

Score  : P(dr_grade ≥ 2) = softmax(logit__dr_grade)[:,2:5].sum(axis=1)
Label  : grade 0/1 → 0 (nonreferable), grade 2/3/4 → 1 (referable)
Missing: label == -1 → excluded (mask set to 0.0 before passing to bootstrap_ci)

ZeroPositivesInResampleError from bootstrap_ci is NOT caught here; it propagates
to the driver. The driver must halt with BLOCKED_C1_BOOTSTRAP_ZERO_POSITIVE_FAILURE
and report the affected cell/pair — do NOT silently return NA.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from retina_screen.evaluation.bootstrap_ci import (
    CellCIResult,
    DeltaCIResult,
    ZeroPositivesInResampleError,  # re-exported so callers need not import bootstrap_ci
    compute_cell_ci,
    compute_paired_delta_ci,
)

__all__ = [
    "REFERABLE_DR_TASK_METADATA",
    "ZeroPositivesInResampleError",
    "compute_referable_dr_score",
    "compute_referable_dr_label",
    "make_referable_dr_from_dr_grade_logits",
    "_prob_to_logit",
    "compute_referable_dr_bootstrap_ci",
    "compute_referable_dr_pair_delta",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task metadata for bootstrap_ci wrappers
# ---------------------------------------------------------------------------

REFERABLE_DR_TASK_METADATA: dict[str, dict[str, Any]] = {
    "referable_dr": {
        "task_type": "binary",
        "sparse": True,  # always compute AUPRC; n_pos=73 for BRSET test set (sparse endpoint)
    }
}

# ---------------------------------------------------------------------------
# Pure array functions — no I/O, no dataset coupling
# ---------------------------------------------------------------------------


def compute_referable_dr_score(logit_dr_grade: np.ndarray) -> np.ndarray:
    """Compute P(dr_grade ≥ 2) = softmax(logits)[:,2:5].sum(axis=1).

    Parameters
    ----------
    logit_dr_grade:
        Shape (N, 5). Raw per-class logits for the dr_grade task.

    Returns
    -------
    np.ndarray of shape (N,), float64, values in [0, 1].
    """
    if logit_dr_grade.ndim != 2 or logit_dr_grade.shape[1] != 5:
        raise ValueError(
            f"logit_dr_grade must have shape (N, 5), got {logit_dr_grade.shape}"
        )
    # Numerically stable softmax: subtract per-row maximum before exp
    arr = logit_dr_grade.astype(np.float64)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    # Sum classes 2, 3, 4 — the referable grades
    return probs[:, 2:].sum(axis=1)


def compute_referable_dr_label(
    label_dr_grade: np.ndarray,
    missing_value: float = -1.0,
    referable_min_grade: int = 2,
) -> np.ndarray:
    """Map dr_grade integer labels to referable-DR binary labels.

    Mapping:
      grade 0, 1          → 0.0  (nonreferable)
      grade 2, 3, 4       → 1.0  (referable: moderate/severe DR, PDR)
      label == missing_value → missing_value sentinel (to be masked out)

    Parameters
    ----------
    label_dr_grade:
        Shape (N,). Integer labels in {0,1,2,3,4} or missing_value.
    missing_value:
        Placeholder for missing/unknown labels. Default -1.0.
    referable_min_grade:
        Minimum grade that is considered referable. Default 2.

    Returns
    -------
    np.ndarray of shape (N,), float64.
    Values: 0.0 (nonreferable), 1.0 (referable), or missing_value (missing).
    """
    arr = np.asarray(label_dr_grade, dtype=np.float64)
    is_missing = arr == missing_value
    is_referable = (~is_missing) & (arr >= referable_min_grade)
    result = np.where(is_missing, missing_value, np.where(is_referable, 1.0, 0.0))
    return result.astype(np.float64)


def make_referable_dr_from_dr_grade_logits(
    logits_5class: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    patient_ids: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    referable_min_grade: int = 2,
) -> dict:
    """Filter to valid (mask==1.0) samples and compute referable-DR score and label.

    Samples with mask==0.0 are excluded. Within the valid subset, any remaining
    missing-label samples (label == -1.0) are also excluded.

    Parameters
    ----------
    logits_5class:
        Shape (N, 5). Raw per-class logits for dr_grade.
    labels:
        Shape (N,). dr_grade labels (0–4) or -1.0 for missing.
    mask:
        Shape (N,). float64 mask: 1.0 = observed, 0.0 = missing.
    patient_ids:
        Shape (N,), optional. Patient identifiers (passed through for context).
    sample_ids:
        Shape (N,), optional. Sample identifiers (passed through for context).
    referable_min_grade:
        Minimum grade for referable classification. Default 2.

    Returns
    -------
    dict with keys:
        'score'       : (N_valid,) float64 — P(dr_grade ≥ referable_min_grade)
        'label'       : (N_valid,) float64 — 0.0 (nonreferable) or 1.0 (referable)
        'patient_ids' : (N_valid,) or None
        'sample_ids'  : (N_valid,) or None
        'n_valid'     : int — samples after mask and missing-label exclusion
        'n_pos'       : int — referable positives (label==1.0)
        'n_neg'       : int — nonreferable samples (label==0.0)
    """
    valid = mask == 1.0
    logits_v = logits_5class[valid]
    labels_v = labels[valid]

    ref_label = compute_referable_dr_label(labels_v, referable_min_grade=referable_min_grade)
    score = compute_referable_dr_score(logits_v)

    # Within the masked-valid set, further exclude any missing-label samples
    label_ok = ref_label != -1.0
    score_out = score[label_ok]
    label_out = ref_label[label_ok]

    n_pos = int((label_out == 1.0).sum())
    n_neg = int((label_out == 0.0).sum())

    pids_out = patient_ids[valid][label_ok] if patient_ids is not None else None
    sids_out = sample_ids[valid][label_ok] if sample_ids is not None else None

    return {
        "score": score_out,
        "label": label_out,
        "patient_ids": pids_out,
        "sample_ids": sids_out,
        "n_valid": len(score_out),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def _prob_to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert probability to logit space for bootstrap_ci compatibility.

    bootstrap_ci.py applies sigmoid internally to binary inputs. Since AUROC and
    AUPRC are rank-invariant under any strictly monotone transform, passing
    logit(p) gives identical metric values to passing raw probabilities — the
    sigmoid inside bootstrap_ci exactly inverts the logit transform.

    logit(p) = log(p / (1-p))

    Parameters
    ----------
    p:
        Probabilities; values will be clipped to [eps, 1-eps] before log.
    eps:
        Clipping value to guard against log(0). Default 1e-7.

    Returns
    -------
    np.ndarray of same shape, float64.
    """
    p_clipped = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


# ---------------------------------------------------------------------------
# bootstrap_ci wrappers
# ---------------------------------------------------------------------------


def compute_referable_dr_bootstrap_ci(
    logit_dr_grade: np.ndarray,
    label_dr_grade: np.ndarray,
    mask_dr_grade: np.ndarray,
    patient_ids: np.ndarray,
    cell_name: str,
    n_resamples: int = 2000,
    seed: int = 42,
) -> CellCIResult:
    """Compute patient-level bootstrap 95% CIs for the referable-DR binary endpoint.

    Thin wrapper over bootstrap_ci.compute_cell_ci. Converts P(dr_grade ≥ 2)
    to logit space before passing (AUROC/AUPRC rank-invariant under sigmoid).

    ZeroPositivesInResampleError from bootstrap_ci is NOT caught here; it
    propagates to the driver. The driver should treat this as
    BLOCKED_C1_BOOTSTRAP_ZERO_POSITIVE_FAILURE — do NOT silently return NA.

    Parameters
    ----------
    logit_dr_grade:
        Shape (N, 5). Raw per-class logits.
    label_dr_grade:
        Shape (N,). dr_grade labels (0–4 or -1.0 missing sentinel).
    mask_dr_grade:
        Shape (N,). float64 mask: 1.0 = observed, 0.0 = missing.
    patient_ids:
        Shape (N,). Patient identifier for each sample.
    cell_name:
        Identifier for the model cell (set on the returned result).
    n_resamples:
        Number of bootstrap resamples. Default 2000 (per A1 specification).
    seed:
        RNG seed. Default 42.

    Returns
    -------
    CellCIResult with cell_name set.

    Raises
    ------
    ValueError:
        If all samples are masked out or the valid set has zero referable positives.
    ZeroPositivesInResampleError:
        Propagated from bootstrap_ci if a resample draws zero positive patients.
    """
    n_valid = int((mask_dr_grade == 1.0).sum())
    if n_valid == 0:
        raise ValueError(
            f"[{cell_name}] compute_referable_dr_bootstrap_ci: "
            "all samples have mask=0; cannot compute metrics."
        )

    # Convert 5-class logits → P(grade ≥ 2)
    score = compute_referable_dr_score(logit_dr_grade)

    # Build binary label; set mask=0 for any missing-label samples
    ref_label = compute_referable_dr_label(label_dr_grade)
    ref_mask = mask_dr_grade.copy().astype(np.float64)
    ref_mask[ref_label == -1.0] = 0.0  # exclude missing-label samples

    # Replace -1 sentinel with 0.0 in the label array (these samples are masked out)
    safe_label = ref_label.copy()
    safe_label[safe_label == -1.0] = 0.0

    n_pos = int((safe_label[ref_mask == 1.0] == 1.0).sum())
    if n_pos == 0:
        raise ValueError(
            f"[{cell_name}] compute_referable_dr_bootstrap_ci: "
            "zero referable-positive samples in the valid set; cannot compute AUROC. "
            "Verify mask, labels, and referable_min_grade."
        )

    # Convert score → logit space (bootstrap_ci applies sigmoid internally)
    logit_score = _prob_to_logit(score)

    logger.info(
        "[%s] referable_dr: n_valid=%d, n_pos=%d, n_neg=%d",
        cell_name,
        int((ref_mask == 1.0).sum()),
        n_pos,
        int((ref_mask == 1.0).sum()) - n_pos,
    )

    result = compute_cell_ci(
        predictions={"referable_dr": logit_score},
        labels={"referable_dr": safe_label},
        masks={"referable_dr": ref_mask},
        patient_ids=patient_ids,
        task_metadata=REFERABLE_DR_TASK_METADATA,
        n_resamples=n_resamples,
        seed=seed,
    )
    result.cell_name = cell_name
    return result


def compute_referable_dr_pair_delta(
    logit_dr_grade_a: np.ndarray,
    label_dr_grade_a: np.ndarray,
    mask_dr_grade_a: np.ndarray,
    patient_ids_a: np.ndarray,
    logit_dr_grade_b: np.ndarray,
    label_dr_grade_b: np.ndarray,
    mask_dr_grade_b: np.ndarray,
    patient_ids_b: np.ndarray,
    cell_name_a: str,
    cell_name_b: str,
    n_resamples: int = 2000,
    seed: int = 42,
) -> DeltaCIResult:
    """Compute paired bootstrap delta CIs for referable-DR between two cells.

    Thin wrapper over bootstrap_ci.compute_paired_delta_ci. Both cells must share
    the same patient set (verified inside compute_paired_delta_ci).

    Delta = metric_A − metric_B.

    ZeroPositivesInResampleError propagates to the driver.

    Parameters
    ----------
    logit_dr_grade_a / _b:
        Shape (N, 5). Raw per-class logits for cells A and B.
    label_dr_grade_a / _b:
        Shape (N,). dr_grade labels.
    mask_dr_grade_a / _b:
        Shape (N,). float64 masks.
    patient_ids_a / _b:
        Shape (N,). Patient identifiers (must have identical sorted unique sets).
    cell_name_a / _b:
        Identifiers (set on the returned DeltaCIResult).
    n_resamples:
        Number of bootstrap resamples. Default 2000.
    seed:
        RNG seed. Default 42.

    Returns
    -------
    DeltaCIResult with cell_a and cell_b set.
    """
    def _prep(
        logit_5cls: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        score = compute_referable_dr_score(logit_5cls)
        ref_label = compute_referable_dr_label(label)
        ref_mask = mask.copy().astype(np.float64)
        ref_mask[ref_label == -1.0] = 0.0
        safe_label = ref_label.copy()
        safe_label[safe_label == -1.0] = 0.0
        logit_score = _prob_to_logit(score)
        return logit_score, safe_label, ref_mask

    logit_a, lbl_a, msk_a = _prep(logit_dr_grade_a, label_dr_grade_a, mask_dr_grade_a)
    logit_b, lbl_b, msk_b = _prep(logit_dr_grade_b, label_dr_grade_b, mask_dr_grade_b)

    result = compute_paired_delta_ci(
        predictions_a={"referable_dr": logit_a},
        labels_a={"referable_dr": lbl_a},
        masks_a={"referable_dr": msk_a},
        patient_ids_a=patient_ids_a,
        predictions_b={"referable_dr": logit_b},
        labels_b={"referable_dr": lbl_b},
        masks_b={"referable_dr": msk_b},
        patient_ids_b=patient_ids_b,
        task_metadata=REFERABLE_DR_TASK_METADATA,
        n_resamples=n_resamples,
        seed=seed,
    )
    result.cell_a = cell_name_a
    result.cell_b = cell_name_b
    return result
