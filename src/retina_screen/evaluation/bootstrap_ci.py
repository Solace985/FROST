from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)




class ZeroPositivesInResampleError(Exception):
    """Raised when a bootstrap resample has zero positive labels for a binary task.

    Parameters
    ----------
    task_name:
        Name of the task (passed in from driver/metadata; not used for logic).
    resample_idx:
        Index of the resample in which zero positives were encountered.
    """

    def __init__(self, task_name: str, resample_idx: int) -> None:
        self.task_name = task_name
        self.resample_idx = resample_idx
        super().__init__(
            f"Zero positive labels in resample {resample_idx} for task '{task_name}'."
        )




@dataclass
class TaskCIResult:
    """Bootstrap CI result for one (task, metric) pair within a single cell.

    Attributes
    ----------
    task_name:
        Name of the task (from driver metadata; not used for logic here).
    task_type:
        Type of task: "binary" or "ordinal".
    metric_name:
        Name of the metric: "auroc", "auprc", "balanced_accuracy", "macro_f1", "accuracy",
        or "binary_macro_auroc".
    point_estimate:
        Metric computed on the unresampled test set (GATE P reference value).
    ci_lo:
        Lower percentile CI bound (e.g. 2.5th percentile).
    ci_hi:
        Upper percentile CI bound (e.g. 97.5th percentile).
    n_included:
        Number of samples with mask == 1.0 (used in metric computation).
    n_total:
        Total number of samples.
    n_resamples_ok:
        Number of resamples that produced a valid metric value.
    n_resamples_skip:
        Number of resamples skipped (e.g. zero positives, single class).
    status:
        "ok" if CI was computed; "na" if metric could not be computed.
    reason:
        Empty string when status=ok; describes failure reason when status=na.
    """

    task_name: str
    task_type: str
    metric_name: str
    point_estimate: float | None
    ci_lo: float | None
    ci_hi: float | None
    n_included: int
    n_total: int
    n_resamples_ok: int
    n_resamples_skip: int
    status: str
    reason: str


@dataclass
class CellCIResult:
    """Bootstrap CI results for all tasks of a single cell.

    Attributes
    ----------
    cell_name:
        Identifier for the model cell (from driver; not used for logic here).
    tasks:
        List of per-(task, metric) CI results.
    n_resamples:
        Total number of bootstrap resamples requested.
    seed:
        RNG seed used.
    n_patients:
        Number of unique patients in the test set.
    n_samples:
        Total number of samples in the test set.
    """

    cell_name: str
    tasks: list[TaskCIResult]
    n_resamples: int
    seed: int
    n_patients: int
    n_samples: int


@dataclass
class TaskDeltaCIResult:
    """Paired bootstrap delta CI for one (task, metric) pair between two cells.

    Attributes
    ----------
    task_name:
        Task name (from driver metadata).
    metric_name:
        Metric name.
    delta_point:
        Point delta: metric_a(unresampled) - metric_b(unresampled).
    delta_ci_lo:
        Lower percentile CI bound on the delta distribution.
    delta_ci_hi:
        Upper percentile CI bound on the delta distribution.
    n_resamples_ok:
        Number of resamples that produced a valid delta.
    status:
        "supported" if CI excludes zero; "not_supported" if CI overlaps zero; "na" if
        metrics could not be computed for this pair.
    source:
        Always "paired_bootstrap" (DeLong not implemented in A1).
    """

    task_name: str
    metric_name: str
    delta_point: float | None
    delta_ci_lo: float | None
    delta_ci_hi: float | None
    n_resamples_ok: int
    status: str
    source: str


@dataclass
class DeltaCIResult:
    """Paired bootstrap delta CI results for all tasks between two cells.

    Attributes
    ----------
    cell_a:
        Name of cell A (from driver; not used for logic here).
    cell_b:
        Name of cell B (from driver; not used for logic here).
    tasks:
        List of per-(task, metric) delta CI results.
    n_resamples:
        Total number of bootstrap resamples requested.
    seed:
        RNG seed used.
    """

    cell_a: str
    cell_b: str
    tasks: list[TaskDeltaCIResult]
    n_resamples: int
    seed: int




def _compute_binary_auroc(
    logits: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> float | None:
    """Compute AUROC for a binary task after masking and sigmoid transform.

    Returns None if the masked subset has zero positives or is single-class.
    Raises ZeroPositivesInResampleError with task_name="" and resample_idx=-1 as
    a sentinel when n_positives == 0 so the caller can handle it appropriately.
    The caller must convert to the appropriate error type.
    """
    valid = masks == 1.0
    y_true = labels[valid]
    y_score = 1.0 / (1.0 + np.exp(-logits[valid].astype(np.float64)))
    n_pos = int(y_true.sum())
    if n_pos == 0 or int((y_true == 0).sum()) == 0:
        return None
    return float(roc_auc_score(y_true, y_score))


def _compute_binary_auprc(
    logits: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> float | None:
    """Compute AUPRC (average precision) for a binary task after masking."""
    valid = masks == 1.0
    y_true = labels[valid]
    y_score = 1.0 / (1.0 + np.exp(-logits[valid].astype(np.float64)))
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def _compute_ordinal_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float | None]:
    """Compute accuracy, macro_f1, balanced_accuracy for an ordinal task after masking.

    Returns a dict with keys "accuracy", "macro_f1", "balanced_accuracy".
    Any metric that cannot be computed (single-class subset) is None.
    """
    valid = masks == 1.0
    y_true = labels[valid].astype(int)
    y_pred = np.argmax(logits[valid], axis=-1)
    results: dict[str, float | None] = {}
    n = len(y_true)
    if n == 0:
        return {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    correct = int((y_true == y_pred).sum())
    results["accuracy"] = correct / n
    if len(np.unique(y_true)) < 2:
        results["macro_f1"] = None
        results["balanced_accuracy"] = None
    else:
        results["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    return results




def _build_patient_lookup(
    patient_ids: np.ndarray,
    unique_patients: np.ndarray,
) -> dict[Any, np.ndarray]:
    """Precompute patient_id → sample-index array mapping.

    Call once before the bootstrap loop.  Each resample then does O(n_patients)
    dict lookups instead of O(n_patients × n_samples) np.where comparisons.

    Parameters
    ----------
    patient_ids:
        Shape (n_samples,) — patient ID for each sample.
    unique_patients:
        Shape (n_patients,) — sorted unique patient IDs (from np.sort(np.unique(...))).

    Returns
    -------
    dict mapping each patient ID to a numpy int64 array of its sample indices.
    """
    return {pid: np.where(patient_ids == pid)[0].astype(np.int64) for pid in unique_patients}


def _build_patient_resample_indices(
    patient_ids: np.ndarray,
    unique_patients: np.ndarray,
    sampled_patient_idx: np.ndarray,
    lookup: dict[Any, np.ndarray] | None = None,
) -> np.ndarray:
    """Build sample indices for one bootstrap resample.

    Parameters
    ----------
    patient_ids:
        Shape (n_samples,) — patient ID for each sample.
    unique_patients:
        Shape (n_patients,) — sorted unique patient IDs.
    sampled_patient_idx:
        Shape (n_patients,) — indices into unique_patients for this resample.
    lookup:
        Optional precomputed patient → indices mapping (from _build_patient_lookup).
        When provided, each resample runs in O(n_patients) instead of
        O(n_patients × n_samples).  Defaults to None (falls back to np.where loop).

    Returns
    -------
    np.ndarray of sample indices, potentially longer than n_samples if patients
    are drawn multiple times.
    """
    if lookup is not None:
        parts = [lookup[unique_patients[pi]] for pi in sampled_patient_idx]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    indices: list[int] = []
    for pi in sampled_patient_idx:
        pid = unique_patients[pi]
        indices.extend(int(i) for i in np.where(patient_ids == pid)[0])
    return np.array(indices, dtype=np.int64)




def compute_cell_ci(
    predictions: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    patient_ids: np.ndarray,
    task_metadata: dict[str, dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 42,
    percentile_lo: float = 2.5,
    percentile_hi: float = 97.5,
) -> CellCIResult:
    """Compute bootstrap 95% CIs at the patient level for all tasks in one cell.

    Parameters
    ----------
    predictions:
        task_name → raw logit array.
        Binary tasks: shape (n_samples,).
        Ordinal tasks: shape (n_samples, n_classes).
    labels:
        task_name → float64 label array, shape (n_samples,).
    masks:
        task_name → float64 mask array, shape (n_samples,).
        1.0 = observed, 0.0 = missing.
    patient_ids:
        Shape (n_samples,) — patient identifier for each sample.
    task_metadata:
        task_name → {"task_type": "binary"|"ordinal", "n_classes": int (ordinal only),
                     "sparse": bool (optional, whether to compute AUPRC)}.
    n_resamples:
        Number of bootstrap resamples. Default 2000 per A1 specification.
    seed:
        RNG seed. Default 42.
    percentile_lo / percentile_hi:
        Percentile bounds for CIs. Default 2.5 / 97.5 (95% CI).

    Returns
    -------
    CellCIResult
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.sort(np.unique(patient_ids))
    n_patients = len(unique_patients)
    n_samples = len(patient_ids)

    task_metric_pairs: list[tuple[str, str]] = []
    for task_name, meta in task_metadata.items():
        ttype = meta["task_type"]
        if ttype == "binary":
            task_metric_pairs.append((task_name, "auroc"))
            if meta.get("sparse", False):
                task_metric_pairs.append((task_name, "auprc"))
        elif ttype == "ordinal":
            task_metric_pairs.append((task_name, "accuracy"))
            task_metric_pairs.append((task_name, "macro_f1"))
            task_metric_pairs.append((task_name, "balanced_accuracy"))

    binary_tasks = [t for t, m in task_metadata.items() if m["task_type"] == "binary"]
    has_macro = len(binary_tasks) > 0
    if has_macro:
        task_metric_pairs.append(("__macro__", "binary_macro_auroc"))

    distrib: dict[str, list[float]] = {
        f"{tn}__{mn}": [] for tn, mn in task_metric_pairs
    }
    skip_counts: dict[str, int] = {k: 0 for k in distrib}

    point_ests: dict[str, float | None] = {}
    for task_name, meta in task_metadata.items():
        ttype = meta["task_type"]
        if ttype == "binary":
            key = f"{task_name}__auroc"
            val = _compute_binary_auroc(
                predictions[task_name], labels[task_name], masks[task_name]
            )
            point_ests[key] = val
            if meta.get("sparse", False):
                key_ap = f"{task_name}__auprc"
                point_ests[key_ap] = _compute_binary_auprc(
                    predictions[task_name], labels[task_name], masks[task_name]
                )
        elif ttype == "ordinal":
            ord_metrics = _compute_ordinal_metrics(
                predictions[task_name], labels[task_name], masks[task_name]
            )
            for mn, v in ord_metrics.items():
                point_ests[f"{task_name}__{mn}"] = v

    if has_macro:
        macro_vals = [
            point_ests.get(f"{t}__auroc") for t in binary_tasks
        ]
        valid_macro = [v for v in macro_vals if v is not None]
        point_ests["__macro____binary_macro_auroc"] = (
            float(np.mean(valid_macro)) if valid_macro else None
        )

    patient_lookup = _build_patient_lookup(patient_ids, unique_patients)

    for r in range(n_resamples):
        sampled_idx = rng.choice(n_patients, size=n_patients, replace=True)
        indices = _build_patient_resample_indices(
            patient_ids, unique_patients, sampled_idx, lookup=patient_lookup
        )

        resample_aurocs: list[float] = []

        for task_name, meta in task_metadata.items():
            ttype = meta["task_type"]
            pred_r = predictions[task_name][indices]
            lbl_r = labels[task_name][indices]
            msk_r = masks[task_name][indices]

            if ttype == "binary":
                key = f"{task_name}__auroc"
                n_pos = int((lbl_r[msk_r == 1.0] == 1.0).sum())
                if n_pos == 0:
                    raise ZeroPositivesInResampleError(task_name, r)
                auroc_val = _compute_binary_auroc(pred_r, lbl_r, msk_r)
                if auroc_val is None:
                    skip_counts[key] += 1
                else:
                    distrib[key].append(auroc_val)
                    resample_aurocs.append(auroc_val)
                if meta.get("sparse", False):
                    key_ap = f"{task_name}__auprc"
                    ap_val = _compute_binary_auprc(pred_r, lbl_r, msk_r)
                    if ap_val is None:
                        skip_counts[key_ap] += 1
                    else:
                        distrib[key_ap].append(ap_val)

            elif ttype == "ordinal":
                ord_metrics = _compute_ordinal_metrics(pred_r, lbl_r, msk_r)
                for mn, v in ord_metrics.items():
                    key = f"{task_name}__{mn}"
                    if v is None:
                        skip_counts[key] += 1
                    else:
                        distrib[key].append(v)

        if has_macro:
            macro_key = "__macro____binary_macro_auroc"
            if len(resample_aurocs) == len(binary_tasks):
                distrib[macro_key].append(float(np.mean(resample_aurocs)))
            else:
                skip_counts[macro_key] += 1

    n_included: dict[str, int] = {
        task_name: int((masks[task_name] == 1.0).sum())
        for task_name in task_metadata
    }

    task_ci_results: list[TaskCIResult] = []
    for task_name, metric_name in task_metric_pairs:
        if task_name == "__macro__":
            key = "__macro____binary_macro_auroc"
            pt = point_ests.get("__macro____binary_macro_auroc")
            n_inc = n_samples
            t_type = "binary"
        else:
            key = f"{task_name}__{metric_name}"
            pt = point_ests.get(key)
            n_inc = n_included.get(task_name, n_samples)
            t_type = task_metadata[task_name]["task_type"]

        boot_vals = distrib[key]
        n_ok = len(boot_vals)
        n_skip = skip_counts[key]

        if n_ok < 2:
            task_ci_results.append(TaskCIResult(
                task_name=task_name if task_name != "__macro__" else "binary_macro",
                task_type=t_type,
                metric_name=metric_name,
                point_estimate=pt,
                ci_lo=None,
                ci_hi=None,
                n_included=n_inc,
                n_total=n_samples,
                n_resamples_ok=n_ok,
                n_resamples_skip=n_skip,
                status="na",
                reason="insufficient_bootstrap_resamples",
            ))
        else:
            ci_lo = float(np.percentile(boot_vals, percentile_lo))
            ci_hi = float(np.percentile(boot_vals, percentile_hi))
            task_ci_results.append(TaskCIResult(
                task_name=task_name if task_name != "__macro__" else "binary_macro",
                task_type=t_type,
                metric_name=metric_name,
                point_estimate=pt,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                n_included=n_inc,
                n_total=n_samples,
                n_resamples_ok=n_ok,
                n_resamples_skip=n_skip,
                status="ok",
                reason="",
            ))

    return CellCIResult(
        cell_name="",
        tasks=task_ci_results,
        n_resamples=n_resamples,
        seed=seed,
        n_patients=n_patients,
        n_samples=n_samples,
    )


def compute_paired_delta_ci(
    predictions_a: dict[str, np.ndarray],
    labels_a: dict[str, np.ndarray],
    masks_a: dict[str, np.ndarray],
    patient_ids_a: np.ndarray,
    predictions_b: dict[str, np.ndarray],
    labels_b: dict[str, np.ndarray],
    masks_b: dict[str, np.ndarray],
    patient_ids_b: np.ndarray,
    task_metadata: dict[str, dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 42,
    percentile_lo: float = 2.5,
    percentile_hi: float = 97.5,
) -> DeltaCIResult:
    """Compute paired bootstrap delta CIs between two cells.

    Both cells must share identical patient_ids (verified by caller / preflight).
    The same bootstrap resample indices are applied to both cells (paired inference).

    Parameters mirror those of compute_cell_ci for cells A and B.
    Delta is defined as: metric_A - metric_B.

    Returns
    -------
    DeltaCIResult with one TaskDeltaCIResult per (task, metric) pair.
    Delta status:
      - "supported"    : 95% CI of delta excludes zero (both bounds same sign).
      - "not_supported": 95% CI overlaps zero.
      - "na"           : delta could not be computed.
    """
    if not np.array_equal(np.sort(np.unique(patient_ids_a)),
                          np.sort(np.unique(patient_ids_b))):
        raise ValueError(
            "compute_paired_delta_ci: patient_ids_a and patient_ids_b must have "
            "identical patient sets for paired bootstrap."
        )

    rng = np.random.default_rng(seed)
    unique_patients = np.sort(np.unique(patient_ids_a))
    n_patients = len(unique_patients)
    n_samples = len(patient_ids_a)

    task_metric_pairs: list[tuple[str, str]] = []
    for task_name, meta in task_metadata.items():
        ttype = meta["task_type"]
        if ttype == "binary":
            task_metric_pairs.append((task_name, "auroc"))
            if meta.get("sparse", False):
                task_metric_pairs.append((task_name, "auprc"))
        elif ttype == "ordinal":
            task_metric_pairs.append((task_name, "accuracy"))
            task_metric_pairs.append((task_name, "macro_f1"))
            task_metric_pairs.append((task_name, "balanced_accuracy"))

    binary_tasks = [t for t, m in task_metadata.items() if m["task_type"] == "binary"]
    if binary_tasks:
        task_metric_pairs.append(("__macro__", "binary_macro_auroc"))

    delta_distrib: dict[str, list[float]] = {
        f"{tn}__{mn}": [] for tn, mn in task_metric_pairs
    }
    skip_counts: dict[str, int] = {k: 0 for k in delta_distrib}

    def _point_for_task_metric(
        preds: dict[str, np.ndarray],
        lbls: dict[str, np.ndarray],
        msks: dict[str, np.ndarray],
    ) -> dict[str, float | None]:
        pts: dict[str, float | None] = {}
        for tn, meta in task_metadata.items():
            ttype = meta["task_type"]
            if ttype == "binary":
                pts[f"{tn}__auroc"] = _compute_binary_auroc(preds[tn], lbls[tn], msks[tn])
                if meta.get("sparse", False):
                    pts[f"{tn}__auprc"] = _compute_binary_auprc(preds[tn], lbls[tn], msks[tn])
            elif ttype == "ordinal":
                ord_m = _compute_ordinal_metrics(preds[tn], lbls[tn], msks[tn])
                for mn, v in ord_m.items():
                    pts[f"{tn}__{mn}"] = v
        if binary_tasks:
            vals = [pts.get(f"{t}__auroc") for t in binary_tasks]
            valid_vals = [v for v in vals if v is not None]
            pts["__macro____binary_macro_auroc"] = (
                float(np.mean(valid_vals)) if valid_vals else None
            )
        return pts

    pt_a = _point_for_task_metric(predictions_a, labels_a, masks_a)
    pt_b = _point_for_task_metric(predictions_b, labels_b, masks_b)

    lookup_a = _build_patient_lookup(patient_ids_a, unique_patients)
    lookup_b = _build_patient_lookup(patient_ids_b, unique_patients)

    for r in range(n_resamples):
        sampled_idx = rng.choice(n_patients, size=n_patients, replace=True)
        idx_a = _build_patient_resample_indices(
            patient_ids_a, unique_patients, sampled_idx, lookup=lookup_a
        )
        idx_b = _build_patient_resample_indices(
            patient_ids_b, unique_patients, sampled_idx, lookup=lookup_b
        )

        macro_a_vals: list[float] = []
        macro_b_vals: list[float] = []

        for task_name, meta in task_metadata.items():
            ttype = meta["task_type"]
            pred_a_r = predictions_a[task_name][idx_a]
            lbl_a_r = labels_a[task_name][idx_a]
            msk_a_r = masks_a[task_name][idx_a]
            pred_b_r = predictions_b[task_name][idx_b]
            lbl_b_r = labels_b[task_name][idx_b]
            msk_b_r = masks_b[task_name][idx_b]

            if ttype == "binary":
                key = f"{task_name}__auroc"
                n_pos_a = int((lbl_a_r[msk_a_r == 1.0] == 1.0).sum())
                n_pos_b = int((lbl_b_r[msk_b_r == 1.0] == 1.0).sum())
                if n_pos_a == 0:
                    raise ZeroPositivesInResampleError(task_name, r)
                if n_pos_b == 0:
                    raise ZeroPositivesInResampleError(task_name, r)
                va = _compute_binary_auroc(pred_a_r, lbl_a_r, msk_a_r)
                vb = _compute_binary_auroc(pred_b_r, lbl_b_r, msk_b_r)
                if va is not None and vb is not None:
                    delta_distrib[key].append(va - vb)
                    macro_a_vals.append(va)
                    macro_b_vals.append(vb)
                else:
                    skip_counts[key] += 1

                if meta.get("sparse", False):
                    key_ap = f"{task_name}__auprc"
                    va_ap = _compute_binary_auprc(pred_a_r, lbl_a_r, msk_a_r)
                    vb_ap = _compute_binary_auprc(pred_b_r, lbl_b_r, msk_b_r)
                    if va_ap is not None and vb_ap is not None:
                        delta_distrib[key_ap].append(va_ap - vb_ap)
                    else:
                        skip_counts[key_ap] += 1

            elif ttype == "ordinal":
                ord_a = _compute_ordinal_metrics(pred_a_r, lbl_a_r, msk_a_r)
                ord_b = _compute_ordinal_metrics(pred_b_r, lbl_b_r, msk_b_r)
                for mn in ("accuracy", "macro_f1", "balanced_accuracy"):
                    key = f"{task_name}__{mn}"
                    va, vb = ord_a.get(mn), ord_b.get(mn)
                    if va is not None and vb is not None:
                        delta_distrib[key].append(va - vb)
                    else:
                        skip_counts[key] += 1

        if binary_tasks:
            macro_key = "__macro____binary_macro_auroc"
            if len(macro_a_vals) == len(binary_tasks) and len(macro_b_vals) == len(binary_tasks):
                delta_distrib[macro_key].append(
                    float(np.mean(macro_a_vals)) - float(np.mean(macro_b_vals))
                )
            else:
                skip_counts[macro_key] += 1

    delta_results: list[TaskDeltaCIResult] = []
    for task_name, metric_name in task_metric_pairs:
        if task_name == "__macro__":
            key = "__macro____binary_macro_auroc"
            pt_delta_a = pt_a.get("__macro____binary_macro_auroc")
            pt_delta_b = pt_b.get("__macro____binary_macro_auroc")
            display_task = "binary_macro"
        else:
            key = f"{task_name}__{metric_name}"
            pt_delta_a = pt_a.get(key)
            pt_delta_b = pt_b.get(key)
            display_task = task_name

        delta_vals = delta_distrib[key]
        n_ok = len(delta_vals)
        n_skip = skip_counts[key]

        if pt_delta_a is not None and pt_delta_b is not None:
            delta_pt = pt_delta_a - pt_delta_b
        else:
            delta_pt = None

        if n_ok < 2:
            delta_results.append(TaskDeltaCIResult(
                task_name=display_task,
                metric_name=metric_name,
                delta_point=delta_pt,
                delta_ci_lo=None,
                delta_ci_hi=None,
                n_resamples_ok=n_ok,
                status="na",
                source="paired_bootstrap",
            ))
        else:
            d_lo = float(np.percentile(delta_vals, percentile_lo))
            d_hi = float(np.percentile(delta_vals, percentile_hi))
            if d_lo > 0.0 or d_hi < 0.0:
                status = "supported"
            else:
                status = "not_supported"
            delta_results.append(TaskDeltaCIResult(
                task_name=display_task,
                metric_name=metric_name,
                delta_point=delta_pt,
                delta_ci_lo=d_lo,
                delta_ci_hi=d_hi,
                n_resamples_ok=n_ok,
                status=status,
                source="paired_bootstrap",
            ))

    return DeltaCIResult(
        cell_a="",
        cell_b="",
        tasks=delta_results,
        n_resamples=n_resamples,
        seed=seed,
    )
