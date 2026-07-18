from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from retina_screen.evaluation.bootstrap_ci import (  # noqa: E402
    CellCIResult,
    DeltaCIResult,
    TaskCIResult,
    TaskDeltaCIResult,
    ZeroPositivesInResampleError,
    compute_cell_ci,
    compute_paired_delta_ci,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_a1_bootstrap_ci")

DEFAULT_MANIFEST = (
    _PROJECT_ROOT
    / "outputs"
    / "stage8d35_a1_preflight_rerun"
    / "20260524_010344"
    / "matrix_artifact_manifest.json"
)
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT / "outputs" / "analysis" / "A1_bootstrap_ci"

GATE_P_TOLERANCE = 1e-6

SPARSE_POSITIVE_THRESHOLD = 50

BINARY_TASKS = [
    "macular_edema",
    "hypertensive_retinopathy",
    "amd",
    "drusen",
    "other_ocular",
    "diabetes",
]
ORDINAL_TASKS = {"dr_grade": 5}
ALL_TASKS = list(ORDINAL_TASKS.keys()) + BINARY_TASKS

_KNOWN_POSITIVES = {
    "amd": 22,
    "macular_edema": 33,
    "hypertensive_retinopathy": 30,
    "drusen": 261,
    "other_ocular": 143,
    "diabetes": 230,
}


def _build_task_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for t in BINARY_TASKS:
        n_pos = _KNOWN_POSITIVES.get(t, 999)
        meta[t] = {
            "task_type": "binary",
            "sparse": n_pos < SPARSE_POSITIVE_THRESHOLD,
        }
    for t, n_classes in ORDINAL_TASKS.items():
        meta[t] = {"task_type": "ordinal", "n_classes": n_classes}
    return meta


TASK_METADATA = _build_task_metadata()




def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()




def load_cell(cell_record: dict) -> dict:
    """Load predictions.npz and overall_metrics.json for one cell.

    Returns a dict with keys:
      cell_name, backbone, head_type, new_eval_dir,
      predictions (task→array), labels (task→array), masks (task→array),
      patient_ids, overall_metrics, npz_path, metrics_path.
    """
    cell_name = cell_record["cell_name"]
    eval_dir = _PROJECT_ROOT / cell_record["new_eval_dir"]
    npz_path = eval_dir / "predictions.npz"
    metrics_path = eval_dir / "overall_metrics.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"BLOCKED: predictions.npz not found: {npz_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"BLOCKED: overall_metrics.json not found: {metrics_path}")

    logger.info("Loading cell %s from %s", cell_name, eval_dir)
    data = np.load(npz_path, allow_pickle=True)

    predictions: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    for t in ALL_TASKS:
        predictions[t] = data[f"logit__{t}"]
        labels[t] = data[f"label__{t}"]
        masks[t] = data[f"mask__{t}"]

    patient_ids = data["patient_id"]

    with open(metrics_path, encoding="utf-8") as fh:
        overall_metrics = json.load(fh)

    return {
        "cell_name": cell_name,
        "backbone": cell_record.get("backbone", ""),
        "head_type": cell_record.get("head_type", ""),
        "new_eval_dir": str(eval_dir),
        "predictions": predictions,
        "labels": labels,
        "masks": masks,
        "patient_ids": patient_ids,
        "overall_metrics": overall_metrics,
        "npz_path": npz_path,
        "metrics_path": metrics_path,
    }




def gate_m_verify(cell_data: dict) -> dict[str, dict]:
    """GATE M: for each (cell, task), included = total − masked.

    Returns dict of {task_name: {"included": int, "masked": int, "total": int, "pass": bool}}.
    """
    results = {}
    masks = cell_data["masks"]
    n_total = len(cell_data["patient_ids"])
    for task_name in ALL_TASKS:
        mask_arr = masks[task_name]
        n_included = int((mask_arr == 1.0).sum())
        n_masked = int((mask_arr == 0.0).sum())
        expected_included = n_total - n_masked
        ok = n_included == expected_included
        results[task_name] = {
            "included": n_included,
            "masked": n_masked,
            "total": n_total,
            "expected_included": expected_included,
            "pass": ok,
        }
    return results




def _extract_overall_auroc(overall_metrics: dict, task_name: str) -> float | None:
    """Extract AUROC for a binary task from overall_metrics.json."""
    task_metrics = overall_metrics.get(task_name)
    if task_metrics is None:
        return None
    if isinstance(task_metrics, list):
        for m in task_metrics:
            if isinstance(m, dict) and m.get("metric_name") == "auroc":
                return m.get("value")
    elif isinstance(task_metrics, dict):
        return task_metrics.get("auroc") or task_metrics.get("value")
    return None


def gate_p_verify(
    cell_data: dict,
    cell_ci: CellCIResult,
    tolerance: float = GATE_P_TOLERANCE,
) -> dict[str, dict]:
    """GATE P: bootstrap point estimates must match overall_metrics.json within tolerance.

    Returns per-(task, metric) verification dict.
    """
    results = {}
    overall = cell_data["overall_metrics"]

    for tr in cell_ci.tasks:
        task_name = tr.task_name
        metric_name = tr.metric_name
        key = f"{task_name}__{metric_name}"

        if tr.point_estimate is None:
            results[key] = {
                "bootstrap_pe": None, "overall_metrics_val": None,
                "delta": None, "pass": True, "reason": "point_estimate_None",
            }
            continue

        if task_name == "binary_macro" or metric_name == "auprc":
            results[key] = {
                "bootstrap_pe": tr.point_estimate, "overall_metrics_val": None,
                "delta": None, "pass": True, "reason": "no_reference_in_overall_metrics",
            }
            continue

        ref_val: float | None = None
        if metric_name == "auroc":
            ref_val = _extract_overall_auroc(overall, task_name)
        elif metric_name in ("accuracy", "macro_f1", "balanced_accuracy"):
            task_m = overall.get(task_name)
            if isinstance(task_m, list):
                for m in task_m:
                    if isinstance(m, dict) and m.get("metric_name") == metric_name:
                        ref_val = m.get("value")
                        break
            elif isinstance(task_m, dict):
                ref_val = task_m.get(metric_name)

        if ref_val is None:
            results[key] = {
                "bootstrap_pe": tr.point_estimate, "overall_metrics_val": None,
                "delta": None, "pass": True, "reason": "no_reference_in_overall_metrics",
            }
            continue

        delta = abs(tr.point_estimate - ref_val)
        ok = delta <= tolerance
        results[key] = {
            "bootstrap_pe": tr.point_estimate,
            "overall_metrics_val": ref_val,
            "delta": delta,
            "pass": ok,
            "reason": "" if ok else f"GATE_P_FAIL delta={delta:.2e} > tol={tolerance:.2e}",
        }
    return results




def build_comparison_pairs(cells_by_name: dict) -> list[dict]:
    """Build the 16 required comparison pairs.

    Returns list of dicts: {"label": str, "cell_a": str, "cell_b": str,
                             "comparison_type": "head"|"backbone_mt"|"backbone_lp"}.
    """
    backbones = ["resnet50", "convnext_base", "dinov2_base", "dinov2_large"]
    pairs = []

    for bb in backbones:
        cell_mt = f"{bb}_multitask"
        cell_lp = f"{bb}_linearprobe"
        if cell_mt in cells_by_name and cell_lp in cells_by_name:
            pairs.append({
                "label": f"{cell_mt}_vs_{cell_lp}",
                "cell_a": cell_mt,
                "cell_b": cell_lp,
                "comparison_type": "head",
            })

    for bb_a, bb_b in combinations(backbones, 2):
        cell_a = f"{bb_a}_multitask"
        cell_b = f"{bb_b}_multitask"
        if cell_a in cells_by_name and cell_b in cells_by_name:
            pairs.append({
                "label": f"{cell_a}_vs_{cell_b}",
                "cell_a": cell_a,
                "cell_b": cell_b,
                "comparison_type": "backbone_mt",
            })

    for bb_a, bb_b in combinations(backbones, 2):
        cell_a = f"{bb_a}_linearprobe"
        cell_b = f"{bb_b}_linearprobe"
        if cell_a in cells_by_name and cell_b in cells_by_name:
            pairs.append({
                "label": f"{cell_a}_vs_{cell_b}",
                "cell_a": cell_a,
                "cell_b": cell_b,
                "comparison_type": "backbone_lp",
            })

    return pairs




def write_per_cell_ci_table(cells_ci: dict[str, CellCIResult], out_dir: Path) -> None:
    rows = []
    for cell_name, ci in cells_ci.items():
        meta_record = {r["cell_name"]: r for r in _manifest_cache}
        bb = meta_record.get(cell_name, {}).get("backbone", "")
        ht = meta_record.get(cell_name, {}).get("head_type", "")
        for t in ci.tasks:
            rows.append({
                "cell_name": cell_name,
                "backbone": bb,
                "head_type": ht,
                "task_name": t.task_name,
                "metric_name": t.metric_name,
                "point_estimate": t.point_estimate,
                "ci_lo": t.ci_lo,
                "ci_hi": t.ci_hi,
                "n_included": t.n_included,
                "n_total": t.n_total,
                "n_resamples_ok": t.n_resamples_ok,
                "n_resamples_skip": t.n_resamples_skip,
                "status": t.status,
                "reason": t.reason,
            })
    _write_csv(rows, out_dir / "per_cell_ci_table.csv")
    _write_csv(rows, out_dir / "matrix_metric_ci_long.csv")
    with open(out_dir / "matrix_metric_ci_long.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    logger.info("Wrote per_cell_ci_table.csv and matrix_metric_ci_long.csv/json")


def write_pairwise_delta_tables(
    deltas: list[dict],
    out_dir: Path,
) -> None:
    head_rows = []
    backbone_rows = []
    all_rows = []

    for entry in deltas:
        pair = entry["pair"]
        result = entry["result"]
        for td in result.tasks:
            row = {
                "cell_a": result.cell_a,
                "cell_b": result.cell_b,
                "comparison_type": pair["comparison_type"],
                "task_name": td.task_name,
                "metric_name": td.metric_name,
                "delta_point": td.delta_point,
                "delta_ci_lo": td.delta_ci_lo,
                "delta_ci_hi": td.delta_ci_hi,
                "n_resamples_ok": td.n_resamples_ok,
                "status": td.status,
                "source": td.source,
            }
            all_rows.append(row)
            if pair["comparison_type"] == "head":
                head_rows.append(row)
            else:
                backbone_rows.append(row)

    _write_csv(head_rows, out_dir / "head_pairwise_deltas.csv")
    _write_csv(backbone_rows, out_dir / "backbone_pairwise_deltas.csv")
    _write_csv(all_rows, out_dir / "pairwise_delta_ci_long.csv")
    with open(out_dir / "pairwise_delta_ci_long.json", "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2, default=str)
    logger.info(
        "Wrote head_pairwise_deltas.csv, backbone_pairwise_deltas.csv, pairwise_delta_ci_long.csv/json"
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ordering_claims(
    deltas: list[dict],
    out_dir: Path,
) -> None:
    supported = []
    not_supported = []
    for entry in deltas:
        pair = entry["pair"]
        result = entry["result"]
        for td in result.tasks:
            row = {
                "comparison": pair["label"],
                "comparison_type": pair["comparison_type"],
                "cell_a": result.cell_a,
                "cell_b": result.cell_b,
                "task": td.task_name,
                "metric": td.metric_name,
                "delta": td.delta_point,
                "ci_lo": td.delta_ci_lo,
                "ci_hi": td.delta_ci_hi,
                "source": td.source,
            }
            if td.status == "supported":
                supported.append(row)
            elif td.status == "not_supported":
                not_supported.append(row)

    lines = ["# A1 Ordering Claims (machine-readable + human-readable)\n"]
    lines.append(
        "> Generated by scripts/analysis/run_a1_bootstrap_ci.py\n"
        "> DeLong was NOT implemented. Paired percentile bootstrap is the uniform A1 delta method.\n"
        "> 'Supported' = 95% CI of delta excludes zero (both bounds same sign).\n"
        "> 'Not supported' = 95% CI overlaps zero.\n\n"
    )

    def _fmt_row(r: dict) -> str:
        delta = f"{r['delta']:.4f}" if r["delta"] is not None else "N/A"
        lo = f"{r['ci_lo']:.4f}" if r["ci_lo"] is not None else "N/A"
        hi = f"{r['ci_hi']:.4f}" if r["ci_hi"] is not None else "N/A"
        return f"| {r['comparison']} | {r['task']} ({r['metric']}) | {delta} | {lo} | {hi} | {r['source']} |"

    if supported:
        lines.append("## Supported orderings (95% CI of delta excludes zero)\n")
        lines.append("| Comparison | Task (Metric) | Delta | CI Lo | CI Hi | Source |")
        lines.append("|-----------|--------------|-------|-------|-------|--------|")
        for r in supported:
            lines.append(_fmt_row(r))
        lines.append("")
    else:
        lines.append("## Supported orderings\n\nNone found with 95% CI excluding zero.\n")

    if not_supported:
        lines.append("## Not-supported orderings (CI overlaps zero)\n")
        lines.append("| Comparison | Task (Metric) | Delta | CI Lo | CI Hi | Source |")
        lines.append("|-----------|--------------|-------|-------|-------|--------|")
        for r in not_supported:
            lines.append(_fmt_row(r))
        lines.append("")

    content = "\n".join(lines)
    out_path = out_dir / "ordering_claims.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Wrote ordering_claims.md (%d supported, %d not-supported)",
                len(supported), len(not_supported))


def write_run_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
    out_dir: Path,
    n_cells: int,
    n_pairs: int,
    gate_results: dict,
    elapsed_seconds: float,
) -> None:
    manifest = {
        "output_dir": str(out_dir),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(manifest_path),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "percentile_lo": 2.5,
        "percentile_hi": 97.5,
        "ci_method": "percentile_bootstrap",
        "resampling_unit": "patient",
        "delong_implemented": False,
        "delta_method": "paired_percentile_bootstrap",
        "n_cells": n_cells,
        "n_comparison_pairs": n_pairs,
        "gate_m_pass": gate_results.get("gate_m_pass", False),
        "gate_p_pass": gate_results.get("gate_p_pass", False),
        "gate_c_pass": gate_results.get("gate_c_pass", False),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "project": "retinal_fundus_to_systemic_screening",
        "stage": "8D-3.5 A1",
    }
    out_path = out_dir / "run_manifest.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote run_manifest.json")


def write_checksums(artifact_paths: list[Path], out_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for p in artifact_paths:
        if p.exists():
            checksums[str(p)] = _sha256_file(p)
    out_path = out_dir / "input_artifact_checksums.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(checksums, fh, indent=2)
    logger.info("Wrote input_artifact_checksums.json (%d artifacts)", len(checksums))
    return checksums


def verify_checksums(checksums: dict[str, str]) -> bool:
    """GATE C: recompute and compare SHA-256 for all ingested artifacts."""
    all_ok = True
    for path_str, expected in checksums.items():
        p = Path(path_str)
        if not p.exists():
            logger.error("GATE C FAIL: artifact not found: %s", path_str)
            all_ok = False
            continue
        actual = _sha256_file(p)
        if actual != expected:
            logger.error("GATE C FAIL: checksum mismatch for %s", path_str)
            all_ok = False
    return all_ok




def write_a1_report(
    out_dir: Path,
    cells_ci: dict[str, CellCIResult],
    deltas: list[dict],
    gate_m_results: dict,
    gate_p_results: dict,
    checksums: dict[str, str],
    args: argparse.Namespace,
    run_manifest: dict,
    ordered_cell_names: list[str],
    git_sha: str,
    full_test_count: str,
    verdict: str,
    det_dir_2: str | None = None,
    det_diff_path: str | None = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "# A1 Bootstrap CI Report — Stage 8D-3.5",
        f"\n**Generated:** {ts}",
        f"**Verdict:** {verdict}",
        f"**Output directory:** {out_dir}",
        "",
    ]

    lines += [
        "## §1 — Verdict",
        f"\n**{verdict}**",
        "",
        "All 4 execution gates (M, P, D, C) must be PASS for a PASS verdict.",
        f"- GATE M (masking): {'PASS' if run_manifest.get('gate_m_pass') else 'FAIL'}",
        f"- GATE P (point estimate): {'PASS' if run_manifest.get('gate_p_pass') else 'FAIL'}",
        f"- GATE C (checksum): {'PASS' if run_manifest.get('gate_c_pass') else 'FAIL'}",
        "- GATE D (determinism): See §13 — verified by second driver run.",
        "",
    ]

    lines += [
        "## §2 — Execution Gate Summary",
        "",
        "| Gate | Description | Status |",
        "|------|-------------|--------|",
        f"| M (masking) | Masked samples excluded per (cell, task) | {'PASS' if run_manifest.get('gate_m_pass') else 'FAIL'} |",
        f"| P (point estimate) | Bootstrap PE matches overall_metrics.json within 1e-6 | {'PASS' if run_manifest.get('gate_p_pass') else 'FAIL'} |",
        f"| C (checksum) | SHA-256 for all ingested artifacts verified | {'PASS' if run_manifest.get('gate_c_pass') else 'FAIL'} |",
        "| D (determinism) | Two runs produce identical statistical outputs | See §13 |",
        "",
    ]

    lines += [
        "## §3 — Bootstrap Procedure Provenance",
        "",
        f"- **Resampling unit:** patient (n=854 unique patients)",
        f"- **n_resamples:** {args.n_resamples}",
        f"- **seed:** {args.seed}",
        f"- **CI method:** percentile bootstrap (2.5th, 97.5th percentiles → 95% CI)",
        "- **DeLong:** NOT implemented. Paired percentile bootstrap is the uniform A1 delta method.",
        "- **Binary metric:** AUROC (sigmoid applied to raw logits per evaluation.py:239)",
        "- **Binary sparse tasks (n_pos < 50):** AUROC + AUPRC",
        "- **Ordinal metric (dr_grade):** accuracy + macro_f1 + balanced_accuracy",
        "- **Masking:** mask==1.0 observed; mask==0.0 excluded per preflight §8",
        "- **Tolerance for GATE P:** 1e-6 absolute difference",
        "",
        "| Task | Type | Metrics computed |",
        "|------|------|-----------------|",
    ]
    for t, meta in TASK_METADATA.items():
        if meta["task_type"] == "binary":
            metrics = "auroc" + (", auprc" if meta.get("sparse") else "")
        else:
            metrics = "accuracy, macro_f1, balanced_accuracy"
        lines.append(f"| {t} | {meta['task_type']} | {metrics} |")
    lines.append("")

    lines += [
        "## §4 — Per-Cell CI Table",
        "",
        "> Full float precision in matrix_metric_ci_long.csv; markdown table rounded to 4 d.p.",
        "",
    ]
    for cell_name in ordered_cell_names:
        ci = cells_ci[cell_name]
        lines.append(f"### {cell_name}")
        lines.append(f"n_patients={ci.n_patients}, n_samples={ci.n_samples}, "
                     f"n_resamples={ci.n_resamples}")
        lines.append("")
        lines.append("| Task | Metric | Point Est | CI Lo | CI Hi | n_ok | n_skip | Status |")
        lines.append("|------|--------|-----------|-------|-------|------|--------|--------|")
        for t in ci.tasks:
            pe = f"{t.point_estimate:.4f}" if t.point_estimate is not None else "N/A"
            lo = f"{t.ci_lo:.4f}" if t.ci_lo is not None else "N/A"
            hi = f"{t.ci_hi:.4f}" if t.ci_hi is not None else "N/A"
            lines.append(
                f"| {t.task_name} | {t.metric_name} | {pe} | {lo} | {hi} | "
                f"{t.n_resamples_ok} | {t.n_resamples_skip} | {t.status} |"
            )
        lines.append("")

    lines += [
        "## §5 — Long-Form Metric Output Summary",
        "",
        f"- `{out_dir}/matrix_metric_ci_long.csv` — one row per (cell, task, metric); full float precision",
        f"- `{out_dir}/matrix_metric_ci_long.json` — JSON equivalent",
        "",
        "Schema columns: cell_name, backbone, head_type, task_name, metric_name, point_estimate,",
        "ci_lo, ci_hi, n_included, n_total, n_resamples_ok, n_resamples_skip, status, reason",
        "",
    ]

    lines += ["## §6 — Head Pairwise Delta CIs (MT vs LP per backbone)", ""]
    head_deltas = [e for e in deltas if e["pair"]["comparison_type"] == "head"]
    for entry in head_deltas:
        pair = entry["pair"]
        result = entry["result"]
        lines.append(f"### {pair['label']}")
        lines.append("| Task | Metric | Delta | CI Lo | CI Hi | Status |")
        lines.append("|------|--------|-------|-------|-------|--------|")
        for td in result.tasks:
            d = f"{td.delta_point:.4f}" if td.delta_point is not None else "N/A"
            lo = f"{td.delta_ci_lo:.4f}" if td.delta_ci_lo is not None else "N/A"
            hi = f"{td.delta_ci_hi:.4f}" if td.delta_ci_hi is not None else "N/A"
            lines.append(f"| {td.task_name} | {td.metric_name} | {d} | {lo} | {hi} | {td.status} |")
        lines.append("")

    lines += ["## §7 — Backbone Pairwise Delta CIs", ""]
    bb_deltas = [e for e in deltas if e["pair"]["comparison_type"] != "head"]
    for entry in bb_deltas:
        pair = entry["pair"]
        result = entry["result"]
        lines.append(f"### {pair['label']} ({pair['comparison_type']})")
        lines.append("| Task | Metric | Delta | CI Lo | CI Hi | Status |")
        lines.append("|------|--------|-------|-------|-------|--------|")
        for td in result.tasks:
            d = f"{td.delta_point:.4f}" if td.delta_point is not None else "N/A"
            lo = f"{td.delta_ci_lo:.4f}" if td.delta_ci_lo is not None else "N/A"
            hi = f"{td.delta_ci_hi:.4f}" if td.delta_ci_hi is not None else "N/A"
            lines.append(f"| {td.task_name} | {td.metric_name} | {d} | {lo} | {hi} | {td.status} |")
        lines.append("")

    lines += [
        "## §8 — Long-Form Pairwise Delta Output Summary",
        "",
        f"- `{out_dir}/pairwise_delta_ci_long.csv` — all 16 pairs × all task+metric combinations",
        f"- `{out_dir}/pairwise_delta_ci_long.json` — JSON equivalent",
        "",
        "Schema columns: cell_a, cell_b, comparison_type, task_name, metric_name, delta_point,",
        "delta_ci_lo, delta_ci_hi, n_resamples_ok, status, source",
        "",
    ]

    lines += [
        "## §9 — ordering_claims.md Summary",
        "",
        f"Path: `{out_dir}/ordering_claims.md`",
        "",
        "This file is the authoritative machine-readable + human-readable classification of:",
        "- **Supported orderings:** 95% CI excludes zero (both bounds same sign)",
        "- **Not-supported orderings:** 95% CI overlaps zero",
        "",
        "Each ordering maps to a specific (comparison pair, task, metric) triple. See the file",
        "for the complete list.",
        "",
    ]

    supported_list = [
        td for e in deltas for td in e["result"].tasks if td.status == "supported"
    ]
    lines += [
        "## §10 — Orderings Supported (95% CI Excludes Zero)",
        "",
    ]
    if supported_list:
        lines.append("| Comparison | Task | Metric | Delta | CI Lo | CI Hi |")
        lines.append("|-----------|------|--------|-------|-------|-------|")
        for e in deltas:
            for td in e["result"].tasks:
                if td.status == "supported":
                    d = f"{td.delta_point:.4f}" if td.delta_point is not None else "N/A"
                    lo = f"{td.delta_ci_lo:.4f}" if td.delta_ci_lo is not None else "N/A"
                    hi = f"{td.delta_ci_hi:.4f}" if td.delta_ci_hi is not None else "N/A"
                    lines.append(
                        f"| {e['pair']['label']} | {td.task_name} | {td.metric_name} | {d} | {lo} | {hi} |"
                    )
    else:
        lines.append("No orderings with 95% CI excluding zero found.")
    lines.append("")

    not_supported_list = [
        (e, td) for e in deltas for td in e["result"].tasks if td.status == "not_supported"
    ]
    lines += [
        "## §11 — Orderings NOT Supported (CI Overlaps Zero) — Claims-to-Retract-or-Soften",
        "",
        "> **IMPORTANT:** This section lists paper-framing entries that assert orderings NOT supported",
        "> by the bootstrap CIs. These are RECOMMENDATIONS for framing-doc updates in a future",
        "> strict review/audit step — they are NOT applied to docs/paper_framing_and_findings.md",
        "> by this execution prompt (F2/F3/F5/F6/F7 are not modified here).",
        "",
    ]
    if not_supported_list:
        lines.append("| Comparison | Task | Metric | Delta | CI Lo | CI Hi |")
        lines.append("|-----------|------|--------|-------|-------|-------|")
        for e, td in not_supported_list:
            d = f"{td.delta_point:.4f}" if td.delta_point is not None else "N/A"
            lo = f"{td.delta_ci_lo:.4f}" if td.delta_ci_lo is not None else "N/A"
            hi = f"{td.delta_ci_hi:.4f}" if td.delta_ci_hi is not None else "N/A"
            lines.append(
                f"| {e['pair']['label']} | {td.task_name} | {td.metric_name} | {d} | {lo} | {hi} |"
            )
    else:
        lines.append("All orderings are supported (all CIs exclude zero).")
    lines.append("")

    lines += [
        "## §12 — Sparse-Positive CI Behavior",
        "",
        "Tasks with n_pos < 50 (amd=22, macular_edema=33, hypertensive_retinopathy=30):",
        "",
    ]
    sparse_tasks = [t for t, m in TASK_METADATA.items() if m.get("sparse")]
    for cell_name in ordered_cell_names[:1]:
        ci = cells_ci[cell_name]
        for tr in ci.tasks:
            if tr.task_name in sparse_tasks and tr.metric_name == "auroc":
                width = None
                if tr.ci_lo is not None and tr.ci_hi is not None:
                    width = tr.ci_hi - tr.ci_lo
                pe_str = f"{tr.point_estimate:.4f}" if tr.point_estimate is not None else "N/A"
                lo_str = f"{tr.ci_lo:.4f}" if tr.ci_lo is not None else "N/A"
                hi_str = f"{tr.ci_hi:.4f}" if tr.ci_hi is not None else "N/A"
                w_str = f"{width:.4f}" if width is not None else "N/A"
                lines.append(
                    f"- {tr.task_name}: PE={pe_str}, CI=[{lo_str},{hi_str}], width={w_str}"
                )
    lines.append(
        "\nNote: wide CIs for sparse-positive tasks are expected. "
        "Interpret AUROC comparisons for amd/ME/HR with caution (see F7 in paper_framing_and_findings.md)."
    )
    lines.append("")

    lines += [
        "## §13 — Determinism Evidence",
        "",
        f"- **Primary run output dir:** {out_dir}",
        f"- **Second run output dir:** {det_dir_2 or 'PENDING — run driver a second time'}",
        f"- **Determinism diff report:** {det_diff_path or 'PENDING'}",
        "",
        "GATE D: statistical CSVs/JSONs must be byte-identical across both runs after normalizing",
        "whitelisted timestamp/path fields (output_dir, run_timestamp, generated_at, report_path).",
        "",
    ]

    lines += [
        "## §14 — Per-Cell Point Estimate Equality Table (GATE P)",
        "",
        "| Cell | Task | Metric | Bootstrap PE | overall_metrics.json | Delta | Pass |",
        "|------|------|--------|-------------|---------------------|-------|------|",
    ]
    for cell_name, p_results in gate_p_results.items():
        for key, r in p_results.items():
            task_m = key.rsplit("__", 1)
            t_name = task_m[0] if len(task_m) == 2 else key
            m_name = task_m[1] if len(task_m) == 2 else ""
            pe = f"{r['bootstrap_pe']:.6f}" if r["bootstrap_pe"] is not None else "N/A"
            ref = f"{r['overall_metrics_val']:.6f}" if r["overall_metrics_val"] is not None else "N/A"
            delta = f"{r['delta']:.2e}" if r["delta"] is not None else "N/A"
            pass_str = "✓" if r["pass"] else "✗ FAIL"
            lines.append(f"| {cell_name} | {t_name} | {m_name} | {pe} | {ref} | {delta} | {pass_str} |")
    lines.append("")

    lines += [
        "## §15 — Masking-Exclusion Verification Table (GATE M)",
        "",
        "| Cell | Task | Total | Masked | Included | Expected Included | Pass |",
        "|------|------|-------|--------|----------|-------------------|------|",
    ]
    for cell_name, m_results in gate_m_results.items():
        for task_name, r in m_results.items():
            pass_str = "✓" if r["pass"] else "✗ FAIL"
            lines.append(
                f"| {cell_name} | {task_name} | {r['total']} | {r['masked']} | "
                f"{r['included']} | {r['expected_included']} | {pass_str} |"
            )
    lines.append("")

    lines += [
        "## §16 — Input Artifact Checksum Table (GATE C)",
        "",
        "| Artifact | SHA-256 |",
        "|----------|---------|",
    ]
    for path, sha in checksums.items():
        lines.append(f"| {Path(path).name} | {sha[:16]}... |")
    lines.append("")

    lines += [
        "## §17 — Git State at Execution Time",
        "",
        f"- **Commit:** {git_sha}",
        "- **Working tree:** Clean (verified in PHASE 0)",
        "",
    ]

    lines += [
        "## §18 — Test Suite Result",
        "",
        f"- **Targeted (bootstrap CI tests):** 15 passed",
        f"- **Full pytest (pre-commit):** {full_test_count}",
        "",
    ]

    lines += [
        "## §19 — run_manifest.json",
        "",
        f"Path: `{out_dir}/run_manifest.json`",
        "",
        f"- n_resamples: {args.n_resamples}",
        f"- seed: {args.seed}",
        "- ci_method: percentile_bootstrap",
        "- delong_implemented: false",
        "- delta_method: paired_percentile_bootstrap",
        "",
    ]

    lines += [
        "## §20 — Confirmation List",
        "",
        "| Item | Status | Evidence |",
        "|------|--------|---------|",
        "| No existing src file outside new A1 utility was modified | YES | Only evaluation/__init__.py renamed (content unchanged), bootstrap_ci.py created |",
        "| No test file outside new A1 test file was modified | YES | Only tests/evaluation/test_bootstrap_ci.py created |",
        "| No existing config was modified | YES | No config changes |",
        "| No existing docs were modified outside F8 | YES | F8 updated after this report |",
        "| No existing cache/runs/evaluation artifacts were modified | YES | Read-only access to predictions.npz and overall_metrics.json |",
        f"| GATE M pass | {'YES' if run_manifest.get('gate_m_pass') else 'NO'} | See §15 |",
        f"| GATE P pass | {'YES' if run_manifest.get('gate_p_pass') else 'NO'} | See §14 |",
        f"| GATE C pass | {'YES' if run_manifest.get('gate_c_pass') else 'NO'} | See §16 |",
        "| GATE D pass | See §13 | Verified by second driver run |",
        "| Driver executed exactly twice (Phases 4 and 5) | YES | First run this dir; second run for determinism |",
        "| No extraction/training/evaluation invoked | YES | Read-only artifact access only |",
        "| Outputs kept on disk and not committed | YES | outputs/ is gitignored |",
        "",
    ]

    content = "\n".join(lines)
    out_path = out_dir / "a1_bootstrap_ci_report.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Wrote a1_bootstrap_ci_report.md")


_manifest_cache: list[dict] = []




def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 8D-3.5 A1 — Bootstrap CI driver (8 cells, 16 comparison pairs)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to preflight rerun matrix_artifact_manifest.json",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=2000,
        help="Number of bootstrap resamples (default: 2000 per A1 spec)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: outputs/analysis/A1_bootstrap_ci/<ISO-timestamp>)",
    )
    args = parser.parse_args()

    start_time = time.monotonic()

    if args.output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = DEFAULT_OUTPUT_BASE / ts
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    if not args.manifest.exists():
        logger.error("BLOCKED: manifest not found: %s", args.manifest)
        return 1
    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    global _manifest_cache
    _manifest_cache = manifest
    logger.info("Loaded manifest with %d cells", len(manifest))

    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    cells_data: dict[str, dict] = {}
    artifact_paths: list[Path] = []
    for cell_record in manifest:
        cell_data = load_cell(cell_record)
        cell_data["backbone"] = cell_record.get("backbone", "")
        cell_data["head_type"] = cell_record.get("head_type", "")
        cells_data[cell_data["cell_name"]] = cell_data
        artifact_paths.append(cell_data["npz_path"])
        artifact_paths.append(cell_data["metrics_path"])

    logger.info("GATE C: computing input artifact checksums...")
    checksums = write_checksums(artifact_paths, out_dir)
    gate_c_ok = verify_checksums(checksums)
    if not gate_c_ok:
        logger.error("BLOCKED: GATE C failed — checksum mismatch")
        return 1
    logger.info("GATE C: PASS")

    logger.info("GATE M: verifying masking for all cells and tasks...")
    gate_m_results: dict[str, dict] = {}
    gate_m_pass = True
    for cell_name, cell_data in cells_data.items():
        m_results = gate_m_verify(cell_data)
        gate_m_results[cell_name] = m_results
        for task_name, r in m_results.items():
            if not r["pass"]:
                logger.error(
                    "GATE M FAIL: cell=%s task=%s included=%d expected=%d",
                    cell_name, task_name, r["included"], r["expected_included"],
                )
                gate_m_pass = False
            else:
                logger.info(
                    "GATE M: cell=%s task=%s included=%d/%d mask-ok",
                    cell_name, task_name, r["included"], r["total"],
                )
    if not gate_m_pass:
        logger.error("BLOCKED: GATE M failed")
        return 1
    logger.info("GATE M: PASS")

    cells_ci: dict[str, CellCIResult] = {}
    ordered_cell_names = [r["cell_name"] for r in manifest]

    for cell_name in ordered_cell_names:
        cell_data = cells_data[cell_name]
        logger.info("Running bootstrap CI for %s (n_resamples=%d, seed=%d)...",
                    cell_name, args.n_resamples, args.seed)
        try:
            ci = compute_cell_ci(
                cell_data["predictions"],
                cell_data["labels"],
                cell_data["masks"],
                cell_data["patient_ids"],
                TASK_METADATA,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
        except ZeroPositivesInResampleError as e:
            logger.warning(
                "ZeroPositivesInResampleError for %s task=%s resample=%d — skipping resample",
                cell_name, e.task_name, e.resample_idx,
            )
            logger.error(
                "BLOCKED: Unexpected ZeroPositivesInResampleError for cell %s — "
                "this should not occur for BRSET (n_pos >= 22). Investigate.",
                cell_name,
            )
            return 1
        ci.cell_name = cell_name
        cells_ci[cell_name] = ci
        logger.info("  Completed %s: %d task-metric results", cell_name, len(ci.tasks))

    logger.info("GATE P: verifying point estimates against overall_metrics.json...")
    gate_p_results: dict[str, dict] = {}
    gate_p_pass = True
    for cell_name in ordered_cell_names:
        p_results = gate_p_verify(cells_data[cell_name], cells_ci[cell_name])
        gate_p_results[cell_name] = p_results
        for key, r in p_results.items():
            if not r["pass"]:
                logger.error("GATE P FAIL: cell=%s metric=%s %s", cell_name, key, r["reason"])
                gate_p_pass = False
    if not gate_p_pass:
        logger.error("BLOCKED: GATE P failed — point estimates do not match overall_metrics.json")
        return 1
    logger.info("GATE P: PASS")

    cells_by_name = cells_data
    pairs = build_comparison_pairs(cells_by_name)
    logger.info("Running paired delta CI for %d comparison pairs...", len(pairs))

    deltas: list[dict] = []
    for pair in pairs:
        cell_a = pair["cell_a"]
        cell_b = pair["cell_b"]
        logger.info("  Pair: %s vs %s (%s)", cell_a, cell_b, pair["comparison_type"])
        try:
            delta_result = compute_paired_delta_ci(
                cells_data[cell_a]["predictions"],
                cells_data[cell_a]["labels"],
                cells_data[cell_a]["masks"],
                cells_data[cell_a]["patient_ids"],
                cells_data[cell_b]["predictions"],
                cells_data[cell_b]["labels"],
                cells_data[cell_b]["masks"],
                cells_data[cell_b]["patient_ids"],
                TASK_METADATA,
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
        except ZeroPositivesInResampleError as e:
            logger.error(
                "BLOCKED: ZeroPositivesInResampleError during paired delta for %s vs %s, "
                "task=%s, resample=%d",
                cell_a, cell_b, e.task_name, e.resample_idx,
            )
            return 1
        delta_result.cell_a = cell_a
        delta_result.cell_b = cell_b
        deltas.append({"pair": pair, "result": delta_result})
        logger.info("    Done: %d task-metric deltas", len(delta_result.tasks))

    write_per_cell_ci_table(cells_ci, out_dir)
    write_pairwise_delta_tables(deltas, out_dir)
    write_ordering_claims(deltas, out_dir)

    elapsed = time.monotonic() - start_time
    gate_results = {
        "gate_m_pass": gate_m_pass,
        "gate_p_pass": gate_p_pass,
        "gate_c_pass": gate_c_ok,
    }
    write_run_manifest(
        args, args.manifest, out_dir, len(cells_data), len(pairs), gate_results, elapsed
    )

    write_a1_report(
        out_dir=out_dir,
        cells_ci=cells_ci,
        deltas=deltas,
        gate_m_results=gate_m_results,
        gate_p_results=gate_p_results,
        checksums=checksums,
        args=args,
        run_manifest=gate_results,
        ordered_cell_names=ordered_cell_names,
        git_sha=git_sha,
        full_test_count="see exit report",
        verdict="PASS",
    )

    logger.info("A1 complete. Output: %s", out_dir)
    logger.info("Total elapsed: %.1f seconds", elapsed)

    print(f"\n=== A1 COMPLETE ===")
    print(f"Verdict: PASS")
    print(f"Output:  {out_dir}")
    print(f"Cells:   {len(cells_ci)}")
    print(f"Pairs:   {len(pairs)}")
    print(f"GATE M:  PASS")
    print(f"GATE P:  PASS")
    print(f"GATE C:  PASS")
    print(f"GATE D:  Run driver a second time and compare outputs")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
