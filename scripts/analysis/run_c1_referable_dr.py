from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from retina_screen.evaluation.referable_dr import (  # noqa: E402
    ZeroPositivesInResampleError,
    compute_referable_dr_bootstrap_ci,
    compute_referable_dr_pair_delta,
    compute_referable_dr_score,
    compute_referable_dr_label,
    REFERABLE_DR_TASK_METADATA,
)


REQUIRED_OUTPUT_FILES: list[str] = [
    "c1_referable_dr_report.md",
    "c1_referable_dr_manifest.json",
    "referable_dr_cell_metrics.csv",
    "referable_dr_cell_metrics.json",
    "referable_dr_pairwise_deltas.csv",
    "referable_dr_pairwise_deltas.json",
    "dr_grade_5class_limitation_table.csv",
    "dr_grade_5class_limitation_table.json",
    "input_artifact_checksums.json",
    "c1_results.csv",
    "c1_results.json",
    "c1_pair_deltas.csv",
    "c1_report.md",
]

assert len(REQUIRED_OUTPUT_FILES) == 13, "REQUIRED_OUTPUT_FILES must have exactly 13 entries"


CELL_REGISTRY: dict[str, dict[str, str]] = {
    "c01": {
        "backbone": "resnet50",
        "head": "mt",
        "protocol": "main_matrix",
        "display": "ResNet-50 MT",
        "eval_dir": "outputs/evaluation/20260524_000040",
    },
    "c02": {
        "backbone": "resnet50",
        "head": "lp",
        "protocol": "main_matrix",
        "display": "ResNet-50 LP",
        "eval_dir": "outputs/evaluation/20260524_000201",
    },
    "c03": {
        "backbone": "convnext_base",
        "head": "mt",
        "protocol": "main_matrix",
        "display": "ConvNeXt-Base MT",
        "eval_dir": "outputs/evaluation/20260524_000334",
    },
    "c04": {
        "backbone": "convnext_base",
        "head": "lp",
        "protocol": "main_matrix",
        "display": "ConvNeXt-Base LP",
        "eval_dir": "outputs/evaluation/20260524_000354",
    },
    "c05": {
        "backbone": "dinov2_base",
        "head": "mt",
        "protocol": "main_matrix",
        "display": "DINOv2-Base MT",
        "eval_dir": "outputs/evaluation/20260524_000512",
    },
    "c06": {
        "backbone": "dinov2_base",
        "head": "lp",
        "protocol": "main_matrix",
        "display": "DINOv2-Base LP",
        "eval_dir": "outputs/evaluation/20260524_000532",
    },
    "c07": {
        "backbone": "dinov2_large",
        "head": "mt",
        "protocol": "main_matrix",
        "display": "DINOv2-Large MT",
        "eval_dir": "outputs/evaluation/20260524_000636",
    },
    "c08": {
        "backbone": "dinov2_large",
        "head": "lp",
        "protocol": "main_matrix",
        "display": "DINOv2-Large LP",
        "eval_dir": "outputs/evaluation/20260524_000654",
    },
    "c09": {
        "backbone": "retfound_green_matched224",
        "head": "mt",
        "protocol": "main_matrix",
        "display": "RETFound-Green matched-224 MT",
        "eval_dir": "outputs/evaluation/20260525_035535",
    },
    "c10": {
        "backbone": "retfound_green_matched224",
        "head": "lp",
        "protocol": "main_matrix",
        "display": "RETFound-Green matched-224 LP",
        "eval_dir": "outputs/evaluation/20260525_035808",
    },
    "c11": {
        "backbone": "dinov3_large",
        "head": "mt",
        "protocol": "main_matrix",
        "display": "DINOv3-Large MT",
        "eval_dir": "outputs/evaluation/20260527_091715",
    },
    "c12": {
        "backbone": "dinov3_large",
        "head": "lp",
        "protocol": "main_matrix",
        "display": "DINOv3-Large LP",
        "eval_dir": "outputs/evaluation/20260527_091902",
    },
    "c13": {
        "backbone": "retfound_green_native392",
        "head": "mt",
        "protocol": "off_protocol_comparator",
        "display": "RETFound-Green native-392 MT",
        "eval_dir": "outputs/evaluation/20260525_113924",
    },
    "c14": {
        "backbone": "retfound_green_native392",
        "head": "lp",
        "protocol": "off_protocol_comparator",
        "display": "RETFound-Green native-392 LP",
        "eval_dir": "outputs/evaluation/20260525_114150",
    },
}


PAIRWISE_COMPARISONS: list[dict[str, str]] = [
    {"group": "A", "label": "ResNet-50 MT vs LP",               "cell_a": "c01", "cell_b": "c02"},
    {"group": "A", "label": "ConvNeXt-Base MT vs LP",           "cell_a": "c03", "cell_b": "c04"},
    {"group": "A", "label": "DINOv2-Base MT vs LP",             "cell_a": "c05", "cell_b": "c06"},
    {"group": "A", "label": "DINOv2-Large MT vs LP",            "cell_a": "c07", "cell_b": "c08"},
    {"group": "A", "label": "RETFound matched-224 MT vs LP",    "cell_a": "c09", "cell_b": "c10"},
    {"group": "A", "label": "DINOv3-Large MT vs LP",            "cell_a": "c11", "cell_b": "c12"},
    {"group": "A", "label": "RETFound native-392 MT vs LP",     "cell_a": "c13", "cell_b": "c14"},
    {"group": "B", "label": "DINOv3-Large MT vs DINOv2-Large MT",  "cell_a": "c11", "cell_b": "c07"},
    {"group": "B", "label": "DINOv3-Large MT vs DINOv2-Base MT",   "cell_a": "c11", "cell_b": "c05"},
    {"group": "B", "label": "DINOv2-Large MT vs DINOv2-Base MT",   "cell_a": "c07", "cell_b": "c05"},
    {"group": "B", "label": "RETFound matched-224 MT vs DINOv2-Large MT", "cell_a": "c09", "cell_b": "c07"},
    {"group": "B", "label": "RETFound matched-224 MT vs DINOv3-Large MT", "cell_a": "c09", "cell_b": "c11"},
    {"group": "C", "label": "DINOv3-Large LP vs DINOv2-Large LP",  "cell_a": "c12", "cell_b": "c08"},
    {"group": "C", "label": "DINOv3-Large LP vs DINOv2-Base LP",   "cell_a": "c12", "cell_b": "c06"},
    {"group": "C", "label": "DINOv2-Large LP vs DINOv2-Base LP",   "cell_a": "c08", "cell_b": "c06"},
    {"group": "D", "label": "RETFound native-392 MT vs matched-224 MT", "cell_a": "c13", "cell_b": "c09"},
    {"group": "D", "label": "RETFound native-392 LP vs matched-224 LP", "cell_a": "c14", "cell_b": "c10"},
    {"group": "E", "label": "RETFound native-392 MT vs DINOv2-Large MT",  "cell_a": "c13", "cell_b": "c07"},
    {"group": "E", "label": "RETFound native-392 MT vs DINOv3-Large MT",  "cell_a": "c13", "cell_b": "c11"},
    {"group": "E", "label": "RETFound native-392 MT vs DINOv2-Base MT",   "cell_a": "c13", "cell_b": "c05"},
]

assert len(PAIRWISE_COMPARISONS) == 20, "PAIRWISE_COMPARISONS must have exactly 20 entries"

SPOT_CHECK_CELLS = ["c01", "c07", "c11"]




def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cell(cell_id: str) -> dict[str, Any]:
    """Load predictions.npz for a cell. Returns dict with arrays."""
    meta = CELL_REGISTRY[cell_id]
    npz_path = Path(meta["eval_dir"]) / "predictions.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "logit_dr_grade": data["logit__dr_grade"],
        "label_dr_grade": data["label__dr_grade"],
        "mask_dr_grade":  data["mask__dr_grade"],
        "patient_id":     data["patient_id"],
        "sample_id":      data["sample_id"],
    }


def _load_5class_metrics(cell_id: str) -> list[dict[str, Any]]:
    """Load dr_grade 5-class metrics from overall_metrics.json."""
    meta = CELL_REGISTRY[cell_id]
    metrics_path = Path(meta["eval_dir"]) / "overall_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)
    return all_metrics.get("dr_grade", [])


def _fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "na"


def _cell_result_to_row(
    cell_id: str,
    cell_meta: dict[str, str],
    ci_result: Any,
    n_pos: int,
) -> dict[str, Any]:
    """Extract AUROC and AUPRC from CellCIResult → flat dict for CSV/JSON."""
    auroc_task = next(
        (t for t in ci_result.tasks if t.task_name == "referable_dr" and t.metric_name == "auroc"),
        None,
    )
    auprc_task = next(
        (t for t in ci_result.tasks if t.task_name == "referable_dr" and t.metric_name == "auprc"),
        None,
    )
    return {
        "cell_id": cell_id,
        "display": cell_meta["display"],
        "backbone": cell_meta["backbone"],
        "head": cell_meta["head"],
        "protocol": cell_meta["protocol"],
        "n_valid": 1623,
        "n_referable_positive": n_pos,
        "auroc_point": auroc_task.point_estimate if auroc_task else None,
        "auroc_ci_lo": auroc_task.ci_lo if auroc_task else None,
        "auroc_ci_hi": auroc_task.ci_hi if auroc_task else None,
        "auroc_status": auroc_task.status if auroc_task else "na",
        "auprc_point": auprc_task.point_estimate if auprc_task else None,
        "auprc_ci_lo": auprc_task.ci_lo if auprc_task else None,
        "auprc_ci_hi": auprc_task.ci_hi if auprc_task else None,
        "auprc_status": auprc_task.status if auprc_task else "na",
        "n_resamples": ci_result.n_resamples,
        "seed": ci_result.seed,
        "n_patients": ci_result.n_patients,
    }


def _delta_result_to_rows(
    comp: dict[str, str],
    delta_result: Any,
) -> list[dict[str, Any]]:
    """Extract AUROC and AUPRC deltas from DeltaCIResult → list of dicts."""
    rows = []
    for metric_name in ("auroc", "auprc"):
        td = next(
            (t for t in delta_result.tasks
             if t.task_name == "referable_dr" and t.metric_name == metric_name),
            None,
        )
        rows.append({
            "group": comp["group"],
            "label": comp["label"],
            "cell_a": comp["cell_a"],
            "cell_b": comp["cell_b"],
            "display_a": CELL_REGISTRY[comp["cell_a"]]["display"],
            "display_b": CELL_REGISTRY[comp["cell_b"]]["display"],
            "metric": metric_name,
            "delta_point": td.delta_point if td else None,
            "delta_ci_lo": td.delta_ci_lo if td else None,
            "delta_ci_hi": td.delta_ci_hi if td else None,
            "status": td.status if td else "na",
            "n_resamples_ok": td.n_resamples_ok if td else 0,
            "source": td.source if td else "paired_bootstrap",
        })
    return rows




def main(args: argparse.Namespace) -> int:
    run_ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("outputs/stage8d35_c1_referable_dr") / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    n_resamples = args.n_resamples
    seed = args.seed
    strict = args.strict

    warnings: list[str] = []
    blocked: list[str] = []

    logger.info("=== PHASE 0: Pre-execution verification ===")

    checksums: dict[str, str] = {}
    for cell_id, meta in CELL_REGISTRY.items():
        npz_path = Path(meta["eval_dir"]) / "predictions.npz"
        schema_path = Path(meta["eval_dir"]) / "predictions_schema.json"
        metrics_path = Path(meta["eval_dir"]) / "overall_metrics.json"
        for p in (npz_path, schema_path, metrics_path):
            if not p.exists():
                blocked.append(f"PHASE0_MISSING_FILE: {p}")
                logger.error("BLOCKED: missing %s", p)
            else:
                checksums[str(p)] = _sha256_file(p)
    if blocked:
        logger.error("BLOCKED in PHASE 0 — aborting.")
        return 2

    logger.info("=== GATE M: n_valid and n_referable verification ===")

    cell_data: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_REGISTRY:
        d = _load_cell(cell_id)
        n_valid = int((d["mask_dr_grade"] == 1.0).sum())
        valid_labels = d["label_dr_grade"][d["mask_dr_grade"] == 1.0]
        n_referable = int((valid_labels >= 2).sum())

        if n_valid != 1623:
            blocked.append(f"GATE_M_NVALID_{cell_id}: expected 1623, got {n_valid}")
        if n_referable != 73:
            blocked.append(f"GATE_M_NREFERABLE_{cell_id}: expected 73, got {n_referable}")

        cell_data[cell_id] = d
        logger.info("GATE M %s: n_valid=%d, n_referable=%d", cell_id, n_valid, n_referable)

    if blocked:
        logger.error("BLOCKED in GATE M — aborting.")
        return 2

    logger.info("=== GATE P: Point estimate spot-check ===")

    spot_check_results: dict[str, dict[str, Any]] = {}
    for cell_id in SPOT_CHECK_CELLS:
        d = cell_data[cell_id]
        score = compute_referable_dr_score(d["logit_dr_grade"])
        ref_label = compute_referable_dr_label(d["label_dr_grade"])
        first10_scores = score[:10].tolist()
        first10_labels = ref_label[:10].tolist()
        spot_check_results[cell_id] = {
            "first_10_scores": [round(x, 6) for x in first10_scores],
            "first_10_labels": [int(x) if x != -1.0 else -1 for x in first10_labels],
        }
        logger.info(
            "GATE P %s: first-10 scores=%s",
            cell_id,
            [f"{x:.4f}" for x in first10_scores],
        )

    logger.info("=== PHASE 1: Per-cell referable-DR bootstrap CIs ===")

    cell_rows: list[dict[str, Any]] = []
    for cell_id, meta in CELL_REGISTRY.items():
        d = cell_data[cell_id]
        n_referable = int(
            (compute_referable_dr_label(d["label_dr_grade"][d["mask_dr_grade"] == 1.0]) == 1.0).sum()
        )
        logger.info("Computing CI for %s (%s)...", cell_id, meta["display"])
        try:
            ci_result = compute_referable_dr_bootstrap_ci(
                logit_dr_grade=d["logit_dr_grade"],
                label_dr_grade=d["label_dr_grade"],
                mask_dr_grade=d["mask_dr_grade"],
                patient_ids=d["patient_id"],
                cell_name=f"{cell_id}_{meta['backbone']}_{meta['head']}",
                n_resamples=n_resamples,
                seed=seed,
            )
        except ZeroPositivesInResampleError as exc:
            msg = (
                f"BLOCKED_C1_BOOTSTRAP_ZERO_POSITIVE_FAILURE: cell={cell_id} "
                f"task={exc.task_name} resample={exc.resample_idx}. "
                "Verify mask, labels, and sparse=True are passed correctly. "
                "Do NOT change seed to dodge this error. "
                "If confirmed correct, this indicates an irresolvable bootstrap issue."
            )
            logger.error(msg)
            blocked.append(msg)
            return 2

        auroc_task = next(
            (t for t in ci_result.tasks
             if t.task_name == "referable_dr" and t.metric_name == "auroc"),
            None,
        )
        logger.info(
            "  %s AUROC=%.4f [%.4f, %.4f] status=%s",
            cell_id,
            auroc_task.point_estimate if auroc_task and auroc_task.point_estimate is not None else float("nan"),
            auroc_task.ci_lo if auroc_task and auroc_task.ci_lo is not None else float("nan"),
            auroc_task.ci_hi if auroc_task and auroc_task.ci_hi is not None else float("nan"),
            auroc_task.status if auroc_task else "na",
        )

        row = _cell_result_to_row(cell_id, meta, ci_result, n_referable)
        cell_rows.append(row)

    logger.info("=== PHASE 2: Pairwise delta CIs (%d pairs) ===", len(PAIRWISE_COMPARISONS))

    delta_rows: list[dict[str, Any]] = []
    for comp in PAIRWISE_COMPARISONS:
        cid_a, cid_b = comp["cell_a"], comp["cell_b"]
        logger.info("Delta: [%s] Group %s — %s", f"{cid_a} vs {cid_b}", comp["group"], comp["label"])

        da, db = cell_data[cid_a], cell_data[cid_b]
        try:
            delta_result = compute_referable_dr_pair_delta(
                logit_dr_grade_a=da["logit_dr_grade"],
                label_dr_grade_a=da["label_dr_grade"],
                mask_dr_grade_a=da["mask_dr_grade"],
                patient_ids_a=da["patient_id"],
                logit_dr_grade_b=db["logit_dr_grade"],
                label_dr_grade_b=db["label_dr_grade"],
                mask_dr_grade_b=db["mask_dr_grade"],
                patient_ids_b=db["patient_id"],
                cell_name_a=f"{cid_a}_{CELL_REGISTRY[cid_a]['backbone']}_{CELL_REGISTRY[cid_a]['head']}",
                cell_name_b=f"{cid_b}_{CELL_REGISTRY[cid_b]['backbone']}_{CELL_REGISTRY[cid_b]['head']}",
                n_resamples=n_resamples,
                seed=seed,
            )
        except ZeroPositivesInResampleError as exc:
            msg = (
                f"BLOCKED_C1_BOOTSTRAP_ZERO_POSITIVE_FAILURE: pair={cid_a}_vs_{cid_b} "
                f"task={exc.task_name} resample={exc.resample_idx}."
            )
            logger.error(msg)
            blocked.append(msg)
            return 2

        rows = _delta_result_to_rows(comp, delta_result)
        auroc_row = next((r for r in rows if r["metric"] == "auroc"), None)
        if auroc_row:
            logger.info(
                "  AUROC delta=%.4f [%.4f, %.4f] %s",
                auroc_row["delta_point"] if auroc_row["delta_point"] is not None else float("nan"),
                auroc_row["delta_ci_lo"] if auroc_row["delta_ci_lo"] is not None else float("nan"),
                auroc_row["delta_ci_hi"] if auroc_row["delta_ci_hi"] is not None else float("nan"),
                auroc_row["status"],
            )
        delta_rows.extend(rows)

    logger.info("=== PHASE 3: 5-class limitation table ===")

    limitation_rows: list[dict[str, Any]] = []
    for cell_id, meta in CELL_REGISTRY.items():
        metrics_5class = _load_5class_metrics(cell_id)
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "display": meta["display"],
            "backbone": meta["backbone"],
            "head": meta["head"],
            "protocol": meta["protocol"],
        }
        for entry in metrics_5class:
            metric = entry.get("metric", "")
            val = entry.get("value")
            status = entry.get("status", "na")
            if metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                row[f"dr5_{metric}"] = val if status == "ok" else None
                row[f"dr5_{metric}_status"] = status
        limitation_rows.append(row)
        logger.info(
            "5-class %s: acc=%.4f macro_f1=%.4f bal_acc=%.4f",
            cell_id,
            row.get("dr5_accuracy") or float("nan"),
            row.get("dr5_macro_f1") or float("nan"),
            row.get("dr5_balanced_accuracy") or float("nan"),
        )

    logger.info("=== PHASE 4: Writing output files ===")

    checksums_path = out_dir / "input_artifact_checksums.json"
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump({"checksums": checksums, "algorithm": "sha256"}, f, indent=2)

    cell_metrics_csv = out_dir / "referable_dr_cell_metrics.csv"
    _write_csv(cell_rows, cell_metrics_csv)
    cell_metrics_json = out_dir / "referable_dr_cell_metrics.json"
    with open(cell_metrics_json, "w", encoding="utf-8") as f:
        json.dump(cell_rows, f, indent=2)

    _copy_file(cell_metrics_csv, out_dir / "c1_results.csv")
    _copy_file(cell_metrics_json, out_dir / "c1_results.json")

    deltas_csv = out_dir / "referable_dr_pairwise_deltas.csv"
    _write_csv(delta_rows, deltas_csv)
    deltas_json = out_dir / "referable_dr_pairwise_deltas.json"
    with open(deltas_json, "w", encoding="utf-8") as f:
        json.dump(delta_rows, f, indent=2)

    _copy_file(deltas_csv, out_dir / "c1_pair_deltas.csv")

    limit_csv = out_dir / "dr_grade_5class_limitation_table.csv"
    _write_csv(limitation_rows, limit_csv)
    limit_json = out_dir / "dr_grade_5class_limitation_table.json"
    with open(limit_json, "w", encoding="utf-8") as f:
        json.dump(limitation_rows, f, indent=2)

    logger.info("=== PHASE 5: Writing report and manifest ===")

    report_text = _build_report(
        run_ts=run_ts,
        cell_rows=cell_rows,
        delta_rows=delta_rows,
        limitation_rows=limitation_rows,
        spot_check_results=spot_check_results,
        n_resamples=n_resamples,
        seed=seed,
        warnings=warnings,
        blocked=blocked,
        out_dir=out_dir,
    )

    report_path = out_dir / "c1_referable_dr_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    _copy_file(report_path, out_dir / "c1_report.md")

    manifest = {
        "stage": "8D-3.5 C1",
        "run_timestamp_utc": run_ts,
        "n_cells": len(CELL_REGISTRY),
        "n_pairwise_comparisons": len(PAIRWISE_COMPARISONS),
        "n_delta_rows": len(delta_rows),
        "n_resamples": n_resamples,
        "seed": seed,
        "output_files": REQUIRED_OUTPUT_FILES,
        "warnings": warnings,
        "blocked": blocked,
        "status": "BLOCKED" if blocked else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
    }
    manifest_path = out_dir / "c1_referable_dr_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    missing_outputs: list[str] = []
    for fname in REQUIRED_OUTPUT_FILES:
        if not (out_dir / fname).exists():
            missing_outputs.append(fname)
    if missing_outputs:
        logger.error("MISSING OUTPUT FILES: %s", missing_outputs)
        return 2

    logger.info("=== C1 COMPLETE — %s ===", manifest["status"])
    logger.info("Output: %s", out_dir)
    logger.info("Files: %d required, all present", len(REQUIRED_OUTPUT_FILES))
    if warnings:
        logger.warning("Warnings: %s", warnings)

    if strict and warnings:
        return 1
    if blocked:
        return 2
    return 0




def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_file(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())




def _build_report(
    run_ts: str,
    cell_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    limitation_rows: list[dict[str, Any]],
    spot_check_results: dict[str, Any],
    n_resamples: int,
    seed: int,
    warnings: list[str],
    blocked: list[str],
    out_dir: Path,
) -> str:
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(*args: str) -> None:
        lines.append(" ".join(args))

    def blank() -> None:
        lines.append("")

    lines.append(f"# Stage 8D-3.5 C1 — Referable-DR Binary Endpoint Analysis\n")
    lines.append(f"**Generated:** {run_ts}  \n**Seed:** {seed}  \n**Bootstrap resamples:** {n_resamples}\n")

    h(2, "Stage Context and Purpose")
    p(
        "Stage 8D-3.5 C1 is a post-hoc analysis that recasts the BRSET dr_grade ordinal "
        "predictions into the clinically meaningful referable-DR binary endpoint (grade ≥ 2). "
        "This addresses the documented limitation that 5-class macro-accuracy is an "
        "inadequate clinical proxy for screening utility.",
    )
    blank()
    p(
        "This analysis reads only existing per-sample prediction artifacts "
        "(predictions.npz) from 14 accepted evaluation cells. No model training, "
        "embedding extraction, or evaluation rerun was performed.",
    )

    h(2, "Referable-DR Definition")
    lines.append("| Grade | Description | Referable? |\n|---|---|---|")
    lines.append("| 0 | No DR | No (nonreferable) |")
    lines.append("| 1 | Mild DR | No (nonreferable) |")
    lines.append("| 2 | Moderate DR | **Yes (referable)** |")
    lines.append("| 3 | Severe DR | **Yes (referable)** |")
    lines.append("| 4 | Proliferative DR (PDR) | **Yes (referable)** |")
    blank()
    lines.append(
        "**Score:** P(dr_grade ≥ 2) = softmax(logit__dr_grade)[:,2] + [:,3] + [:,4]  \n"
        "**Threshold:** grade ≥ 2 (not tuned; chosen by clinical convention)  \n"
        "**N positives:** 73 / 1623 test samples (4.50%)  \n"
        "**Class breakdown:** grade0=1517, grade1=33, grade2=11, grade3=22, grade4=40"
    )

    h(2, "Data Provenance")
    p(
        "All predictions were produced by Stage 8D-3.5 A1 patch "
        "(commits 9f2b95f8 + 861d1ed6). Predictions were saved by scripts/05_evaluate.py "
        "with final_test_result=false for all cells. BRSET test split: 1623 samples, "
        "854 unique patients.",
    )
    blank()
    p("**Privacy:** No raw patient IDs, image filenames, or metadata rows appear in this report.")

    h(2, "Computation Method")
    lines.append(
        "- Score: `softmax(logit__dr_grade, axis=1)[:,2:5].sum(axis=1)` (numerically stable)  \n"
        "- Bootstrap: patient-level cluster resampling, n=854 patients, 2000 resamples  \n"
        "- CI method: percentile bootstrap (2.5th–97.5th percentile)  \n"
        "- Logit conversion: `score → logit(score)` before passing to bootstrap_ci "
        "(AUROC/AUPRC rank-invariant under sigmoid applied internally by bootstrap_ci.py)  \n"
        "- AUPRC: always computed (forced `sparse=True` in task_metadata)  \n"
        "- Paired delta: same bootstrap patient resamples applied to both cells"
    )

    h(2, "Per-Cell Referable-DR AUROC (with 95% CI)")
    lines.append("| Cell | Model | AUROC | 95% CI | AUROC status |")
    lines.append("|------|-------|-------|--------|--------------|")
    for row in sorted(cell_rows, key=lambda r: -(r["auroc_point"] or 0)):
        ci_str = (
            f"[{_fmt(row['auroc_ci_lo'])}, {_fmt(row['auroc_ci_hi'])}]"
            if row["auroc_ci_lo"] is not None
            else "na"
        )
        lines.append(
            f"| {row['cell_id']} | {row['display']} "
            f"| {_fmt(row['auroc_point'])} | {ci_str} | {row['auroc_status']} |"
        )

    h(2, "Per-Cell Referable-DR AUPRC (with 95% CI)")
    lines.append("| Cell | Model | AUPRC | 95% CI | AUPRC status |")
    lines.append("|------|-------|-------|--------|--------------|")
    for row in sorted(cell_rows, key=lambda r: -(r["auprc_point"] or 0)):
        ci_str = (
            f"[{_fmt(row['auprc_ci_lo'])}, {_fmt(row['auprc_ci_hi'])}]"
            if row["auprc_ci_lo"] is not None
            else "na"
        )
        lines.append(
            f"| {row['cell_id']} | {row['display']} "
            f"| {_fmt(row['auprc_point'])} | {ci_str} | {row['auprc_status']} |"
        )

    h(2, "Bootstrap CI Validity")
    ok_count = sum(1 for r in cell_rows if r["auroc_status"] == "ok")
    lines.append(
        f"- Cells with AUROC CI status=ok: {ok_count} / {len(cell_rows)}  \n"
        f"- n_resamples: {n_resamples}  \n"
        f"- seed: {seed}  \n"
        f"- ZeroPositivesInResampleError: not encountered (n_pos=73 across 854 patients)"
    )

    h(2, "Pairwise Delta CIs — Group A: Within-Backbone MT vs LP")
    _append_delta_table(lines, delta_rows, "A")

    h(2, "Pairwise Delta CIs — Group B: Top-Tier MT Backbone Comparisons")
    _append_delta_table(lines, delta_rows, "B")

    h(2, "Pairwise Delta CIs — Group C: LP Backbone Comparisons")
    _append_delta_table(lines, delta_rows, "C")

    h(2, "Pairwise Delta CIs — Group D: RETFound Protocol Delta")
    _append_delta_table(lines, delta_rows, "D")

    h(2, "Pairwise Delta CIs — Group E: Off-Protocol Comparator vs Top-Tier MT")
    _append_delta_table(lines, delta_rows, "E")

    h(2, "5-Class Ordinal dr_grade Limitation Table")
    p(
        "The 5-class dr_grade protocol is retained as a documented limitation. "
        "Accuracy is inflated by the dominant grade-0 class (93.5%); macro-F1 is "
        "the meaningful metric but is compromised by very small class counts "
        "(grade2=11, grade3=22). Referable-DR AUROC is the recommended primary DR endpoint."
    )
    blank()
    lines.append("| Cell | Model | Accuracy | Macro-F1 | Balanced Accuracy |")
    lines.append("|------|-------|----------|----------|-------------------|")
    for row in limitation_rows:
        lines.append(
            f"| {row['cell_id']} | {row['display']} "
            f"| {_fmt(row.get('dr5_accuracy'))} "
            f"| {_fmt(row.get('dr5_macro_f1'))} "
            f"| {_fmt(row.get('dr5_balanced_accuracy'))} |"
        )

    h(2, "F3 Activation Decision Support")
    top_auroc_row = max(
        (r for r in cell_rows if r["auroc_point"] is not None),
        key=lambda r: r["auroc_point"],
        default=None,
    )
    if top_auroc_row:
        best_auroc = top_auroc_row["auroc_point"]
        best_ci_lo = top_auroc_row["auroc_ci_lo"]
        lines.append(
            f"Best referable-DR AUROC: **{_fmt(best_auroc)}** "
            f"(CI lo={_fmt(best_ci_lo)}) — {top_auroc_row['display']}"
        )
        blank()
        if best_auroc is not None and best_auroc > 0.85 and best_ci_lo is not None and best_ci_lo > 0.82:
            lines.append(
                "**F3 activation signal: RUN_F3** — Top-tier AUROC > 0.85 with CI lo > 0.82. "
                "Referable-DR endpoint is sufficiently strong to support F3 fairness analysis."
            )
        elif best_auroc is not None and best_auroc > 0.80:
            lines.append(
                "**F3 activation signal: RUN_F3_WITH_LIMITED_SCOPE** — "
                "Moderate referable-DR AUROC; F3 may proceed with documented scope limitations."
            )
        else:
            lines.append(
                "**F3 activation signal: DEFER_F3** — "
                "Referable-DR AUROC is below threshold for F3 activation."
            )
        blank()
        p("Note: The final F3 decision is deferred to the project lead. This section presents findings only.")

    h(2, "Privacy Statement")
    p(
        "This report contains only aggregate statistics. No raw patient IDs, "
        "image filenames, clinical notes, or individual-level data appear anywhere "
        "in this report or its output files. BRSET is credentialed PhysioNet data."
    )

    h(2, "Input Artifact Checksums Reference")
    p(f"SHA-256 checksums for all {len(CELL_REGISTRY) * 3} input files are recorded in:")
    blank()
    lines.append(f"  `{out_dir / 'input_artifact_checksums.json'}`")

    h(2, "GATE P Spot-Check Verification")
    p("First 10 sample referable-DR scores for 3 spot-check cells:")
    blank()
    for cell_id, sc in spot_check_results.items():
        lines.append(f"**{cell_id} ({CELL_REGISTRY[cell_id]['display']}):**")
        lines.append(f"  Scores:  {sc['first_10_scores']}")
        lines.append(f"  Labels:  {sc['first_10_labels']}")
        blank()

    if warnings:
        h(2, "Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    if blocked:
        h(2, "BLOCKED")
        for b in blocked:
            lines.append(f"- {b}")

    return "\n".join(lines)


def _append_delta_table(lines: list[str], delta_rows: list[dict[str, Any]], group: str) -> None:
    group_rows = [r for r in delta_rows if r["group"] == group]
    if not group_rows:
        lines.append("*No pairs in this group.*")
        return
    auroc_rows = [r for r in group_rows if r["metric"] == "auroc"]
    lines.append("| Pair | A | B | AUROC Δ | 95% CI | Status |")
    lines.append("|------|---|---|---------|--------|--------|")
    for row in auroc_rows:
        ci_str = (
            f"[{_fmt(row['delta_ci_lo'])}, {_fmt(row['delta_ci_hi'])}]"
            if row["delta_ci_lo"] is not None
            else "na"
        )
        lines.append(
            f"| {row['label']} | {row['display_a']} | {row['display_b']} "
            f"| {_fmt(row['delta_point'])} | {ci_str} | {row['status']} |"
        )




def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 8D-3.5 C1: referable-DR binary endpoint analysis"
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output directory (default: auto-timestamped under outputs/stage8d35_c1_referable_dr/)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples per cell (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero on any WARNING in addition to BLOCKED conditions",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
