from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.referable_dr_demo.analysis import verify_parity  # noqa: E402
from deploy.referable_dr_demo.backend.service import bundle as bundle_mod  # noqa: E402
from deploy.referable_dr_demo.backend.service.provenance import (  # noqa: E402
    LOCAL_DIR,
    REPO_ROOT,
    ensure_src_importable,
    git_commit,
    utc_timestamp,
)

logger = logging.getLogger(__name__)

TARGET_SENSITIVITY = 0.95
BLOCKED = "BLOCKED_THRESHOLD_SELECTION_VALIDATION_ARTIFACT_MISSING"

THRESHOLD_ALGORITHM = (
    "Compute referable score = softmax(dr_grade_logits)[2:5].sum() for every valid "
    "validation sample (mask==1, observed grade). Let P be the number of referable "
    "positives and needed = ceil(0.95 * P). Sort positive scores in descending "
    "order; the threshold T is the (needed)-th largest positive score. Decision "
    "rule: score >= T -> REFERABLE. This is the largest threshold preserving "
    "validation sensitivity >= 0.95 (maximizing specificity subject to the floor); "
    "boundary ties at exactly T are counted as positive."
)

ensure_src_importable()


def _load_splits(resolved_config_path: str) -> dict[str, list[str]]:
    import yaml  # noqa: PLC0415

    run_cfg = yaml.safe_load(Path(resolved_config_path).read_text(encoding="utf-8"))
    splits_dir = run_cfg.get("splits_dir")
    if not splits_dir:
        raise RuntimeError(f"{BLOCKED}: splits_dir not recorded in resolved_config")
    splits_path = verify_parity._resolve_repo_path(str(splits_dir).replace("\\", "/"))
    csv_path = splits_path / "splits.csv"
    if not csv_path.exists():
        raise RuntimeError(f"{BLOCKED}: splits.csv not found at {csv_path}")
    by_split: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_split.setdefault(row["split_name"], []).append(row["sample_id"])
    return by_split


def _cache_manifest_index(backbone: str, dataset_source: str, prep_hash: str) -> dict[str, dict]:
    cache_dir = REPO_ROOT / "cache" / "embeddings" / backbone / dataset_source / prep_hash
    manifest = cache_dir / "manifest.csv"
    if not manifest.exists():
        raise RuntimeError(f"{BLOCKED}: embedding cache manifest not found at {manifest}")
    index: dict[str, dict] = {}
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            index[row["sample_id"]] = row
    return index, cache_dir


def reconstruct_validation_predictions(
    bundle: bundle_mod.DeploymentBundle,
) -> dict[str, np.ndarray]:
    """Return {'logits': (N,5), 'labels': (N,), 'mask': (N,)} for cached val samples."""
    import torch  # noqa: PLC0415

    from retina_screen.data import build_task_targets_and_masks  # noqa: PLC0415
    from retina_screen.embeddings import load_embedding  # noqa: PLC0415
    from retina_screen.model import build_head  # noqa: PLC0415

    rc = bundle.manifest.get("resolved_config_path")
    if not rc or not Path(rc).exists():
        raise RuntimeError(f"{BLOCKED}: resolved_config.yaml unavailable for the run")

    adapter, why = verify_parity._build_adapter_from_run(bundle)
    if adapter is None:
        raise RuntimeError(f"{BLOCKED}: cannot build adapter for validation labels: {why}")

    splits = _load_splits(rc)
    val_sids = splits.get("val", [])
    if not val_sids:
        raise RuntimeError(f"{BLOCKED}: no validation split entries found")

    probe = adapter.load_sample(val_sids[0])
    dataset_source = probe.dataset_source
    index, _cache_dir = _cache_manifest_index(
        bundle.manifest["backbone_identifier"], dataset_source, bundle.preprocessing_hash
    )

    ordered_sids: list[str] = []
    emb_list: list[np.ndarray] = []
    for sid in val_sids:
        row = index.get(sid)
        if row is None:
            continue
        cache_path = verify_parity._resolve_repo_path(row["cache_path"].replace("\\", "/"))
        tensor = load_embedding(cache_path, row["checksum"], int(row["embedding_dim"]))
        emb_list.append(np.asarray(tensor, dtype=np.float32))
        ordered_sids.append(sid)

    if len(ordered_sids) < 30:
        raise RuntimeError(
            f"{BLOCKED}: only {len(ordered_sids)} cached validation embeddings resolved"
        )

    head = build_head(
        embedding_dim=bundle.embedding_dim,
        task_names=list(bundle.task_order),
        head_type="multitask",
    )
    head.load_state_dict(
        torch.load(bundle.head_checkpoint, map_location="cpu", weights_only=True)
    )
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    emb = torch.from_numpy(np.stack(emb_list)).float()
    with torch.inference_mode():
        logits = head(emb)[bundle.manifest["dr_grade_task_key"]]
    logits_np = logits.detach().cpu().numpy().astype(np.float64)

    samples = [adapter.load_sample(sid) for sid in ordered_sids]
    batch = build_task_targets_and_masks(samples, ["dr_grade"])
    labels = np.asarray(batch.targets["dr_grade"], dtype=np.float64)
    mask = np.asarray(batch.masks["dr_grade"], dtype=np.float64)
    return {"logits": logits_np, "labels": labels, "mask": mask}


def select_threshold(score: np.ndarray, label: np.ndarray, target: float) -> tuple[float, int, int]:
    pos = np.asarray(score)[np.asarray(label) == 1.0]
    P = int(pos.size)
    if P == 0:
        raise RuntimeError(f"{BLOCKED}: zero referable positives in validation set")
    needed = int(np.ceil(target * P - 1e-9))
    needed = max(1, min(needed, P))
    pos_desc = np.sort(pos)[::-1]
    threshold = float(pos_desc[needed - 1])
    return threshold, needed, P


def sens_spec(score: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, float]:
    s = np.asarray(score)
    y = np.asarray(label)
    pred = s >= threshold
    P = int((y == 1.0).sum())
    N = int((y == 0.0).sum())
    tp = int((pred & (y == 1.0)).sum())
    tn = int((~pred & (y == 0.0)).sum())
    sens = tp / P if P > 0 else float("nan")
    spec = tn / N if N > 0 else float("nan")
    return {"sensitivity": sens, "specificity": spec, "positives": P, "negatives": N}


def _referable_arrays(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from retina_screen.evaluation.referable_dr import (  # noqa: PLC0415
        make_referable_dr_from_dr_grade_logits,
    )

    out = make_referable_dr_from_dr_grade_logits(logits, labels, mask)
    return out["score"], out["label"]


def derive(write: bool = True) -> dict[str, Any]:
    bundle = bundle_mod.load_validated_bundle()

    val = reconstruct_validation_predictions(bundle)
    val_score, val_label = _referable_arrays(val["logits"], val["labels"], val["mask"])
    threshold, needed, P_val = select_threshold(val_score, val_label, TARGET_SENSITIVITY)
    val_metrics = sens_spec(val_score, val_label, threshold)
    if val_metrics["sensitivity"] < TARGET_SENSITIVITY - 1e-9:
        raise RuntimeError(
            f"{BLOCKED}: derived threshold yields validation sensitivity "
            f"{val_metrics['sensitivity']:.4f} < {TARGET_SENSITIVITY}"
        )

    test_metrics: dict[str, Any]
    preds_path = verify_parity._discover_test_predictions(bundle)
    if preds_path is None:
        raise RuntimeError(f"{BLOCKED}: accepted held-out test predictions.npz not found")
    data = np.load(preds_path, allow_pickle=True)
    test_score, test_label = _referable_arrays(
        np.asarray(data["logit__dr_grade"], dtype=np.float64),
        np.asarray(data["label__dr_grade"], dtype=np.float64),
        np.asarray(data["mask__dr_grade"], dtype=np.float64),
    )
    test_metrics = sens_spec(test_score, test_label, threshold)

    manifest: dict[str, Any] = {
        "threshold": threshold,
        "derivation_split": "validation",
        "target_sensitivity": TARGET_SENSITIVITY,
        "validation_sensitivity": val_metrics["sensitivity"],
        "validation_specificity": val_metrics["specificity"],
        "validation_positive_count": val_metrics["positives"],
        "validation_negative_count": val_metrics["negatives"],
        "validation_needed_positives_at_floor": needed,
        "heldout_test_sensitivity": test_metrics["sensitivity"],
        "heldout_test_specificity": test_metrics["specificity"],
        "test_positive_count": test_metrics["positives"],
        "test_negative_count": test_metrics["negatives"],
        "backbone_checkpoint_sha256": bundle.backbone_sha256,
        "head_checkpoint_sha256": bundle.head_sha256,
        "preprocessing_hash": bundle.preprocessing_hash,
        "task_ordering_hash": bundle.task_ordering_hash,
        "model_family": bundle.manifest.get("model_family"),
        "native_protocol": bundle.manifest.get("native_protocol"),
        "threshold_selection_algorithm": THRESHOLD_ALGORITHM,
        "test_predictions_path": str(preds_path),
        "timestamp": utc_timestamp(),
        "git_commit": git_commit(),
    }

    if write:
        override = os.environ.get("RETINA_SCREEN_DEMO_THRESHOLD_PATH")
        out = Path(override) if override else LOCAL_DIR / "operating_point.local.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest["_written_to"] = str(out)
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = derive(write=True)
    print("OK: derived validation-only operating point")
    print(f"  threshold                : {m['threshold']:.6f}")
    print(f"  derivation_split         : {m['derivation_split']}")
    print(f"  target_sensitivity       : {m['target_sensitivity']}")
    print(f"  validation sens / spec   : {m['validation_sensitivity']:.4f} / "
          f"{m['validation_specificity']:.4f} "
          f"(P={m['validation_positive_count']}, N={m['validation_negative_count']})")
    print(f"  held-out test sens / spec: {m['heldout_test_sensitivity']:.4f} / "
          f"{m['heldout_test_specificity']:.4f} "
          f"(P={m['test_positive_count']}, N={m['test_negative_count']})")
    print(f"  written to               : {m.get('_written_to')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
