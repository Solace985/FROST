# Stage 8D-3.5 A1 Bootstrap CI — Pre-flight Summary

**Preflight ID:** A1  
**Date run:** 2026-05-23  
**Timestamp:** 20260523_222803  
**Matrix:** Stage 8D-3 (4 backbones × 2 heads, BRSET full dataset, Decision 027 unweighted recipe)  
**Machine-readable artifact:** `outputs/stage8d35_a1_preflight/20260523_222803/matrix_artifact_manifest.json`

---

## §1 — Verdict

**VERDICT: BLOCKED**

No per-sample prediction artifacts exist for any of the 8 Stage 8D-3 matrix cells. Stage 8D-3.5 A1 bootstrap confidence-interval analysis cannot proceed until `scripts/05_evaluate.py` is patched to save per-sample predictions alongside the aggregate evaluation metrics.

See §4 (Primary Blocker) and §11 (Required Upstream Fix) for the exact diagnosis and the minimum required change.

---

## §2 — Preflight Metadata

| Field | Value |
|-------|-------|
| Preflight ID | A1 |
| Date run | 2026-05-23 |
| Preflight timestamp | 20260523_222803 |
| Matrix | Stage 8D-3: 4 backbones × 2 heads on BRSET full dataset |
| Backbones | ResNet-50, ConvNeXt-Base, DINOv2-Base, DINOv2-Large |
| Heads | MultiTaskHead (multitask), LinearProbeHead (linear_probe) |
| Dataset | BRSET (full dataset run, seed=42) |
| Recipe lock | Decision 027 — unweighted, class_weighting_enabled=false |
| W1 storage fix | Decision 028 — unconditional `_compact_embedding()` at embeddings.py:675, commit 46aef542 |
| Machine-readable manifest | `outputs/stage8d35_a1_preflight/20260523_222803/matrix_artifact_manifest.json` |

---

## §3 — Matrix Cell Inventory

All 8 cells inspected. Aggregate evaluation metrics exist for all 8. Per-sample prediction artifacts absent for all 8.

| Cell | Backbone | Head | Stage | Eval Dir | Agg Metrics | Per-Sample Preds |
|------|----------|------|-------|----------|-------------|-----------------|
| resnet50_multitask | ResNet-50 | multitask | 8D-2 | `outputs/evaluation/20260512_192437` | YES | **NO — BLOCKER** |
| resnet50_linearprobe | ResNet-50 | linear_probe | 8D-3B | `outputs/evaluation/20260515_005309` | YES | **NO — BLOCKER** |
| convnext_base_multitask | ConvNeXt-Base | multitask | 8D-3C | `outputs/evaluation/20260515_152007` | YES | **NO — BLOCKER** |
| convnext_base_linearprobe | ConvNeXt-Base | linear_probe | 8D-3C | `outputs/evaluation/20260515_152223` | YES | **NO — BLOCKER** |
| dinov2_base_multitask | DINOv2-Base | multitask | 8D-3D | `outputs/evaluation/20260515_235231` | YES | **NO — BLOCKER** |
| dinov2_base_linearprobe | DINOv2-Base | linear_probe | 8D-3D | `outputs/evaluation/20260515_235605` | YES | **NO — BLOCKER** |
| dinov2_large_multitask | DINOv2-Large | multitask | 8D-3E | `outputs/evaluation/20260518_233207` | YES | **NO — BLOCKER** |
| dinov2_large_linearprobe | DINOv2-Large | linear_probe | 8D-3E | `outputs/evaluation/20260518_233445` | YES | **NO — BLOCKER** |

Each evaluation directory contains exactly 5 aggregate files and nothing else:
`overall_metrics.json`, `subgroup_metrics.json`, `evaluation_summary.json`, `smoke_metrics.json`, `diagnostics.json`.

---

## §4 — Primary Blocker: Missing Per-Sample Predictions

**Root cause:** `scripts/05_evaluate.py:605` computes per-sample predictions in-memory:

```python
preds_np = {t: preds[t].cpu().numpy() for t in supported_tasks}
```

This dict is used to compute aggregate metrics via `evaluate_predictions()` but is **never saved to disk**. The companion `eval_samples` list — which carries `sample_id` and `patient_id` for each test sample — is also never persisted.

**Consequence:** A1 bootstrap CI requires resampling per-sample (logit, label, mask) triples at the patient level. Because no per-sample artifact exists on disk, there is nothing to resample.

**Exhaustive search:** All 8 evaluation output directories were searched for any file matching `predictions.*`, `preds.*`, `*.npz`, `*.npy`, `*.csv` (beyond splits.csv), `*.pt` (beyond model checkpoints in run_dirs). None found. The 5-file structure is confirmed exhaustive for all 8 cells.

**No per-sample artifact means:**
- Per-sample logits: NOT available
- Per-sample labels: NOT available
- Per-sample masks: NOT available
- sample_id linkage: NOT available in evaluation outputs
- patient_id linkage: NOT available in evaluation outputs

**Blocker label:** `BLOCKER_A1_NO_PER_SAMPLE_PREDICTIONS`

---

## §5 — What IS Available

The following resources are intact and can support a re-evaluation approach after the patch:

| Resource | Location | Note |
|----------|----------|------|
| Model checkpoints (8) | `runs/train/brset_<ts>/model_checkpoint.pt` | All 8 present |
| Embedding caches (4 backbones) | `cache/embeddings/<backbone>/brset/<hash>/` | All 4 present; W1-compact |
| Patient-level splits | `outputs/splits/brset/20260507_143232/splits.csv` | sample_id + patient_id + split_name |
| Task configs | `configs/tasks/brset_default.yaml` | 7 tasks (6 binary + dr_grade ordinal) |
| Aggregate eval metrics (8) | `outputs/evaluation/<ts>/overall_metrics.json` | Point estimates only; cannot bootstrap |

After the patch to `scripts/05_evaluate.py`, all 8 cells can be re-evaluated using existing checkpoints and caches without any retraining or re-extraction.

---

## §6 — Test Split Characterization (aggregate counts only)

Source: `outputs/splits/brset/20260507_143232/splits.csv`

| Metric | Value |
|--------|-------|
| Test samples | 1,623 |
| Unique patients in test split | 854 |
| Maximum images per patient | 3 |

**Bootstrap constraint:** A1 must resample at the patient level, not the sample level, to respect the patient-level split design (Decision per `configs/dataset/brset.yaml`). The effective bootstrap resample unit count is **854 patients**, not 1,623 samples. Patient-to-sample expansion is non-trivial: 768 patients have more than one image; a patient drawn twice contributes all their images twice.

---

## §7 — Label Counts in Test Split (from aggregate artifacts only)

Source: `outputs/evaluation/20260512_192437/overall_metrics.json` (all 8 cells show identical label counts — same test split, same masking).

**Binary tasks (all have observed = 1,623 for all 8 cells):**

| Task | Positives | Negatives | Positive Rate |
|------|-----------|-----------|---------------|
| amd | 22 | 1,601 | 1.4% |
| macular_edema | 33 | 1,590 | 2.0% |
| hypertensive_retinopathy | 30 | 1,593 | 1.8% |
| drusen | 261 | 1,362 | 16.1% |
| other_ocular | 143 | 1,480 | 8.8% |
| diabetes | 230 | 1,393 | 14.2% |

**dr_grade class counts (ordinal, 5 classes):**

| Grade | Count | Proportion |
|-------|-------|-----------|
| 0 (no DR) | 1,517 | 93.5% |
| 1 (mild) | 33 | 2.0% |
| 2 (moderate) | 11 | 0.7% |
| 3 (severe) | 22 | 1.4% |
| 4 (proliferative) | 40 | 2.5% |

**Sparse-positive caveat:** amd (22 positives), macular_edema (33), hypertensive_retinopathy (30), and dr_grade grades 1–4 all have fewer than 40 positives. Bootstrap CI widths on these tasks will be wide. The project's sparse-subgroup NA convention (n < 30, positives < 5) per Decision 014 applies at the subgroup level; at the full test-split level amd/ME/HR are above the n < 30 threshold but are noted as low-prevalence tasks where CIs will dominate any point comparison.

---

## §8 — Masking Convention (from source code inspection)

Sources: `src/retina_screen/data.py`, `src/retina_screen/evaluation.py`

| Symbol | Value | Meaning |
|--------|-------|---------|
| `mask` | `1.0` | Label is observed; sample is included in metric computation |
| `mask` | `0.0` | Label is missing; sample is excluded from metric computation |
| `MISSING_CLASS_PLACEHOLDER` | `-1.0` | Value stored in label field when mask = 0.0 (binary and ordinal tasks) |
| Missing label (regression) | `NaN` | Not applicable for current BRSET tasks |

Evaluation filter (from `src/retina_screen/evaluation.py`):
```python
valid = (mask_arr == 1.0) & ~np.isnan(target_arr)
```

**BRSET test split:** All 8 cells show observed = 1,623 for all 7 tasks, meaning mask = 1.0 for every sample on every task in the BRSET test split. There are no missing labels in the test set.

---

## §9 — Score Orientation (from source code inspection)

Source: `src/retina_screen/evaluation.py:239`

```python
y_score = 1.0 / (1.0 + np.exp(-y_pred.astype(np.float64)))
```

- **Format stored in preds_np:** raw logits (pre-sigmoid)
- **Transform applied before AUROC:** sigmoid
- **Orientation:** higher logit → higher sigmoid score → higher probability of positive class
- **`higher_means_positive`:** `True` for all 6 binary tasks
- **dr_grade (ordinal 5-class):** per-class logits with shape `(n_samples, 5)`; per-class vs rest AUROC is computed; the downstream A1 utility must apply softmax or per-class sigmoid as appropriate when consuming the ordinal logit array

When `scripts/05_evaluate.py` is patched to save `predictions.npz`, the logits must be saved **raw (pre-sigmoid)** at the precision the model outputs them (typically float32). The downstream A1 utility applies sigmoid/softmax; pre-applying in the save is forbidden per the task specification.

---

## §10 — Secondary Issue: Missing Planner Document

`docs/planners/stage_8D-3-5_intermediate_planner.md` was listed as required reading in the A1 preflight task prompt. This file and the `docs/planners/` directory do not exist. The operational instructions in the A1 task prompt itself were sufficient to complete the preflight research. This is a documentation gap, not a blocker for the preflight or the patch.

---

## §11 — Required Upstream Fix to Unblock A1

**Minimum required change:** Modify `scripts/05_evaluate.py` only (no src/retina_screen/ changes, no config changes, no retraining).

After the `preds_np` dict is computed (line 605) and after `batch = build_task_targets_and_masks(eval_samples, supported_tasks)` (line 608), save a per-sample artifact with the following fields:

**Required fields in `predictions.npz`:**
- `sample_id` — from `eval_samples[i].sample_id`
- `patient_id` — from `eval_samples[i].patient_id`
- `logit__<task_name>` — raw per-sample logit for each task in `supported_tasks`
- `label__<task_name>` — per-sample label for each task (from `batch.targets[task]`)
- `mask__<task_name>` — per-sample mask for each task (from `batch.masks[task]`)

**Required companion file `predictions_schema.json`:**
Records `n_eval_samples`, `tasks`, `task_types`, logit/label/mask dtypes, `score_orientation` (higher_means_positive=True for binary tasks; per-class convention for ordinal), masking convention, field name convention, git commit SHA.

**Format:** `numpy.savez_compressed` (`.npz`). Saved in the same evaluation output directory as `overall_metrics.json`.

**Scope:** Only `scripts/05_evaluate.py` needs modification. No architecture changes, no config changes, no retraining, no embedding re-extraction.

**After patch:** Re-run `scripts/05_evaluate.py` for all 8 cells using existing checkpoints and caches. Re-evaluation is deterministic. The re-run must reproduce the original aggregate metrics within float tolerance (|delta| < 1e-6 on all numeric fields).

---

## §12 — Confirmation Checklist

| Item | Status | Evidence |
|------|--------|---------|
| All 8 eval dirs inspected | CONFIRMED | 5-file structure verified for each |
| Per-sample prediction search exhaustive | CONFIRMED | No predictions.* / preds.* / *.npz / *.npy found in any eval dir |
| `scripts/05_evaluate.py` source read for save behavior | CONFIRMED | preds_np at line 605 not saved; eval_samples not persisted |
| `src/retina_screen/evaluation.py` masking convention verified | CONFIRMED | mask=1.0/0.0, MISSING_CLASS_PLACEHOLDER=-1.0 |
| Score orientation verified | CONFIRMED | evaluation.py:239 sigmoid on raw logits; higher=positive |
| Test split characterization from splits.csv | CONFIRMED | 1,623 samples, 854 patients, patient-level splits |
| Label counts from overall_metrics.json only (not computed) | CONFIRMED | Aggregate read-only; no bootstrap performed |
| Privacy constraint: no raw patient IDs in this document | CONFIRMED | Aggregate counts only |
| Privacy constraint: no image filenames in this document | CONFIRMED | No image paths recorded |
| Privacy constraint: no free-text clinical fields in this document | CONFIRMED | |
| matrix_artifact_manifest.json written | CONFIRMED | At `outputs/stage8d35_a1_preflight/20260523_222803/matrix_artifact_manifest.json` |
| No bootstrap, no CI, no performance delta computed | CONFIRMED | Preflight scope: existence check and source inspection only |
