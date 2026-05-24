# Project Decisions Log

This file records locked project decisions.

If this file conflicts with older documents, follow `docs/ai_context/00_source_of_truth_order.md`.

Status labels:

- `locked`: do not change without explicit user approval.
- `open`: unresolved; ask before implementing.
- `deferred`: not part of current MVP.

---

## Decision 001 — Use Modular Monolith Architecture

Status: locked

Decision:

Use a modular monolith under:

src/retina_screen/

Rationale:

The project is larger than a single experiment file. It requires dataset adapters, canonical schema, task registry, FeaturePolicy, patient-level splitting, embedding cache, fairness evaluation, continual-learning simulation, explainability, dashboard inference, and reporting.

Consequences:

- Do not collapse implementation into one giant script.
- Do not over-fragment before MVP.
- Split large files only after MVP if a file exceeds size-budget warnings and the split is approved.

Files affected:

docs/architecture.md
src/retina_screen/\*

---

## Decision 002 — Dataset-Specific Logic Lives Only in Adapters and Configs

Status: locked

Decision:

Dataset-specific parsing, native column names, label vocabularies, camera names, file paths, and dataset-specific quirks belong only in:

src/retina*screen/adapters/*
configs/dataset/_
configs/tasks/_
configs/experiment/_
docs/_
tests/\_

Rationale:

The pipeline must remain dataset-agnostic downstream of adapters so BRSET/mBRSET/external datasets can be added without refactoring model, training, evaluation, or dashboard code.

Consequences:

- No dataset-native parsing in `model.py`, `training.py`, `evaluation.py`, `data.py`, `preprocessing.py`, `embeddings.py`, `continual.py`, or `dashboard_app.py`.
- New datasets should require adapter/config/task additions only.

Files affected:

src/retina_screen/adapters/\*
src/retina_screen/schema.py
src/retina_screen/tasks.py
src/retina_screen/data.py
src/retina_screen/training.py
src/retina_screen/evaluation.py

---

## Decision 003 — Use Canonical Schema and Task Registry

Status: locked

Decision:

All datasets must be converted into a canonical schema. All tasks must be declared through a task registry.

Rationale:

Datasets differ in labels, missingness, DR grading systems, metadata, and supported tasks. Downstream code must operate on canonical fields only.

Consequences:

- `schema.py` is the single source of truth for canonical sample fields.
- `tasks.py` is the single source of truth for task definitions.
- Do not duplicate canonical field lists in docs or adapters.
- Do not hardcode task lists in training/evaluation.

Files affected:

src/retina_screen/schema.py
src/retina_screen/tasks.py
src/retina_screen/adapters/\*

---

## Decision 004 — FeaturePolicy Is Mandatory

Status: locked

Decision:

All metadata entering the model must pass through `feature_policy.py`.

Rationale:

Metadata can leak targets or create shortcuts. Age cannot be used to predict retinal age. Sex cannot be used to predict sex. Camera and dataset source can become shortcuts.

Consequences:

- FeaturePolicy must block leakage.
- Image-only and image-plus-metadata modes must be separate.
- Metadata dropout is not enough to prevent leakage.

Files affected:

src/retina_screen/feature_policy.py
src/retina_screen/data.py
src/retina_screen/model.py
tests/test_feature_policy.py

---

## Decision 005 — Use 60/15/15/10 Patient-Level Split

Status: locked

Decision:

Use:

train: 60%
validation: 15%
reliability: 15%
test: 10%

All splits are patient-level.

Rationale:

The original project specification used a simpler split, but the issues log introduced a separate reliability split for dashboard subgroup reliability lookup. Later-overrides-earlier precedence makes 60/15/15/10 the active rule.

Consequences:

- No image-level splitting.
- Reliability split is not used for training, early stopping, or model selection.
- Reliability split is used to generate subgroup reliability lookup tables.
- Split audit must prove zero patient overlap.

Files affected:

src/retina_screen/splitting.py
src/retina_screen/evaluation.py
tests/test_patient_split.py
tests/test_split_audit.py

---

## Decision 006 — Build DummyAdapter MVP Before Real Dataset Work

Status: locked

Decision:

The first working pipeline must use DummyAdapter.

Target path:

DummyAdapter
→ canonical schema
→ patient-level split
→ mock embeddings
→ dataloader
→ FeaturePolicy
→ task masks
→ model
→ masked loss
→ evaluation

Rationale:

DummyAdapter catches dataset coupling before ODIR or BRSET logic enters the system.

Consequences:

- Do not implement ODIR training before dummy E2E passes.
- Do not jump to real backbones before mock embedding flow works.

Files affected:

src/retina_screen/adapters/dummy.py
src/retina_screen/data.py
src/retina_screen/model.py
src/retina_screen/training.py
src/retina_screen/evaluation.py
tests/test_dummy_e2e.py

---

## Decision 007 — ODIR-Only Mode Cannot Headline Systemic Prediction

Status: locked

Decision:

If only ODIR is available, the paper is framed as:

multi-condition retinal screening + fairness + cross-site robustness + continual-learning simulation

not as definitive systemic disease prediction.

Rationale:

ODIR diabetes/hypertension labels are weak proxy labels, not structured clinical systemic records.

Consequences:

- ODIR diabetes/hypertension heads may exist.
- They must be reported as weak proxy labels.
- ODIR-only mode must not headline whole-body/systemic prediction.
- CV composite and retinal age gap are secondary/dashboard convenience outputs in ODIR-only mode.

Files affected:

configs/paper/claim_mode.yaml
configs/tasks/odir_default.yaml
src/retina_screen/tasks.py
src/retina_screen/evaluation.py
src/retina_screen/reporting.py
src/retina_screen/dashboard_app.py

---

## Decision 008 — BRSET and mBRSET Are Future Adapter-Only Integrations

Status: locked

Decision:

BRSET/mBRSET remain supported as future adapter/config/task additions.

Rationale:

They may provide stronger structured systemic labels, device metadata, and smartphone transfer. The architecture should support them without refactoring downstream layers.

Consequences:

- `brset.py` and `mbrset.py` may remain stubs until access.
- Adding BRSET/mBRSET must not require changing `model.py`, `training.py`, `evaluation.py`, `continual.py`, or `dashboard_app.py`.

Files affected:

src/retina_screen/adapters/brset.py
src/retina_screen/adapters/mbrset.py
configs/dataset/brset.yaml
configs/dataset/mbrset.yaml
configs/tasks/brset_default.yaml
configs/tasks/mbrset_default.yaml

---

## Decision 009 — Foundation Backbones Stay Frozen by Default

Status: locked

Decision:

RETFound, DINOv2, ConvNeXt, and ResNet are frozen embedding extractors by default.

Rationale:

Frozen backbones reduce compute, improve reproducibility, allow caching, and avoid unnecessary fine-tuning risk.

Consequences:

- `requires_grad = False` for backbone parameters by default.
- Backbone fine-tuning is only allowed as an explicit ablation.

Files affected:

src/retina_screen/embeddings.py
configs/backbone/\*
src/retina_screen/model.py

---

## Decision 010 — OOD Uses PCA-64 Mahalanobis by Default

Status: locked

Decision:

OOD detection uses PCA-64 followed by Mahalanobis distance.

Rationale:

Full-dimensional covariance can be unstable in 1024-dimensional embedding space. PCA-64 stabilizes OOD estimation.

Consequences:

- PCA basis, mean, inverse covariance, and threshold are cached by backbone/dataset/preprocessing hash.
- Threshold is calibrated on validation distances.
- OOD thresholds cannot be reused across different embedding spaces.

Files affected:

configs/ood/pca64_mahalanobis.yaml
src/retina_screen/embeddings.py
src/retina_screen/continual.py
src/retina_screen/dashboard_app.py

---

## Decision 011 — RFMiD Has Limited Roles

Status: locked

Decision:

RFMiD is used for:

1. TCAV/concept-direction source.
2. Secondary external ocular validation.

RFMiD is not a continual-learning stream.

Rationale:

RFMiD label schema differs enough from ODIR’s eight-category structure that using it as a stream would complicate the continual-learning protocol.

Consequences:

- RFMiD config should declare concept/external-validation roles only.
- Continual-learning configs should not use RFMiD as incoming stream unless a later decision changes this.

Files affected:

src/retina_screen/adapters/rfmid.py
configs/dataset/rfmid.yaml
src/retina_screen/explainability.py
src/retina_screen/continual.py

---

## Decision 012 — Dashboard Is Inference-Only

Status: locked

Decision:

The dashboard must never retrain or update model parameters from user uploads.

Rationale:

User uploads are unlabeled, unverified, and unsafe for live medical AI learning.

Consequences:

- Dashboard may run quality/OOD checks, prediction, explanation, reliability lookup, and consent-based logging.
- Dashboard must not call training/optimizer/continual-learning update functions.
- Continual learning remains offline simulation only.

Files affected:

src/retina_screen/dashboard_app.py
src/retina_screen/continual.py
tests/test_import_boundaries.py

---

## Decision 013 — Evaluation Protocol Must Be Pre-Registered

Status: locked

Decision:

Before headline experiments, freeze:

- metrics,
- subgroup columns,
- mitigation methods,
- external datasets,
- bootstrap iterations,
- sparse subgroup thresholds,
- multiple-comparison correction.

Rationale:

Prevents post-hoc cherry-picking.

Consequences:

- `configs/evaluation/preregistered_protocol.yaml` must not be changed after headline experiments begin unless logged.
- Baseline and mitigation runs must use the same harness.

Files affected:

configs/evaluation/preregistered_protocol.yaml
src/retina_screen/evaluation.py
src/retina_screen/reporting.py

---

## Decision 014 — Static-Analysis Tests Run Early

Status: locked

Decision:

`test_no_dataset_coupling.py` and `test_import_boundaries.py` are Stage 1 tests.

Rationale:

They pass trivially on an empty codebase and catch architecture violations as files are added.

Consequences:

- Do not delay architecture static tests until after MVP.
- Do not weaken these tests to make bad code pass.

Files affected:

tests/test_no_dataset_coupling.py
tests/test_import_boundaries.py
docs/ai_context/06_testing_protocol.md

---

## Decision 015 — Kendall Uncertainty Weighting Is Default Multi-Task Loss Weighting

Status: locked

Decision:

Use Kendall uncertainty weighting as the default multi-task loss weighting method.

Rationale:

The implementation reference defines it as the default and it avoids one task dominating purely due to loss scale.

Consequences:

- GradNorm may remain configurable later.
- Manual task weights may be used only through config and documented experiments.

Files affected:

src/retina_screen/training.py
configs/training/standard.yaml
configs/model/multitask_default.yaml

---

## Decision 016 — Continual-Learning Chunk Size Defaults to 250

Status: locked

Decision:

Continual-learning stream chunk size defaults to 250 images.

Rationale:

The implementation reference defines 250 as a meaningful chunk size for observing forgetting/adaptation without making chunks too small.

Consequences:

- This value belongs in continual-learning config.
- It may be tuned only through config and logged.

Files affected:

configs/training/continual.yaml
src/retina_screen/continual.py

---

## Decision 017 â€” BRSET and mBRSET Become the Intended Primary Scientific Path

Status: locked

Decision:

Access approval for both BRSET and mBRSET was granted on 2026-05-02.

The active planning scenario shifts from ODIR fallback / ODIR-only toward
BRSET + mBRSET once local files are downloaded and inspected.

BRSET becomes the intended primary scientific dataset for the final paper path.
mBRSET becomes the intended cross-device, smartphone / portable-camera
validation dataset and a candidate continual-learning stream.

ODIR remains useful as:

- the Batch/Stage 7 first real-dataset engineering smoke because it is local
  and testable now,
- an auxiliary ocular benchmark,
- an optional cross-population comparison dataset.

ODIR is no longer the intended final primary dataset.

This decision does not authorize untestable BRSET/mBRSET adapter
implementation before local BRSET/mBRSET files are available for inspection.

Rationale:

BRSET and mBRSET are expected to better match the project's scientific goals:
structured clinical metadata, stronger systemic-label support, and real
clinical-camera versus portable/smartphone device shift. ODIR remains valuable
for engineering smoke tests and ocular benchmarking, but its systemic labels
are weak proxies and should not define the final primary scientific narrative.

Consequences:

- Continue ODIR Batch/Stage 7 as the first real-dataset engineering smoke.
- Do not treat ODIR-only results as the final primary paper path.
- Add BRSET/mBRSET only after local files are downloaded and inspected.
- BRSET/mBRSET integration must require adapter/config/task/test additions
  only, not downstream model/training/evaluation/dashboard refactoring.
- Retain ODIR configs; later add BRSET/mBRSET configs instead of replacing
  shared backbone or pipeline configs.

Files affected:

src/retina_screen/adapters/brset.py
src/retina_screen/adapters/mbrset.py
configs/dataset/brset.yaml
configs/dataset/mbrset.yaml
configs/tasks/brset_default.yaml
configs/tasks/mbrset_default.yaml
docs/mvp_build_order.md
docs/ai_context/03_file_generation_order.md

---

## Decision 018 â€” BRSET/mBRSET Private-Data Handling Rules

Status: locked

Decision:

Raw BRSET and mBRSET files must live only in gitignored/private local
directories.

Use environment variables for local dataset roots:

- `RETINA_SCREEN_BRSET_ROOT`
- `RETINA_SCREEN_MBRSET_ROOT`

Do not commit:

- raw BRSET/mBRSET images,
- raw metadata,
- patient-level manifests,
- patient-level split files,
- embedding caches,
- run outputs containing private sample IDs.

Adapter code, config templates, synthetic fixtures, and documentation may be
public.

Cached embeddings remain local/private unless explicitly reviewed.

Model checkpoints and aggregate evaluation outputs may be publishable only if
they contain no patient-identifying or restricted data.

Rationale:

BRSET/mBRSET access introduces private-data handling obligations. The public
repository should contain reproducible code, templates, and synthetic tests,
but not raw or derived patient-level artifacts.

Consequences:

- BRSET/mBRSET adapters must support environment-variable dataset roots.
- Tests for BRSET/mBRSET must use synthetic fixtures unless guarded by local
  dataset availability.
- Generated patient-level artifacts for BRSET/mBRSET remain local/private by
  default.
- Any decision to publish cached embeddings, checkpoints, or aggregate outputs
  requires explicit review for identifying or restricted content.

Files affected:

src/retina_screen/adapters/brset.py
src/retina_screen/adapters/mbrset.py
configs/dataset/brset.yaml
configs/dataset/mbrset.yaml
configs/tasks/brset_default.yaml
configs/tasks/mbrset_default.yaml
tests/
docs/

---

## Decision 019 — Local Dataset Files Are Now Present

Status: locked

Decision:

As of 2026-05-05, all currently planned local datasets are confirmed present on disk:
ODIR-5K, BRSET, mBRSET, APTOS 2019, IDRiD, Messidor-2, EyePACS DR dataset,
EyePACS-AIROGS-light-V2, RFMiD.

Standalone AIROGS directory is not present; glaucoma lightweight coverage is provided
by EyePACS-AIROGS-light-V2, pending Stage 8G preflight confirmation.

BRSET and mBRSET file presence and structure are confirmed. Adapter implementation
may proceed after a local preflight inspection (Stage 8B).

This resolves the "if access is available" uncertainty in Decision 017.
The "if dataset access is available" framing for BRSET/mBRSET is no longer accurate;
updated framing is "after local preflight and adapter implementation."

Note: Exact dataset variants (e.g., EyePACS dataset variant, RFMiD metadata root layout)
must be confirmed during each dataset's Stage 8G preflight. Do not treat this decision
as a lock on exact dataset structures — only on physical file presence.

Consequences:

- BRSET/mBRSET adapter implementation may begin after Stage 8B preflight.
- Documentation should not say "if BRSET/mBRSET access is available."
- Each external dataset adapter must still begin with a local preflight read-only audit.

Files affected:

docs/dataset_inventory.md
README.md

---

## Decision 020 — Local Data Storage and Privacy Policy Applies to All Datasets

Status: locked

Decision:

The private-data handling policy from Decision 018 (BRSET/mBRSET) extends to all datasets.
No raw dataset files, raw metadata, patient-level manifests, split files with sample IDs,
embedding caches, or run outputs are committed to the repository.

This applies to all datasets including public ones (ODIR-5K, APTOS, IDRiD, Messidor-2,
EyePACS, RFMiD, EyePACS-AIROGS-light-V2) due to size, licensing, and reproducibility constraints.
Public availability does not mean commit-safe.

Allowed to commit: adapter code, config templates, synthetic test fixtures, documentation.
Not allowed to commit: raw images, raw metadata, patient IDs, generated caches/splits/runs/outputs.

data/, cache/, runs/, outputs/ are gitignored.
Root-level stray artifacts (full_df.csv, preprocessed_images/) are also gitignored.

Consequences:

- Every dataset adapter must support an environment-variable override for its dataset root.
- All tests for real datasets must use synthetic fixtures guarded by local availability checks.
- Licensing must be verified from local dataset LICENSE files or official dataset sources before
  any public sharing, paper submission, or model publication.

Files affected:

.gitignore
docs/dataset_inventory.md

---

## Decision 021 — Stage 8 Is Restructured Into Substages 8A–8G

Status: locked

Decision:

The original undifferentiated "Stage 8 — Baseline / Fairness Mitigation" is replaced by
a staged 8A–8G sequence.

Stage 8A: Real foundation backbone integration (DINOv2, ConvNeXt, ResNet-50, RETFound if available).
          Use ODIR --limit 32 for verification only. No full ODIR scientific bake-off.
          No silent mock fallback.
Stage 8B: BRSET local preflight (read-only).
Stage 8C: BRSET adapter/config/tasks/tests + smoke gates. BRSET becomes primary dataset path.
Stage 8D: First real BRSET baseline + head ablations (linear probe, ImageNet baselines).
Stage 8E: Baseline visual diagnostics (preliminary plots from 8D artifacts, no retraining).
Stage 8F: mBRSET preflight + adapter + cross-device validation (may split 8F-1 / 8F-2).
Stage 8G: External dataset preflights/adapters — one dataset per substage (APTOS 2019, IDRiD,
          Messidor-2, EyePACS DR, EyePACS-AIROGS-light-V2, RFMiD; AIROGS if found separately).

After 8G: resume original planner sequence (fairness mitigation, continual learning,
explainability, reporting, dashboard).

Rationale:

Real backbone integration must precede BRSET/mBRSET because backbone choice affects
embedding dimensionality and cache namespace. ODIR is used only for backbone verification
(Stage 8A), not as the primary scientific substrate.

Files affected:

docs/mvp_build_order.md

---

## Decision 022 — Early Visual Diagnostics After First Real BRSET Baseline

Status: locked

Decision:

Baseline visual diagnostics (plots, attention maps, calibration curves) are generated
after the first real BRSET baseline run (Stage 8E), not deferred until the dashboard
or reporting stage. Outputs are explicitly marked preliminary/non-paper-final.

Visual diagnostic scripts must not retrain; they read existing checkpoint and output
artifacts only.

Files affected:

docs/mvp_build_order.md
scripts/07_generate_paper_outputs.py (future)

---

## Decision 023 — macular_edema Added to Canonical Schema and Task Registry

Status: locked

Decision:

macular_edema is added to CanonicalSample as a binary optional label field and to
TASK_REGISTRY as a HIGH-quality binary classification task with allowed_as_headline=True.

BRSET (PhysioNet v1.0.1) provides a direct binary macular_edema column (diabetic macular
edema) graded by ophthalmologists. This is a direct retinal finding, not derived from
dr_grade or any other column. It is a landmark clinical target.

Implementation:
- src/retina_screen/schema.py: macular_edema added to CanonicalSample and CANONICAL_LABEL_FIELDS.
- src/retina_screen/tasks.py: TaskDefinition(name="macular_edema", task_type=BINARY,
  loss=BCE, primary_metric=AUROC, label_quality=HIGH, allowed_as_headline=True).

Constraints:
- macular_edema must not be derived from dr_grade.
- Missing/invalid values must be None, never 0.
- macular_edema is a retinal finding, not a synonym for diabetic_retinopathy severity.

Files affected:

src/retina_screen/schema.py
src/retina_screen/tasks.py

---

## Decision 024 — BRSET Canonical DR Grading Source is DR_SDRG

Status: locked

Decision:

DR_SDRG is used as the canonical dr_grade source for BRSET. DR_ICDR is retained in
get_dataset_audit() audit metadata only and is not exposed through the canonical schema.

Both DR_SDRG and DR_ICDR are 0-4 ordinal grades present in BRSET. Exact disagreement
between the two systems was recomputed from local data: 327 of 16,266 rows (2.01%).
Grade-level analysis shows ICDR reclassifies SDRG grades 1 and 3 heavily into grade 2,
resulting in an anomalous ICDR distribution (grade 3=78 < grade 1=158 < grade 2=451).

This decision is a project canonicalization choice for reproducibility. It is not a
claim of medical superiority of DR_SDRG over DR_ICDR. DR_ICDR is preserved in the
dataset audit output for potential later ablation.

The adapter sets dr_grade_source_scheme = "DR_SDRG" and dr_grade_mapping_confidence = EXACT
for all BRSET samples.

Files affected:

src/retina_screen/adapters/brset.py
configs/dataset/brset.yaml
configs/tasks/brset_default.yaml

---

## Decision 025 — BRSET patient_sex and exam_eye Encodings Confirmed

Status: locked

Decision:

From BRSET PhysioNet v1.0.1 documentation:
- patient_sex: 1 = Male, 2 = Female.
- exam_eye: 1 = right eye, 2 = left eye.

The adapter maps these to canonical Sex.MALE / Sex.FEMALE / Sex.UNKNOWN and
EyeLaterality.RIGHT / EyeLaterality.LEFT respectively. Unexpected values map to
Sex.UNKNOWN or None; they are never guessed.

FeaturePolicy rules still apply:
- sex must not be used as a feature to predict sex.
- sex may be used for subgroup / fairness evaluation if the evaluation pipeline
  supports it safely.

Files affected:

src/retina_screen/adapters/brset.py

---

## Decision 026 — BRSET Stage 8C Adapter Locked Defaults

Status: locked

Decision:

The following BRSET-specific adapter defaults are locked for Stage 8C:

1. increased_cup_disc is NOT mapped to glaucoma. It is an anatomical marker (optic disc
   enlargement), not a confirmed glaucoma diagnosis. glaucoma = None for all BRSET samples.
   Glaucoma is listed as unsupported in configs/tasks/brset_default.yaml.

2. cataract is absent from BRSET metadata. cataract = None for all BRSET samples.
   cataract is listed as unsupported in configs/tasks/brset_default.yaml.

3. hypertensive_retinopathy is a direct fundoscopic retinal finding graded by
   ophthalmologists. It is NOT systemic hypertension. Do not equate the two.

4. diabetes is a clinical/medical-record label. PROXY quality. Do not overclaim as
   a retinal-derived or direct systemic diagnosis.

5. comorbidities is free text (213 unique combinations, 50% missing). It is dropped
   entirely and must not be exposed raw through canonical samples, manifests, logs,
   or public outputs.

6. Canonical IDs use deterministic pseudonymous format: patient_id = brset_pNNNNNN,
   sample_id = brset_sNNNNNN, derived from sorted-index mapping of native IDs.
   Raw native patient_id and image_id values are never embedded in canonical IDs,
   split files, cache manifests, or cache filenames. Cross-dataset collision is
   prevented by the brset_p/brset_s namespace.

7. insuline (insulin_use) and diabetes_time_y (diabetes_duration_years) are deferred
   to Stage 8D+. Both fields have >88% missingness. They are set to None in Stage 8C.

8. BRSET supported tasks for Stage 8C: dr_grade, macular_edema, hypertensive_retinopathy,
   amd, drusen, other_ocular, diabetes.
   pathological_myopia is DEFERRED: BRSET myopic_fundus is a fundoscopic/anatomical
   proxy (indirect finding). The global TASK_REGISTRY marks pathological_myopia as
   HIGH/headline but per-dataset label-quality override is not yet enforced downstream.
   Setting pathological_myopia as supported would risk overclaiming. Reconsider in
   Stage 8D+ once a per-dataset override mechanism exists.

Files affected:

src/retina_screen/adapters/brset.py
configs/dataset/brset.yaml
configs/tasks/brset_default.yaml

---

## Decision 027 — Stage 8D-3 Standard Training Recipe: Unweighted Corrected Baseline; Class-Weighted Run Retained as Sensitivity Variant Only

Status: locked

Decision:

The corrected unweighted ResNet-50 training run (Stage 8D-2, corrected) is the standard and
default training recipe for all Stage 8D-3 backbone comparison runs (DINOv2, ConvNeXt, RETFound).
`class_weighting_enabled: false` in `configs/training/standard.yaml` is the default for all
Stage 8D-3 configs.

The class-weighted ResNet-50 rerun (Stage 8D-2, class-weighted sensitivity experiment) is
retained as a documented secondary sensitivity result. Its verdict is PASS_WITH_WARNINGS.
It is not the primary training recipe and is not used as the Stage 8D-3 baseline.

Rationale:

Two full BRSET ResNet-50 training runs were completed under the corrected training
infrastructure (mini-batch 256, AdamW 1e-4, cosine+warmup LR schedule, early stopping on
val macro-AUROC, patience=10):

  Run A — Corrected unweighted baseline (class_weighting_enabled=false):
    Run dir: runs/train/brset_20260512_191830/ (best-val epoch 33)
    Test metrics: dr_grade balanced_accuracy=0.384, macro_f1=0.388;
    macular_edema AUROC=0.956, hypertensive_retinopathy AUROC=0.691,
    amd AUROC=0.875, drusen AUROC=0.729, other_ocular AUROC=0.812,
    diabetes AUROC=0.847.
    dr_grade classes 1 (mild) and 2 (moderate): 0 predictions each.
    Verdict: PASS.
    Report: outputs/stage8d2_full_resnet50_multitask/2026-05-12T19-50-00/corrected_training_report.md

  Run B — Class-weighted sensitivity experiment (class_weighting_enabled=true,
    max_class_weight=10.0, applied to ALL classification tasks):
    Config: configs/experiment/stage8d2_brset_resnet50_full_multitask_classweighted.yaml
    Run dir: runs/train/brset_20260512_235438/ (best-val epoch 24)
    Test metrics: dr_grade balanced_accuracy=0.495 (+0.111 over Run A),
    macro_f1=0.366 (−0.022 vs Run A); class 1 recall=0.364 (up from 0),
    class 2: 0 TP despite 3 predictions; class 1 precision=0.109 (98 FP / 12 TP);
    hypertensive_retinopathy AUROC=0.770 (+0.079); binary macro-AUROC delta=+0.011.
    Verdict: PASS_WITH_WARNINGS.
    Warnings: macro_f1 decreased (−0.022, below +0.03 improvement threshold); class 2
    unrecovered (0 TP); class 1 precision very low (10.9%).
    Report: outputs/stage8d2_full_resnet50_multitask/2026-05-12T20-30-00/class_weighted_training_report.md

Neither run is strictly superior. Run A has higher macro_f1 and no spurious DR predictions.
Run B has higher balanced_accuracy and partial class 1 recall, but introduces a high
false-alarm rate for mild DR and does not meet both improvement thresholds simultaneously
(balanced_accuracy ≥ +0.03 AND macro_f1 ≥ +0.03). Per the evaluation criteria in the
class-weighted report (§12), this maps to KEEP_BOTH_AS_VARIANTS, not UPGRADE_DEFAULT.

For Stage 8D-3 backbone comparison runs, the unweighted recipe is chosen as the default
because:
  1. Run A is the accepted corrected baseline. Using it for all backbone runs ensures that
     any performance delta between backbones is attributable to the backbone, not the
     training recipe.
  2. Run A's macro_f1 is higher (0.388 vs 0.366), making it a more balanced dr_grade metric.
  3. Class-weighted results carry higher false-alarm risk for rare DR classes, which is
     undesirable in a systematic backbone comparison.
  4. The scientific goal of Stage 8D-3 is backbone comparison, not imbalance mitigation.
     Imbalance mitigation experiments are deferred to a dedicated ablation.

Run B provides a documented data point on the effect of class weighting for the BRSET
imbalance regime. It is available for reference and may be used as a secondary result in
reporting.

Reference documents:

  docs/implementation_reference.md §4 — Training hyperparameters (AdamW, cosine+warmup,
    max 100 epochs, early stopping, batch 256, gradient clip 1.0).
  docs/implementation_reference.md §13 — Published baselines (Nakayama DR AUC ~0.95,
    Diabetes AUC ~0.87; Khan Hypertension AUC ~0.79) used to assess whether corrected
    baseline is in a meaningful range.
  docs/mvp_build_order.md — Stage 8D→8E→8F→8G ordering. Decision 022 specifies visual
    diagnostics after the first BRSET baseline.
  Decision 015 — Kendall uncertainty weighting is the default multi-task loss. Class
    weighting is an additive option on top of Kendall weighting, not a replacement.
  Decision 022 — Early visual diagnostics after Stage 8D first BRSET baseline; backbone
    comparison (Stage 8D-3) follows after.
  configs/training/standard.yaml — Authoritative hyperparameter source; class_weighting_enabled
    defaults to false.
  configs/experiment/stage8d2_brset_resnet50_full_multitask.yaml — Unweighted Stage 8D-2
    experiment config (the Stage 8D-3 template).
  configs/experiment/stage8d2_brset_resnet50_full_multitask_classweighted.yaml — Class-weighted
    sensitivity config (retained, not promoted to default).
  outputs/splits/brset/20260507_143232/split_audit.json — Patient-level split integrity
    verified (valid=true, no overlap_pairs, test=1623 samples, 854 patients).

Consequences:

- All Stage 8D-3 experiment configs (DINOv2, ConvNeXt, RETFound) use
  class_weighting_enabled: false unless an explicit ablation decision overrides this.
- class_weighting_enabled: true is NOT the default for any future backbone run without
  a new explicit locked decision.
- Do not re-run class-weighted variants unless a new explicit decision is made.
- Do not implement dr_grade-only weighting, focal loss, oversampling, threshold tuning,
  new loss terms, new heads, or architecture changes to address dr_grade imbalance without
  a new locked decision. These are deferred.
- The dr_grade imbalance (93.5% class-0 in BRSET test split) is a dataset characteristic,
  not a pipeline bug. It is expected to partially resolve with a retinal-adapted backbone
  (RETFound — Stage 8E scope).

Files affected:

configs/training/standard.yaml
configs/experiment/stage8d2_brset_resnet50_full_multitask.yaml
configs/experiment/stage8d2_brset_resnet50_full_multitask_classweighted.yaml
docs/decisions.md

---

## Decision 028 — Stage 8D-3D DINOv2-Base Post-Run Issue Handling and Pre-DINOv2-Large Storage Guard

Status: locked

Decision:

This decision records six issues surfaced during Stage 8D-3D strict review and their handling.
It does not authorise any result-optimisation, re-extraction, or new training work.

1. W1 — Embedding cache storage-view compaction (patched before DINOv2-Large):

   Root cause: `backbone(tensor).squeeze(0).cpu()` in `src/retina_screen/embeddings.py`
   saved the CLS token as a PyTorch view into the full ViT token-sequence backing storage.
   For DINOv2-Base ViT-B/14 (257 tokens) this inflated each .pt file to ~773 KB instead of
   ~3 KB; total cache grew to 12.875 GB instead of ~50 MB.  Storage ratio = 257x.

   Fix applied in `src/retina_screen/embeddings.py`:
   - Added private helper `_compact_embedding(raw)` that calls
     `.detach().cpu().clone().contiguous()` to break the storage-view link before saving.
   - Extraction line updated to call `_compact_embedding(backbone(tensor).squeeze(0))`.
   - Regression tests added in `tests/test_embedding_storage.py`.

   DINOv2-Large extraction is blocked until this patch and regression tests are committed
   and pass (enforced by this decision).

   Existing DINOv2-Base cache (12.875 GB) is not re-extracted or overwritten — strict review
   confirmed all embeddings load with shape (768,), dtype float32, no NaN/Inf, correct
   provenance and metrics.  The cache remains valid for reuse via its existing manifest.

   Any future DINOv2-Base cache compaction is a separate maintenance step requiring explicit
   user approval and is not part of this patch.

2. Existing DINOv2-Base cache and results accepted:

   Both Stage 8D-3D runs accepted after PHASE A strict review (verdict PASS_WITH_WARNINGS,
   no critical issue, no FAIL gate).  Results are preliminary, internal, and non-paper-final
   (final_test_result=false in both evaluation_summary.json files).

   DINOv2-Base MultiTaskHead test binary macro-AUROC = 0.8745 (best in Stage 8D-3 matrix).
   DINOv2-Base LinearProbeHead test binary macro-AUROC = 0.8506.

3. dr_grade classes 1/2 limitation not patched during Stage 8D-3:

   Classes 1 (mild DR) and 2 (moderate DR) receive zero or near-zero predictions under the
   locked unweighted frozen-embedding protocol.  This is a dataset imbalance characteristic
   (93.5% class-0 in BRSET test split), not a pipeline bug.  Do not add class weighting,
   focal loss, oversampling, threshold tuning, ordinal loss, or any new loss terms during
   Stage 8D-3.  These are deferred to a dedicated ablation after matrix lock.

4. LinearProbe full-epoch / no-early-stopping behaviour not patched during Stage 8D-3:

   The LP head ran all 100 epochs without early stopping (best_epoch=99).  This is expected
   behaviour for a shallower head under the locked recipe.  Increasing max_epochs or tuning
   patience during the active matrix is forbidden because it would break locked recipe
   comparability (Decision 027).  Optional extended-epoch sensitivity can be considered only
   after matrix lock with a new explicit locked decision.

5. Unicode arrow logging cosmetic issue deferred:

   Windows console may render "→" as garbled characters.  This is a cosmetic display artifact
   only; it does not affect log files, metrics, or results.  Not patched during Stage 8D-3.
   Optional later cleanup only; do not refactor logging broadly.

6. RETFound remains deferred:

   RETFound weights and access remain unavailable.  RETFound is not part of the active
   Stage 8D-3 backbone comparison matrix.  No timeline is set.  This is not a blocker for
   accepting DINOv2-Base, ConvNeXt-Base, or ResNet-50 results.

Consequences:

- DINOv2-Large extraction MUST NOT begin until the W1 patch and regression tests in
  `tests/test_embedding_storage.py` are committed and all tests pass.
- Existing DINOv2-Base cache (12.875 GB) is valid for reuse and must not be deleted or
  overwritten without explicit user approval.
- `class_weighting_enabled: false` remains the Stage 8D-3 default (Decision 027).
- Do not implement any result-optimisation changes without a new explicit locked decision.

Files affected:

src/retina_screen/embeddings.py
tests/test_embedding_storage.py
docs/decisions.md

---

## Decision 029 — RETFound-Green Pretraining-Overlap Policy and Matched-Protocol Selection

Status: locked

Decision:

This decision records the RETFound-Green B1 verification outcome and locks four policies
governing RETFound-Green's use in the project. It is recorded as part of Stage 8D-3.5 B1
(RETFound-Green partial closeout, 2026-05-24). It does not authorise extraction, training,
or evaluation.

1. Pretraining corpus:

   RETFound-Green (Engelmann & Bernabeu, Nature Comms 2025, arXiv 2405.00117) is
   pretrained on ~75,000 publicly available retinal images from three sources:
   - AIROGS (53,327 images)
   - DDR (full dataset)
   - ODIR-2019 (full dataset)

   BRSET, mBRSET, IDRiD, Messidor/Messidor-2, APTOS/APTOS-2019, and RFMiD are not present
   in the RETFound-Green pretraining corpus. RETFound-Green is not blocked for BRSET primary
   matrix inclusion on contamination grounds.

2. ODIR pretraining caveat and external-validation exclusion:

   RETFound-Green pretraining includes ODIR-2019. This project uses ODIR-5K, which is the
   same dataset family (same source, same ophthalmological disease labelling scheme,
   overlapping image population). These are treated as equivalent for contamination purposes.

   RETFound-Green MUST NOT be used as a frozen feature extractor for external validation on
   ODIR in any form — ODIR-2019, ODIR-5K, or any ODIR-derived split — because doing so
   would constitute pretraining/evaluation overlap and invalidate any generalisation claim.

   If AIROGS-based evaluation is later adopted by the project, the same exclusion applies
   (RETFound-Green pretraining includes AIROGS).

   DDR is not currently a project evaluation dataset. Do not claim DDR is a project dataset
   unless project documentation explicitly says so.

3. B2 matched-224 protocol (subdecision resolved):

   The Stage 8D-3 matrix uses 224px input with the standard default_224 preprocessing
   pipeline. RETFound-Green B2 will use the same matched-224/default_224 protocol for
   comparability with ResNet-50, ConvNeXt-Base, DINOv2-Base, and DINOv2-Large.

   When loading 392×392-trained RETFound-Green weights at 224×224, timm automatically
   interpolates positional embeddings. This is expected and correct behaviour for
   cross-resolution inference, but it means B2 matched-224 results are not directly
   comparable to the published native-392/avg-pool numbers.

   Native-392/avg-pool protocol is conditionally deferred. It may be run as a secondary
   protocol row only after matched-224 B2 is complete and only with explicit user decision.
   Running native-392 by default is forbidden.

4. retfound.yaml mismatch — B2 must create a new config:

   The existing configs/backbone/retfound.yaml has embedding_dim=1024 and model_type=retfound.
   This corresponds to a ViT-Large variant (original RETFound or RETFound-MEH) and must not
   be reused or overwritten for RETFound-Green.

   B2 must create a new config file, e.g. configs/backbone/retfound_green_matched224.yaml,
   with embedding_dim=384, reflecting RETFound-Green's ViT-Small architecture.

Consequences:

- RETFound-Green is approved for B2 matched-224 BRSET matrix execution (B2: GO).
- B2 must add timm as a project dependency (not currently installed).
- B2 must implement a timm loader branch in src/retina_screen/embeddings.py.
- B2 must create configs/backbone/retfound_green_matched224.yaml with embedding_dim=384.
- RETFound-Green must not be externally validated on ODIR in any form.
- RETFound-Green native-392/avg-pool is deferred and must not be run by default.
- This decision does not affect DINOv3-Large, FLAIR, DINOv2-Giant, or RETFound-MEH.

Files referenced:

docs/verifications/B1_backbone_candidates_verification.md
docs/paper_framing_and_findings.md (F10)
configs/backbone/retfound.yaml (read-only reference; do not modify)
src/retina_screen/embeddings.py (B2 will add timm loader branch)

---

# Open Decisions

These decisions are not locked yet. Ask before implementing if they become relevant.

## Open 001 — Dashboard Framework

Options:

- Streamlit
- Gradio

Default leaning:

- Streamlit for prototype unless changed later.

Affected files:

src/retina_screen/dashboard_app.py
scripts/08_launch_dashboard.py
requirements.txt

## Open 002 — Exact RFMiD Concept Set for TCAV

Decision needed:

Which RFMiD labels should be used as TCAV concept directions.

Affected files:

src/retina_screen/adapters/rfmid.py
src/retina_screen/explainability.py
configs/dataset/rfmid.yaml

## Open 003 — Clinical Co-Author / Clinical Language Strength

Decision needed:

Whether a clinical co-author is available to strengthen clinical framing.

Default:

If no clinical co-author exists, keep all systemic claims conservative.

Affected files:

configs/paper/claim_mode.yaml
src/retina_screen/reporting.py
src/retina_screen/dashboard_app.py

## Open 004 — Include RetiZero or DINORET

Decision needed:

Whether weights are accessible and worth adding.

Default:

Do not delay MVP. Use DINOv2 + RETFound + ConvNeXt/ResNet baseline first.

Affected files:

configs/backbone/\*
src/retina_screen/embeddings.py

## Open 005 — Full Fine-Tuning / Backbone LoRA Ablation

Decision needed:

Whether to run a later ablation that adds LoRA to the foundation backbone.

Default:

Do not include in MVP. Head-only training first.

Affected files:

src/retina_screen/embeddings.py
src/retina_screen/model.py
src/retina_screen/continual.py
configs/training/continual.yaml
