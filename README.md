# FROST: Frozen Representations for Ocular Screening and Triage in Referable Diabetic Retinopathy

FROST is a reproducible Python research system for benchmarking frozen visual representations on retinal fundus images and deploying the empirically selected representation as a low-compute referable diabetic retinopathy (DR) triage demonstrator.

The repository implements the complete experimental path used in the study: patient-level data partitioning, deterministic image preprocessing, frozen-backbone feature extraction, provenance-aware embedding caches, lightweight head training, held-out evaluation, and patient-clustered statistical comparison.

**Live research demonstrator:** [FROST on Hugging Face Spaces](https://nikunj985-frost-referable-dr.hf.space/)

> **Research use only.** FROST is not a diagnostic medical device. Its output is a screening-oriented research result and requires independent clinical assessment.

## Study at a Glance

| Item | Reported study |
|---|---|
| Dataset | BRSET |
| Cohort | 16,266 fundus images from 8,524 patients |
| Split | Patient-level 60/15/15/10 train/validation/reliability/test |
| Test set | 1,623 images from 854 patients |
| Patient overlap across splits | 0 |
| Comparison | 7 backbone/protocol rows x 2 prediction heads = 14 experimental cells |
| Primary endpoint | Referable DR, defined as DR grade >= 2 |
| Secondary endpoints | Six-condition macro-AUROC and 5-class DR balanced accuracy |
| Statistical analysis | 2,000-resample patient-clustered bootstrap with paired comparisons |
| Best referable-DR result | RETFound-Green native-392 + multi-task head: AUROC 0.985 [0.964-0.997] |
| Deployment model | RETFound-Green native-392 + locked multi-task head |

## What Is Implemented

The paper-facing system is focused on the work that was actually completed and evaluated:

- BRSET ingestion through a dataset adapter and canonical sample schema.
- Patient-level splitting with explicit zero-overlap auditing.
- Deterministic matched-224 preprocessing for all backbones.
- Native-392 RETFound-Green extraction as a controlled protocol comparison.
- Frozen feature extraction for supervised CNNs, general self-supervised ViTs, and a retina-specific self-supervised model.
- Reusable embedding caches with manifests, preprocessing identity, and extraction provenance.
- A per-task linear probe and a shared multi-task prediction head.
- Task-masked losses for incomplete labels.
- AdamW training, warmup, cosine scheduling, gradient clipping, validation-based early stopping, and best-checkpoint restoration.
- Held-out test evaluation for referable DR, six binary conditions, and 5-class DR grading.
- Patient-clustered bootstrap confidence intervals and paired delta testing.
- A hosted single-image research demonstrator using the empirically selected parameter-efficient model.

Earlier plans for external validation, fairness mitigation, continual learning, OOD gating, online updating, and saliency-based explainability are **not part of the reported study and are not claimed here**.

## Experimental Pipeline
<img width="1349" height="614" alt="image" src="https://github.com/user-attachments/assets/9e692d8a-afaf-49a7-9b58-c46c5e2aa35b" />

## Compared Representations

Seven backbone/protocol rows were evaluated. RETFound-Green appears twice because matched and native extraction were treated as an explicit experimental axis.

| Family | Backbone | Protocol | Embedding dimension |
|---|---|---:|---:|
| Supervised CNN | ResNet-50 | 224 matched | 2,048 |
| Supervised CNN | ConvNeXt-Base | 224 matched | 1,024 |
| General SSL ViT | DINOv2-Base | 224 matched | 768 |
| General SSL ViT | DINOv2-Large | 224 matched | 1,024 |
| General SSL ViT | DINOv3-Large | 224 matched | 1,024 |
| Retina-specific SSL ViT | RETFound-Green | 224 matched | 384 |
| Retina-specific SSL ViT | RETFound-Green | 392 native | 384 |

All backbone parameters remained frozen. Only the downstream prediction heads were trained.

### Prediction heads

**Linear probe (LP)**

A single linear layer is fitted per task on top of the frozen embedding. This measures the direct linear separability of each representation.

**Multi-task head (MT)**

A shared nonlinear trunk is trained jointly across the available tasks, followed by task-specific output layers. Cross-entropy is used for 5-class DR grading, binary cross-entropy is used for the six binary conditions, and per-task losses are combined with Kendall uncertainty weighting.

## Endpoints

### Primary endpoint: referable DR

The model predicts a five-class DR distribution. The referable-DR score is computed as:

```text
P(referable DR) = P(grade 2) + P(grade 3) + P(grade 4)
```

This directly represents the study's triage question: whether an image crosses the moderate-or-worse referral boundary.

### Multi-condition panel

The secondary binary panel contains:

- macular edema,
- hypertensive retinopathy,
- age-related macular degeneration (AMD),
- drusen,
- other ocular findings,
- diabetes as a record-derived proxy label.

The six-task macro-AUROC excludes the five-class DR output. Hypertensive retinopathy is an ophthalmological grading, not a systemic blood-pressure measurement, and the diabetes flag is treated as a screening proxy rather than a diagnosis from the image alone.

## Training and Evaluation Protocol

A single pre-specified recipe was applied across all 14 cells:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 256 |
| Warmup | 5 epochs |
| Scheduler | Cosine annealing |
| Maximum epochs | 100 |
| Early-stopping patience | 10 |
| Gradient clipping | 1.0 |
| Checkpoint criterion | Validation six-task macro-AUROC |
| Random seed | 42 |
| Class weighting in primary matrix | Disabled |
| Backbone fine-tuning | Disabled |

The validation split was used for model selection. The reliability split was held out for work outside the reported paper. Final metrics were calculated once on the untouched test partition.

For uncertainty estimation, all images belonging to the same patient were resampled together. Each metric used 2,000 patient-clustered bootstrap replicates, with the 2.5th and 97.5th percentiles reported as the 95% confidence interval.

## Main Results

### Complete backbone x head matrix

| Backbone and protocol | Referable-DR AUROC, MT | Referable-DR AUROC, LP | Six-task macro-AUROC, MT | Six-task macro-AUROC, LP | 5-class balanced accuracy, MT | 5-class balanced accuracy, LP |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50, 224 | 0.942 [0.901-0.974] | 0.920 [0.885-0.949] | 0.818 [0.790-0.845] | 0.774 [0.742-0.804] | 0.384 | 0.239 |
| ConvNeXt-Base, 224 | 0.975 [0.954-0.989] | 0.943 [0.913-0.969] | 0.846 [0.821-0.871] | 0.800 [0.771-0.829] | 0.400 | 0.307 |
| DINOv2-Base, 224 | 0.973 [0.951-0.989] | 0.958 [0.931-0.981] | 0.874 [0.855-0.892] | 0.851 [0.830-0.871] | 0.436 | 0.418 |
| DINOv2-Large, 224 | 0.979 [0.956-0.993] | 0.974 [0.958-0.987] | 0.880 [0.854-0.904] | 0.847 [0.819-0.875] | 0.474 | 0.423 |
| RETFound-Green, 224 matched | 0.978 [0.959-0.991] | 0.968 [0.950-0.984] | 0.843 PE | 0.806 PE | 0.399 | 0.355 |
| DINOv3-Large, 224 | 0.977 [0.956-0.993] | 0.962 [0.938-0.980] | 0.875 PE | 0.829 PE | 0.419 | 0.347 |
| RETFound-Green, 392 native | **0.985 [0.964-0.997]** | 0.968 [0.951-0.982] | 0.875 PE | 0.819 PE | 0.463 | 0.332 |

`PE` indicates a point estimate where the final manuscript does not report a confidence interval for that macro-AUROC cell.

### Primary Referable-DR performance across the matrix
<img width="1082" height="568" alt="image" src="https://github.com/user-attachments/assets/cde9bd43-f705-4d6a-ba5b-95ef18d770f1" />


### Findings supported by the comparison

1. **The largest model was not automatically the best deployment choice.** RETFound-Green native-392 used approximately 22M parameters and 384-dimensional embeddings, yet its referable-DR result was not separable from DINOv2-Large or DINOv3-Large. It had a supported +0.012 AUROC advantage over DINOv2-Base.

2. **Pretraining paradigm mattered more than within-family scaling for the reported macro endpoint.** ConvNeXt-Base to DINOv2-Base improved macro-AUROC by +0.029 [0.010-0.049], while DINOv2-Base to DINOv2-Large added +0.006 [-0.015-0.025] and was not separable.

3. **The multi-task head improved the six-task panel.** MT exceeded LP in every row. For the four backbones with paired macro-AUROC intervals, the gains were supported and ranged from +0.024 to +0.046.

4. **The head benefit was concentrated on sparse targets.** The largest gains appeared on AMD, the rarest binary condition: +0.131 [0.057-0.215] for ResNet-50 and +0.109 [0.015-0.212] for DINOv2-Large.

5. **Extraction protocol was a real performance variable.** Moving RETFound-Green from matched-224 to native-392 changed the six-task macro-AUROC by +0.032 in point estimates. The referable-DR AUROC difference was not separable, but referable-DR AUPRC improved by +0.048 [0.003-0.096].

6. **Binary triage was substantially more reliable than fine-grained grading.** Referable-DR discrimination was strong, while five-class balanced accuracy remained limited because grade 0 represented 93.5% of the test images and mild/moderate grades were extremely sparse.

## FROST Research Demonstrator

The hosted FROST demonstrator uses the deployment choice supported by the comparison:

```text
Uploaded fundus image
    -> native 392 preprocessing
    -> frozen RETFound-Green backbone
    -> 384-dimensional embedding
    -> locked multi-task head
    -> DR grade probability distribution
    -> probability mass on grades 2, 3, and 4
    -> validation-selected threshold
    -> research triage result
```

The interface:

- accepts a professionally acquired JPEG or PNG fundus image,
- processes the image in memory,
- displays image and preprocessing information,
- reports the referable-DR research score and fixed threshold,
- visualizes the five DR-grade outputs,
- shows the exact inference sequence used to produce the triage decision,
- runs on commodity CPU hardware in under one second after model loading.

The fixed validation-selected threshold is approximately `0.0444`. At this operating point, the reported sensitivity was `0.973` and specificity was `0.953` on the study's internal BRSET evaluation. The displayed score is an operating-point triage score, not a calibrated probability.

The hosted interface is a companion deployment artifact. The reproduction workflow below covers the research repository's feature-extraction, training, and evaluation pipeline.

<img width="1160" height="810" alt="image" src="https://github.com/user-attachments/assets/5b5cfc98-1bc8-47c7-8231-b6299ba734fa" />

<img width="1132" height="679" alt="image" src="https://github.com/user-attachments/assets/2b0391e4-1ded-47ab-97d8-8caeaea89ff2" />

<img width="294" height="244" alt="image" src="https://github.com/user-attachments/assets/97486bbd-9056-4569-81e8-44c4711df09d" />



## Engineering Design

The repository uses a config-driven modular architecture rather than notebook-only experiments.

Key engineering properties include:

- dataset-specific parsing isolated behind adapters,
- canonical downstream sample and task contracts,
- patient-leakage checks before training,
- deterministic backbone-specific preprocessing,
- one-image backbone verification before full extraction,
- frozen model loading with explicit embedding-dimension validation,
- cache namespaces keyed by backbone, dataset, and preprocessing identity,
- embedding manifests and extraction provenance,
- reusable caches shared by MT and LP runs,
- task masks for missing labels,
- resolved configuration and run metadata saved with each training run,
- best-validation and last-epoch checkpoints stored separately,
- head-type-safe checkpoint reconstruction during evaluation,
- sparse and single-class metric safeguards,
- machine-readable JSON evaluation artifacts,
- unit, integration, and architecture-boundary tests.

## Installation

FROST supports Python 3.10-3.12.

```bash
git clone https://github.com/Solace985/FROST.git
cd FROST

python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the research dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current `requirements.txt` installs the package in editable mode with the development, PyTorch, and visualization extras.

## BRSET Data Setup

BRSET is distributed through PhysioNet under credentialed access and is not included in this repository:

[BRSET v1.0.1 on PhysioNet](https://physionet.org/content/brazilian-ophthalmological/1.0.1/)

Keep raw images, metadata, split manifests containing identifiers, embedding caches, checkpoints, and generated run artifacts outside version control.

Configure the local dataset root using the project configuration or environment variable:

```bash
# Linux/macOS
export RETINA_SCREEN_BRSET_ROOT=/absolute/path/to/brset
```

```powershell
# Windows PowerShell
$env:RETINA_SCREEN_BRSET_ROOT = "C:\path\to\brset"
```

## Running the Research Pipeline

Experiment behavior is controlled through YAML files in `configs/`. Use the exact paper-facing configuration for the backbone, protocol, and head being reproduced.

### 1. Run the test suite

```bash
pytest
```

### 2. Create or verify the patient-level split

```bash
python scripts/01_make_splits.py --config <experiment-config.yaml>
```

### 3. Verify one image through the selected backbone

```bash
python scripts/02_verify_backbone_one_image.py --config <experiment-config.yaml>
```

This gate verifies model loading, preprocessing, frozen status, output dimension, and cache compatibility before full extraction.

### 4. Extract and cache embeddings

```bash
python scripts/03_extract_embeddings.py --config <experiment-config.yaml>
```

### 5. Train the selected head

```bash
python scripts/04_train.py --config <experiment-config.yaml>
```

### 6. Evaluate the best-validation checkpoint

```bash
python scripts/05_evaluate.py \
  --config <experiment-config.yaml> \
  --run-dir runs/train/<run-id>
```

The two head types for a backbone must resolve to the same embedding cache when their backbone and preprocessing protocol are identical.

## Generated Artifacts

A complete run produces machine-readable provenance and evaluation outputs similar to:

```text
cache/embeddings/<backbone>/<dataset>/<preprocessing-hash>/
  manifest.csv
  cache_provenance.json
  <sample-id>.pt

runs/train/<run-id>/
  resolved_config.yaml
  run_metadata.json
  train_log.csv
  model_checkpoint.pt
  model_last.pt

outputs/evaluation/<evaluation-id>/
  evaluation_summary.json
  overall_metrics.json
  subgroup_metrics.json
  diagnostics.json
```

Generated data and model artifacts are intentionally excluded from Git.

## Scope and Limitations

- The reported benchmark is an internal BRSET evaluation; no external population or cross-device validation is claimed.
- The matrix uses one fixed random seed and one shared training recipe to isolate representation and head effects.
- The test set contains only 73 referable-DR images, and several secondary conditions contain fewer than 35 positives.
- Patient-clustered confidence intervals reduce false precision but do not remove uncertainty caused by sparse outcomes.
- The primary matrix does not use class weighting, oversampling, focal loss, or threshold tuning per backbone.
- Fine-grained five-class DR grading remains weak under the highly imbalanced class distribution.
- The FROST demonstrator is intended for professionally acquired fundus photographs and has only been validated on the internal BRSET distribution.
- Demonstrator output must not be interpreted as a diagnosis or as a calibrated probability of disease.

## Citation

```bibtex
@misc{chauhan2026frost,
  title  = {FROST: Frozen Representations for Ocular Screening and Triage in Referable Diabetic Retinopathy},
  author = {Chauhan, Ritu},
  year   = {2026},
  note   = {Research manuscript and software repository}
}
```

## License

A project license has not yet been specified. Add a `LICENSE` file before redistribution or reuse outside the terms explicitly granted by the repository owner and upstream model/dataset providers.
