# FROST — Frozen Representation for Ocular Screening and Triage

**FROST Referable Diabetic Retinopathy Research Demonstrator**

FROST is a **locally-hosted research demonstrator** that reproduces the study's
exact referable-diabetic-retinopathy (referable-DR) pipeline on a single
uploaded colour fundus photograph. The name expands to **F**rozen
**R**epresentation for **O**cular **S**creening and **T**riage (for referable
diabetic retinopathy).

> **This is a research demonstrator. It is NOT a medical device, is NOT
> validated for clinical decisions, and requires independent clinical
> assessment.** This statement appears in exactly two designed places in the
> interface (the header notice and the About / study-basis section) and once
> here.

---

## What the demonstrator does

It runs the paper's selected deployment candidate end-to-end on one image:

```
uploaded image
  -> native-392 preprocessing  (Resize 392 -> CenterCrop 392 -> ToTensor -> ImageNet normalize)
  -> frozen RETFound-Green backbone  (timm vit_small_patch14_reg4_dinov2, average pooling)
  -> 384-dimensional embedding
  -> locked MultiTaskHead          (the accepted native-392 multi-task checkpoint)
  -> 5-class DR-grade logits
  -> softmax
  -> referable-DR score = P(grade >= 2) = softmax[2] + softmax[3] + softmax[4]
  -> fixed validation-derived operating-point threshold
  -> REFERABLE / NOT REFERABLE  research-triage result
```

The selected deployment candidate is **RETFound-Green native-392 +
MultiTaskHead**. It reproduces the study pipeline exactly; it does **not**
substitute any other backbone (matched-224, DINOv2, DINOv3, ResNet, ConvNeXt),
any other head (linear probe, a new binary head), or any mock/fallback model.

## What the demonstrator does NOT do

- It does **not** train, fine-tune, retrain, or update any model from uploads.
- It does **not** extract or cache embeddings into the pipeline cache.
- It does **not** perform continual learning, LoRA, online calibration, or
  adaptive thresholding.
- It does **not** persist uploads, log filenames, or transmit data externally.
- It does **not** return a five-class grade label, raw embeddings, or systemic /
  other-ocular task outputs.
- It does **not** make a diagnosis or a calibrated-probability claim. The output
  is an operating-point triage result only.

## Study basis (summary)

The demonstrator reproduces a pipeline developed and evaluated **internally on
BRSET** (a Brazilian colour-fundus dataset) under a **patient-level**
train/validation/reliability/test split. Referable DR is defined as **DR grade
>= 2**. Frozen-backbone embeddings feed a lightweight multi-task head; the
RETFound-Green native-392 + multi-task cell was selected for this demonstrator
because of its strong internal referable-DR performance with a compact frozen
~22M-parameter backbone. Study-level confidence intervals were computed with a
patient-clustered bootstrap; they are study-level evidence, **not** per-image
confidence. External-population, device, workflow, calibration, and prospective
clinical validation remain future work.

See `paper/` for the manuscript that defines the study. FROST (the tool) is a
new addition that reproduces that study's pipeline for local inspection.

---

## Local-only status (default) and optional hosting

Run locally, the server binds to `127.0.0.1` only and performs no external
network access at inference time.

For a **free, persistent, CPU** public deployment there is a self-contained
**Hugging Face Docker Space** setup under [`hf_space/`](hf_space/) — one container
serving the FastAPI backend **and** the static frontend. It is deployment-ready
without a separate weights repo: the public RETFound-Green backbone (Apache-2.0)
is fetched + SHA-verified at Docker build, and the 501 KB trained head + the
operating-point JSON travel inside the Space. No credentialed BRSET data is ever
included; the server's startup self-check uses only synthetic/random inputs, and
the study-score reproduction gate (`analysis/verify_parity.py`) is run **locally**
before deploying. See **[`hf_space/DEPLOY.md`](hf_space/DEPLOY.md)** for the full
step-by-step runbook.

## Project artifact prerequisites

FROST imports the canonical pipeline (`src/retina_screen/`) as a **read-only**
library and reads pre-existing local artifacts. It never modifies them. You need:

1. The frozen RETFound-Green backbone checkpoint
   (`retfoundgreen_statedict.pth`, Apache-2.0).
2. The accepted native-392 MultiTaskHead checkpoint
   (`model_checkpoint.pt` from the accepted native-392 training run).
3. The accepted native-392 embedding cache (validation embeddings + manifest)
   for threshold derivation.
4. The accepted held-out test predictions (`predictions.npz`) for descriptive
   operating-point reporting.
5. (Optional, for study-linked parity) a few local BRSET test images.

These are private and are **never** committed.

## Required environment variables

All private local paths are resolved from environment variables (never hardcoded
in committed files):

| Variable | Meaning |
| --- | --- |
| `RETINA_SCREEN_RETF_GREEN_CHECKPOINT` | Path to the RETFound-Green backbone state-dict (`retfoundgreen_statedict.pth`). |
| `RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT` | Path to the accepted native-392 MultiTaskHead checkpoint (`model_checkpoint.pt`). |
| `RETINA_SCREEN_DEMO_BUNDLE_PATH` | (optional) Override path to the generated local deployment bundle JSON. |
| `RETINA_SCREEN_DEMO_THRESHOLD_PATH` | (optional) Override path to the generated local operating-point JSON. |
| `RETINA_SCREEN_DEMO_PARITY_CASES_PATH` | (optional) Override path to the local study-linked parity-case fixture JSON. |

If the backbone / head environment variables are unset, the bundle builder also
falls back to the canonical default locations used by the pipeline
(`models/retfoundgreen_statedict.pth` for the backbone, and the accepted
native-392 run directory for the head) and the canonical
`RETFOUND_GREEN_CHECKPOINT` variable for the backbone.

---

## Step 1 — Build the local deployment bundle

Discovers the accepted native-392 artifacts, validates SHA-256 hashes, input
size (392), pooling (average), embedding dim (384), head type, and task order,
then writes the **ignored** local manifest
`deploy/referable_dr_demo/.local/deployment_bundle.local.json`.

```bash
uv run python deploy/referable_dr_demo/analysis/build_local_deployment_bundle.py
```

## Step 2 — Derive the validation-only operating-point threshold

Computes the referable score on the **validation** split only, selects the
largest threshold preserving validation sensitivity >= 0.95, applies it
descriptively to the accepted held-out test predictions, and writes the
**ignored** `deploy/referable_dr_demo/.local/operating_point.local.json`.

```bash
uv run python deploy/referable_dr_demo/analysis/derive_threshold.py
```

The threshold is bound to the backbone/head/preprocessing/task-ordering hashes
and is refused by the app if any of them change.

## Step 3 — Verify inference parity

Runs the canonical synthetic parity check (app inference vs canonical pipeline
inference, max abs dr_grade-logit diff <= 1e-5 on CPU) and, if local BRSET test
images are configured, the study-linked parity check (app score vs accepted
`predictions.npz`, abs diff < 1e-4).

```bash
uv run python deploy/referable_dr_demo/analysis/verify_parity.py
```

## Step 4 — Launch locally

```bash
uv run --with-requirements deploy/referable_dr_demo/requirements-local.txt \
  uvicorn deploy.referable_dr_demo.backend.app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>. The frontend is served from the same local
FastAPI process. `GET /health` reports readiness; `POST /predict` accepts one
JPEG/PNG image.

## Step 5 — Run tests

```bash
# App-local tests (run with the local web deps overlaid on the project env):
uv run --with-requirements deploy/referable_dr_demo/requirements-local.txt \
  pytest deploy/referable_dr_demo/tests -q

# Repository architecture-boundary tests (must stay green):
uv run pytest tests/test_no_dataset_coupling.py tests/test_import_boundaries.py -q
```

Heavy tests that need the private checkpoints (parity, frozen-backbone checks)
**skip** cleanly when those artifacts are not configured, so the committed suite
passes in any environment. They run fully on a machine where the artifacts and
environment variables are present.

---

## Privacy behaviour

Default behaviour is **memory-only**:

- Uploads are processed in memory and never written to disk.
- Filenames, patient identifiers, sample IDs, raw image bytes, raw embeddings,
  and image-linked scores are **never** logged.
- No external HTTP requests, cloud APIs, or external vision services are used.

Permitted local logs contain only: timestamp, app/bundle version,
success/failure, timings, technical-error category, and parity status.

## Hash / provenance validation

The bundle loader **fails closed** if a checkpoint is missing, a SHA-256 hash
mismatches, the embedding dim is not 384, pooling is not native average pooling,
the task ordering differs, the native input size is not 392, or the canonical
loader/preprocessing route cannot be verified. The operating-point artifact is
invalidated and refused if the backbone hash, head hash, preprocessing hash,
task ordering, model family, or native protocol changes, or if it was derived
from a non-validation split.

## How to interpret the fixed research alert

- **REFERABLE** (score >= validation-selected threshold): the referable-DR score
  is above the validation-selected screening threshold. Prompt ophthalmic
  clinical review is indicated. This is **not** a diagnosis.
- **NOT REFERABLE** (score < threshold): below the validation-selected research
  alert threshold. This is **not** a rule-out result and does not replace
  clinician assessment.

The result is an operating-point triage decision, **not** a calibrated
probability. The reported "at this fixed threshold, held-out test sensitivity X%
/ specificity Y%" values are descriptive context from the accepted internal
BRSET test split.

## Expected directory structure

```
deploy/referable_dr_demo/
  README.md
  requirements-local.txt
  .gitignore
  __init__.py
  backend/
    app.py                 # FastAPI app: GET /health, POST /predict, static hosting
    schemas.py             # response models
    static_server.py       # static frontend mounting
    service/
      bundle.py            # accepted-artifact discovery + fail-closed hash validation
      inference.py         # load-once frozen backbone+head; exact-parity inference
      preprocessing_parity.py  # canonical native-392 preprocessing wrapper
      threshold_policy.py  # validation-only operating-point load + invalidation
      image_checks.py      # technical input checks (not clinical quality)
      provenance.py        # hashing, versions, git commit, task-order hash
      privacy.py           # safe logging / redaction helpers
  frontend/
    index.html  app.js  styles.css  assets/pipeline.svg
  analysis/
    build_local_deployment_bundle.py
    derive_threshold.py
    verify_parity.py
  config/
    deployment_bundle.example.yaml   # committed template
    operating_point.example.json     # committed template
  tests/                             # 26 app-local tests
  .local/                            # generated, IGNORED (bundle/threshold/parity)
```

## Future note (not implemented here)

A later production deployment would add external-population and device
validation, prospective clinical evaluation, probability calibration, image
quality / out-of-distribution gating, access control, audit logging with
consent, and a hardened public-hosting configuration. None of that is present or
claimed in this local demonstrator.
