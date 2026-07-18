---
title: FROST Referable DR Research Demonstrator
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# FROST — Referable Diabetic Retinopathy Research Demonstrator

**Frozen Representation for Ocular Screening and Triage.**

> ⚠️ **This is a research demonstrator. It is NOT a medical device, is NOT
> validated for clinical decisions, and requires independent clinical
> assessment.** It reproduces a pipeline validated **only on BRSET's internal
> distribution** (a Brazilian colour-fundus dataset). Internal performance does
> not guarantee performance on images from other sources.

Upload one colour fundus photograph; the app runs the study's exact frozen
pipeline and returns a **REFERABLE / NOT REFERABLE** operating-point triage
result with a full step-by-step trace. Uploads are processed **in memory only**
and are never stored.

## Pipeline (identical to the study)

```
image → native-392 preprocessing → frozen RETFound-Green backbone (ViT-S/14, avg-pool)
      → 384-d embedding → locked MultiTaskHead → 5-class DR-grade logits → softmax
      → referable-DR score = P(grade ≥ 2) → fixed validation-derived threshold → decision
```

- Backbone: **RETFound-Green** (`vit_small_patch14_reg4_dinov2`, native 392,
  average pooling), Apache-2.0, loaded frozen. Fetched + SHA-verified at build.
- Head: the study's trained native-392 MultiTaskHead (frozen).
- Threshold: selected on the BRSET **validation** split only (largest T keeping
  validation sensitivity ≥ 0.95), then frozen and provenance-bound.

## Notes

- Single FastAPI container serves both the UI (`/`) and inference (`/predict`) —
  same origin, so no CORS config is needed. HTTPS is provided by Spaces.
- CPU-only; the model is a load-once process singleton. On the free tier the
  Space sleeps after inactivity and wakes in a few seconds on the next visit.
- This Space contains no credentialed data. The study-score reproduction gate is
  run locally by the authors before deployment; the server's own startup
  self-check uses only synthetic/random inputs.
