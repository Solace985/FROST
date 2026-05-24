# B2 RETFound-Green Pre-Flight + Setup Report

**Stage:** 8D-3.5 B2  
**Date:** 2026-05-25  
**Verdict:** READY_WITH_ACTION_ITEMS  

---

## 1. Verdict

**READY_WITH_ACTION_ITEMS** — all gates PASS; one user action required before extraction:

> **ACTION:** Set `checkpoint_path` in `configs/backbone/retfound_green_matched224.yaml`
> to the local path of `retfoundgreen_statedict.pth`, or export
> `RETFOUND_GREEN_CHECKPOINT=/path/to/retfoundgreen_statedict.pth`.
>
> Download (Apache-2.0, no access gate):
> `https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth`

All code infrastructure is in place. The B2 RUN prompt requires no code changes — only the checkpoint path must be set.

---

## 2. Purpose

Implement all code, config, and test infrastructure required for Stage 8D-3.5 B2 RETFound-Green BRSET extraction + training + evaluation, so the next B2 RUN prompt can proceed without any code modifications.

---

## 3. Scope

This preflight covers:
- Pre-execution verification (git state, tests, B1 dependency gates)
- timm dependency installation and model verification
- RETFound-Green loader implementation in `src/retina_screen/embeddings.py`
- `checkpoint_path` passthrough in scripts/02 and scripts/03
- `configs/backbone/retfound_green_matched224.yaml` creation
- B2 MT/LP experiment config creation
- Targeted test suite (15 tests, all PASS)
- Architecture smoke (output shape confirmed (1, 384))
- W1 compaction verification for 384-dim output
- Baseline cache snapshot
- Future B2 RUN command specification

**NOT in scope:** BRSET extraction, head training, evaluation, A1 bootstrap CI, native-392 RETFound-Green, DINOv3-Large.

---

## 4. B1 Dependency Status

| Check | Status |
|-------|--------|
| Commit `5eacbd06` present in git log | ✓ PASS |
| `docs/verifications/B1_backbone_candidates_verification.md` exists | ✓ PASS |
| `docs/paper_framing_and_findings.md` F10 = ACTIVE | ✓ PASS |
| `docs/decisions.md` Decision 029 present | ✓ PASS |
| RETFound-Green verdict = INCLUDE_WITH_WARNINGS | ✓ PASS |
| B2 route = GO | ✓ PASS |
| DINOv3-Large remains out of scope | ✓ confirmed |

**Gate B2-0: PASS**

---

## 5. A1 Dependency Status

Stage 8D-3.5 A1 Bootstrap CI: GATE-SATISFIED (commits 97191e47, ce72a5a0, 9845f015).
- All 4 statistical-rigor gates PASS (M/P/C/D)
- No statistical blocker for B2 run
- `scripts/05_evaluate.py` A1 patch present (saves `predictions.npz` + `predictions_schema.json`)

**Gate B2-D: PASS** — per-sample prediction export verified present.

---

## 6. BRSET-Only Binding

Both B2 experiment configs bind exclusively to `dataset: brset`. No ODIR, AIROGS, DDR, mBRSET, APTOS, IDRiD, Messidor, RFMiD, or EyePACS references in any config value.

**Gate B2-DATASET: PASS**

---

## 7. ODIR / AIROGS / DDR Exclusion

Per Decision 029: RETFound-Green must NOT be used for external validation on ODIR-2019, ODIR-5K, ODIR-derivatives, AIROGS, or DDR (pretraining dataset overlap). This is documented in:
- `configs/backbone/retfound_green_matched224.yaml` `external_validation_exclusions`
- `docs/verifications/B1_backbone_candidates_verification.md` §4 and §8
- `docs/decisions.md` Decision 029

---

## 8. Matched-224 Protocol Binding

- **B2 primary:** `matched-224 / default_224 / CLS-token representation`
- **Native-392/avg-pool:** explicitly deferred (`native_392_deferred: true` in backbone YAML)
- Loader creates model at `img_size=(224, 224)`, `num_classes=0`
- Loader does **NOT** set `model.global_pool = "avg"` (that is the native-392 protocol)
- Default `global_pool = "token"` → CLS-token representation, consistent with DINOv2 matrix

Protocol mismatch with published RETFound-Green paper (native-392) is intentional: B2 tests whether retinal-specialized pretraining remains useful under the strict cross-backbone matched protocol.

**Gate B2-M: PASS**

---

## 9. Native-392 Deferral Statement

- No `configs/backbone/retfound_green_native392.yaml` created
- No native-392 experiment configs created
- No native-392 cache namespace defined
- No native-392 tests written
- Native-392/avg-pool may be run as a secondary row after B2 matched-224 is complete

---

## 10. Existing retfound.yaml: Non-Reuse Confirmed

`configs/backbone/retfound.yaml` has `embedding_dim: 1024`, `model_type: retfound`, `status: deferred` — it is the placeholder for the original RETFound (ViT-L, HF gated, out of scope). It was not modified and must not be reused for RETFound-Green.

New config: `configs/backbone/retfound_green_matched224.yaml` — distinct name, model_type, embedding_dim.

---

## 11. timm Dependency Implementation

- Added `timm>=0.9` to `[project.optional-dependencies]` → `torch` group in `pyproject.toml`
- Installed: **timm 1.0.27**
- `vit_small_patch14_reg4_dinov2` confirmed in `timm.list_models()` ✓
- Synthetic output shape `(1, 384)` confirmed ✓
- `global_pool = 'token'` (CLS, not avg) confirmed ✓

**Gate B2-L (dependency part): PASS**

---

## 12. RETFound-Green Loader Implementation

**File:** `src/retina_screen/embeddings.py`  
**Scope:** minimal additions only; no refactoring of unrelated logic.

Changes:
- `import os` added (env var fallback for checkpoint path)
- `BackboneConfig.checkpoint_path: str = ""` added (optional field)
- Module docstring updated: added `retfound_green` to supported types; corrected RETFound note
- `_load_retfound_green()` function added (between `_load_dinov2()` and `load_backbone()`)
- `if config.model_type == "retfound_green": return _load_retfound_green(config, device)` added to `load_backbone()`
- `load_backbone()` docstring and error message updated to list `retfound_green`

Fail-loud checkpoint policy:
1. `config.checkpoint_path` empty AND `RETFOUND_GREEN_CHECKPOINT` env var unset → `BackboneUnavailableError` with download URL
2. Checkpoint file does not exist → `BackboneUnavailableError` with path and download URL
3. `timm.create_model` fails → `BackboneUnavailableError` with timm version context
4. Output shape ≠ `(1, 384)` → `BackboneDimensionError` via `_verify_embedding_dim`
5. timm not installed → `BackboneUnavailableError`
6. `vit_small_patch14_reg4_dinov2` not in `timm.list_models()` → `BackboneUnavailableError`

Zero fallbacks to: mock, random weights, DINOv2, original RETFound, or any other backbone.

Script updates (one line each):
- `scripts/02_verify_backbone_one_image.py` `_build_backbone_config()`: added `checkpoint_path=backbone_raw.get("checkpoint_path", "")`
- `scripts/03_extract_embeddings.py` `_build_backbone_config()`: added `checkpoint_path=backbone_raw.get("checkpoint_path", "")`

**Gate B2-L: PASS**

---

## 13. RETFound-Green Config Created

**`configs/backbone/retfound_green_matched224.yaml`**

Key fields:
```
name: retfound_green
model_type: retfound_green
timm_model_name: vit_small_patch14_reg4_dinov2
version: retfound_green_v0.1
embedding_dim: 384
input_size: 224
native_input_size: 392
matrix_pooling: cls
native_392_deferred: true
checkpoint_path: ""   ← USER MUST SET BEFORE B2 RUN
```

---

## 14. B2 MT/LP Experiment Configs Created

| Config | head_type | embedding_dim | stage | final_test_result |
|--------|-----------|---------------|-------|-------------------|
| `stage8d35_b2_brset_retfound_green_matched224_multitask.yaml` | multitask | 384 | 8D-3.5-B2 | false |
| `stage8d35_b2_brset_retfound_green_matched224_linearprobe.yaml` | linear_probe | 384 | 8D-3.5-B2 | false |

Both: `preliminary: true`, `rehearsal: false`, `full_dataset_run: true`, `fast_dev_run: false`, `dataset: brset`, `class_weighting_enabled` absent (inherits `false` from `configs/training/standard.yaml`, Decision 027).

---

## 15. Loader Smoke / Output-Shape Verification

```python
# No weights download, no BRSET, no cache artifacts
import timm, torch
m = timm.create_model('vit_small_patch14_reg4_dinov2', img_size=(224,224), num_classes=0, pretrained=False)
m.eval()
with torch.no_grad():
    out = m(torch.zeros(1,3,224,224))
# Output shape: (1, 384) ✓
# global_pool:  token   ✓
```

**Result:** Output `(1, 384)` confirmed. `global_pool = 'token'` (CLS, not avg) confirmed.

---

## 16. W1 Compaction Verification

`_compact_embedding()` is called unconditionally at line 675 of `embeddings.py`:
```python
embedding = _compact_embedding(backbone(tensor).squeeze(0))
```

This applies to ALL backbones including RETFound-Green. ViT-S/14-reg4 outputs CLS token as a view into the full token sequence — compaction is essential to avoid storage bloat.

Verified by tests:
- `test_retfound_green_w1_storage_ratio_exceeds_threshold_before_compaction`: storage_ratio > 5.0 before compaction ✓
- `test_retfound_green_w1_compaction_reduces_storage_ratio`: storage_ratio ≤ 2.0 after `_compact_embedding()` ✓

Expected B2 compact file size: ~1.5 KB per sample at 384-dim (vs ~600 KB uncompacted for ViT-S/14-reg4 with 261 token positions).

**Gate B2-W: PASS**

---

## 17. Cache Namespace Plan

```
cache/embeddings/retfound_green/brset/92d0f40b94aea26c/
```

- `backbone_name = retfound_green` (from `name:` in backbone YAML)
- `dataset_source = brset`
- `preprocessing_hash = 92d0f40b94aea26c` (default_224 — same as all existing Stage 8D caches)
- `embedding_dim = 384`
- MT and LP configs resolve to the same namespace (no double extraction)

No collision with: `resnet50`, `convnext_base`, `dinov2_base`, `dinov2_large`, `retfound` (if ever used).

**Gate B2-N: PASS**

---

## 18. Future B2 RUN Commands

```bash
# Step 1: Set checkpoint path (REQUIRED before running steps 2–7)
# Option A — edit YAML:
#   configs/backbone/retfound_green_matched224.yaml
#   set checkpoint_path: "/absolute/path/to/retfoundgreen_statedict.pth"
# Option B — env var:
#   export RETFOUND_GREEN_CHECKPOINT=/path/to/retfoundgreen_statedict.pth
# Download (Apache-2.0): https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth

# Step 2: Verify backbone (one-image, no BRSET extraction, safe)
uv run python scripts/02_verify_backbone_one_image.py \
  --config configs/experiment/stage8d35_b2_brset_retfound_green_matched224_multitask.yaml

# Step 3: Extract embeddings (full BRSET, matched-224)
uv run python scripts/03_extract_embeddings.py \
  --config configs/experiment/stage8d35_b2_brset_retfound_green_matched224_multitask.yaml

# Step 4: Train MultiTaskHead
uv run python scripts/04_train.py \
  --config configs/experiment/stage8d35_b2_brset_retfound_green_matched224_multitask.yaml

# Step 5: Evaluate MultiTaskHead
uv run python scripts/05_evaluate.py \
  --config configs/experiment/stage8d35_b2_brset_retfound_green_matched224_multitask.yaml \
  --run-dir <MT_RUN_DIR>

# Step 6: Train LinearProbeHead (reuses embedding cache — no re-extraction)
uv run python scripts/04_train.py \
  --config configs/experiment/stage8d35_b2_brset_retfound_green_matched224_linearprobe.yaml

# Step 7: Evaluate LinearProbeHead
uv run python scripts/05_evaluate.py \
  --config configs/experiment/stage8d35_b2_brset_retfound_green_matched224_linearprobe.yaml \
  --run-dir <LP_RUN_DIR>
```

---

## 19. Expected B2 RUN Outputs

| Artifact | Path |
|----------|------|
| Embedding cache | `cache/embeddings/retfound_green/brset/92d0f40b94aea26c/` |
| Cache manifest | `cache/embeddings/retfound_green/brset/92d0f40b94aea26c/manifest.csv` |
| Cache provenance | `cache/embeddings/retfound_green/brset/92d0f40b94aea26c/cache_provenance.json` |
| MT run dir | `runs/<stage8d35_b2_mt_<timestamp>>/` |
| LP run dir | `runs/<stage8d35_b2_lp_<timestamp>>/` |
| Per eval run | `overall_metrics.json`, `subgroup_metrics.json`, `evaluation_summary.json`, `diagnostics.json`, `predictions.npz`, `predictions_schema.json` |

Expected cache: ~16,266 samples, ~25–100 MB compact (384-dim × ~1.5 KB each).

---

## 20. Baseline Cache Snapshot

| Backbone | Path | embedding_dim | total_samples | size_MB | provenance SHA-256 (prefix) |
|----------|------|---------------|---------------|---------|------------------------------|
| resnet50 | `cache/embeddings/resnet50/brset/92d0f40b94aea26c/` | 2048 | 16266 | 158.6 | `c4ae89d8...` |
| convnext_base | `cache/embeddings/convnext_base/brset/92d0f40b94aea26c/` | 1024 | 16266 | 95.2 | `8d0c9294...` |
| dinov2_base | `cache/embeddings/dinov2_base/brset/92d0f40b94aea26c/` | 768 | 16266 | 12278.7 | `3fa2629d...` |
| dinov2_large | `cache/embeddings/dinov2_large/brset/92d0f40b94aea26c/` | 1024 | 16266 | 95.2 | `463487f0...` |

Note: dinov2_base at 12278.7 MB is a pre-W1-fix cache (view-storage bug present at extraction time). dinov2_large at 95.2 MB is post-W1-fix (Decision 028). The retfound_green cache will be created by the B2 RUN prompt and should be compact (~25–100 MB).

---

## 21. Tests Run and Results

```
tests/test_retfound_green_b2_setup.py     15 passed
tests/test_no_dataset_coupling.py         320 passed (pre-B2 baseline)
tests/test_import_boundaries.py           (included in baseline 320)
tests/test_embedding_storage.py           (included in full suite)
Full suite (excl. test_odir_adapter.py):  833 passed, 1 skipped
test_odir_adapter.py:                     5 errors (pre-existing, require ODIR raw data)
```

All required tests pass. Pre-existing ODIR real-data smoke errors are unrelated to B2 setup.

---

## 22. Files Changed

**Modified (tracked):**
- `pyproject.toml` — `timm>=0.9` added to torch optional-deps group
- `src/retina_screen/embeddings.py` — BackboneConfig + `_load_retfound_green()` + `load_backbone()` update
- `scripts/02_verify_backbone_one_image.py` — checkpoint_path passthrough
- `scripts/03_extract_embeddings.py` — checkpoint_path passthrough

**Created (tracked):**
- `configs/backbone/retfound_green_matched224.yaml`
- `configs/experiment/stage8d35_b2_brset_retfound_green_matched224_multitask.yaml`
- `configs/experiment/stage8d35_b2_brset_retfound_green_matched224_linearprobe.yaml`
- `tests/test_retfound_green_b2_setup.py`
- `docs/preflights/B2_retfound_green_preflight.md` (this file)

**Created (local, not committed):**
- `outputs/stage8d35_b2_preflight/20260525T003735Z/b2_preflight_report.md`
- `outputs/stage8d35_b2_preflight/20260525T003735Z/b2_readiness.json`
- `outputs/stage8d35_b2_preflight/20260525T003735Z/setup_verification_manifest.json`

**Not modified:**
- `configs/backbone/retfound.yaml` (still deferred for original RETFound/RETFound-MEH)
- All other configs, src files outside embeddings.py, A1 utilities, framing docs

---

## 23. B2 Gates Summary

| Gate | Condition | Result |
|------|-----------|--------|
| B2-0 | B1 dependency (5eacbd06, Decision 029, F10 ACTIVE, B2 GO) | ✓ PASS |
| B2-DATASET | BRSET-only, no ODIR/AIROGS/DDR | ✓ PASS |
| B2-L | timm loader implemented, 384-dim verified, CLS protocol, fail-loud | ✓ PASS |
| B2-W | W1 compaction applies unconditionally, 384-dim test PASS | ✓ PASS |
| B2-M | matched-224 primary, native-392 deferred, no avg-pool | ✓ PASS |
| B2-D | A1 patch present (predictions.npz + schema), future evals will produce it | ✓ PASS |
| B2-N | Cache namespace `retfound_green/brset/92d0f40b94aea26c/`, no collision | ✓ PASS |
| B2-R | Decision 027 unweighted recipe, no class-weighting/focal/threshold changes | ✓ PASS |

---

## 24. Stop Conditions for B2 RUN

Before running extraction/training/evaluation:
- [ ] `checkpoint_path` set in YAML OR `RETFOUND_GREEN_CHECKPOINT` env var set
- [ ] `scripts/02_verify_backbone_one_image.py` runs successfully (verifies loader end-to-end)
- [ ] No new test failures introduced
- [ ] `git status` shows only `outputs/` and `cache/` untracked (no uncommitted code)

During B2 RUN:
- If extraction raises `BackboneUnavailableError` → check checkpoint path, not a code bug
- If output shape ≠ 384 → `BackboneDimensionError` — do not proceed to training
- If `final_test_result=true` appears in any evaluation output → STOP, investigate config
- Do not retrain existing Stage 8D-3 matrix cells

---

## 25. Confirmation

- [x] No extraction
- [x] No training
- [x] No evaluation
- [x] No A1
- [x] No native-392
- [x] No DINOv3
- [x] No ODIR / no external datasets as active data
- [x] BRSET only
- [x] No cache / runs / data / model weights committed
- [x] outputs/ not committed
- [x] retfound.yaml not modified

---

## Final Recommendation

**PROCEED to B2 RUN** after:
1. Downloading `retfoundgreen_statedict.pth` from the URL above
2. Setting `checkpoint_path` in `configs/backbone/retfound_green_matched224.yaml` (or env var)
3. Running `scripts/02_verify_backbone_one_image.py` to confirm end-to-end loader

The B2 RUN prompt job: extraction → MT training → MT evaluation → LP training → LP evaluation → B2 run report/manifest. No code changes required.
