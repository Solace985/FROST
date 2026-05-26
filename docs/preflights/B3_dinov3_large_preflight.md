# B3 DINOv3-Large Readiness Preflight

**Stage:** 8D-3.5 B3  
**Date:** 2026-05-26  
**Verdict:** READY_WITH_ACTION_ITEMS  

---

## 1. Verdict

**READY_WITH_ACTION_ITEMS** — all 6 gates PASS; no code changes required in this preflight.  
Three user-action items required before extraction begins (all in the B3 RUN session):

> **ACTION 1:** Add `_load_dinov3_large()` to `src/retina_screen/embeddings.py` and a
> `'dinov3'` dispatch branch in `load_backbone()`. Use timm model
> `vit_large_patch16_dinov3_qkvb` with `img_size=224`, `global_pool='token'`,
> `checkpoint_path=<path>`. No manual key remap required — timm's
> `checkpoint_filter_fn` in `eva.py` handles all key translations automatically.
>
> **ACTION 2:** Create `configs/backbone/dinov3_large.yaml`,
> `configs/preprocessing/dinov3_large_224.yaml`, and MT+LP experiment configs.
>
> **ACTION 3:** Review the DINOv3 License
> (`https://ai.meta.com/resources/models-and-libraries/dinov3-license/`).
> Weights are NOT Apache-2.0. Code in timm is Apache-2.0. Confirm research-use
> compliance before extraction. Do NOT print or store any signed download URL.

---

## 2. Purpose

Determine whether the DINOv3-Large backbone can be added to the Stage 8D-3.5 six-backbone
matrix as a seventh comparator. The preflight is read-only: it verifies the weights file,
inspects the state-dict without model instantiation, establishes the timm loader path, and
evaluates all 6 gates and 15 dimensions.

**Not in scope:** embedding extraction, training, evaluation, Decision 030.

---

## 3. Weights File

| Field | Value |
|-------|-------|
| Path | `models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| SHA-256 | `8AA4CBDDDA325040FC78DB2C272754AF6EBE8FF2C55F6EC4F1964D8890F66035` |
| Filename prefix match | `8aa4cbdd` matches first 8 hex chars of SHA-256 ✓ |
| File size | 1,213,050,671 bytes (~1.13 GB) |
| Plausible for ViT-L/16? | Yes — ~307M parameters × 4 bytes/param ≈ 1.15 GB fp32 ✓ |
| Access gate | None — direct Meta release, no credentialed download required |
| License | DINOv3 License (Meta). NOT Apache-2.0. See Action 3. |

---

## 4. Gate Summary

| Gate | Description | Verdict | Evidence |
|------|-------------|---------|----------|
| B3-L | LOADER: timm support | **PASS** | timm 1.0.27 lists `vit_large_patch16_dinov3` (EVA module); `_qkvb` variant preferred |
| B3-W | W1 storage: `_compact_embedding()` | **PASS** | Returns 1-D `(1024,)` tensor; clone() breaks storage link unconditionally |
| B3-S | STRICT LOAD feasibility | **PASS** | No manual remap needed; timm `checkpoint_filter_fn` handles all key translations |
| B3-D | Predictions export (A1 patch) | **PASS** | `scripts/05_evaluate.py:429-430` exports `predictions.npz` |
| B3-N | Cache namespace isolation | **PASS** | `dinov3_large` is a new namespace distinct from all 6 existing backbones |
| B3-T | Token count at 224px | **PASS** | 1 CLS + 4 reg + 196 patch = 201 tokens; `num_reg_tokens=4`, `patch_size=16`, RoPE |
| B3-C | LVD-1689M contamination | **PASS** | Web-scale natural images only; no medical/clinical data; no exclusion list needed |

---

## 5. 15-Dimension Checklist

| # | Dimension | Status | Notes |
|---|-----------|--------|-------|
| 1 | Weights file present | PASS | `models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` exists |
| 2 | SHA-256 verified | PASS | `8AA4CBDD...` matches filename prefix |
| 3 | File size plausible | PASS | 1.13 GB ≈ ViT-L/16 fp32 |
| 4 | timm version support | PASS | timm 1.0.27; `vit_large_patch16_dinov3` in `eva.py` |
| 5 | State-dict keys expected | PASS | 368 keys; `cls_token`, `patch_embed.proj.weight`, `blocks.0–23.*`, `norm.weight/bias`, `storage_tokens` all present |
| 6 | embed_dim=1024 confirmed | PASS | `patch_embed.proj.weight` shape `[1024, 3, 16, 16]` |
| 7 | patch_size=16 confirmed | PASS | From `patch_embed.proj.weight` last two dims = 16 |
| 8 | Register/storage tokens (4×) | PASS | `storage_tokens` shape `[1, 4, 1024]` |
| 9 | pos_embed / RoPE handling | PASS | No `pos_embed` key; `rope_embed.periods` discarded by `checkpoint_filter_fn`; `dynamic_img_size=True` handles 224px natively |
| 10 | norm vs fc_norm key names | PASS | `norm.weight/bias` in checkpoint; EVA model has `use_fc_norm=False` (always `norm.*`, any pooling mode); **no remap needed** |
| 11 | W1 storage: `_compact_embedding()` | PASS | `embeddings.py:521`; handles any 1-D tensor ✓ |
| 12 | BackboneConfig fields present | PASS | `input_size`, `global_pool` at `embeddings.py:120–121` ✓ |
| 13 | `load_backbone()` needs "dinov3" | ACTION | `embeddings.py:437–441` dispatch; add `'dinov3'` branch in B3 RUN |
| 14 | A1 predictions patch present | PASS | `scripts/05_evaluate.py:429–430` exports `predictions.npz` |
| 15 | Cache namespace isolation | PASS | `dinov3_large` distinct from `mock`, `resnet50`, `convnext_base`, `dinov2_base`, `dinov2_large`, `retfound_green`, `retfound_green_native392` |

---

## 6. State-Dict Inspection (no model instantiation)

Key results from `torch.load(..., weights_only=True)`:

```
total keys:       368
cls_token:        [1, 1, 1024]
storage_tokens:   [1, 4, 1024]     ← register-equivalent, 4 tokens
patch_embed.proj: [1024, 3, 16, 16]
depth:            24 blocks (blocks.0 – blocks.23)
pos_embed:        absent           ← RoPE replaces learned PE
rope_embed.periods: present        ← discarded by timm filter_fn
norm.weight/bias: present
fc_norm.*:        absent           ← no remap needed for avg-pool
qkv.bias:         present (24 ×)   ← split by filter_fn into q_bias + v_bias
qkv.bias_mask:    present (24 ×)   ← discarded by filter_fn
mask_token:       present          ← discarded by filter_fn
```

---

## 7. timm Loader Path (B3-L gate detail)

timm 1.0.27 implements DINOv3 in `eva.py` (not `vision_transformer.py`). The EVA-based
`checkpoint_filter_fn` detects DINOv3 weights by checking `'storage_tokens' in state_dict`,
then applies these transformations:

| Checkpoint key | timm model key | Action |
|----------------|----------------|--------|
| `storage_tokens` | `reg_token` | Renamed |
| `ls1.gamma` | `gamma_1` | Renamed |
| `ls2.gamma` | `gamma_2` | Renamed |
| `qkv.bias` | `q_bias` + `v_bias` | Split (k_bias fixed at 0) |
| `rope_embed.periods` | — | Discarded |
| `qkv.bias_mask` | — | Discarded |
| `mask_token` | — | Discarded |
| All other keys | Same | Pass-through |

**Recommended timm model:** `vit_large_patch16_dinov3_qkvb`
- `qkv_bias=True` — has separate Q/V bias parameters; Q/V biases from checkpoint are loaded
- `num_reg_tokens=4` — matches `storage_tokens` shape
- `use_fc_norm=False` — `norm.*` always matches, no remap needed
- `dynamic_img_size=True` — handles 224px natively (RoPE, no PE interpolation)

**B3 RUN loading recipe:**
```python
_TIMM_MODEL_NAME = "vit_large_patch16_dinov3_qkvb"
model = timm.create_model(
    _TIMM_MODEL_NAME,
    img_size=(224, 224),
    num_classes=0,
    global_pool="token",   # CLS-token — matches original DINOv3 protocol
    checkpoint_path=checkpoint_path,
)
```
No manual state-dict key remap required (unlike `_load_retfound_green`).

**Protocol note:** Original DINOv3 uses CLS-token pooling. timm's EVA architecture defaults
to avg-pool. Pass `global_pool='token'` explicitly for protocol fidelity. This is the
recommended setting for a matrix-consistent evaluation.

---

## 8. W1 Storage Analysis (B3-W gate detail)

`_compact_embedding()` at `src/retina_screen/embeddings.py:512–521`:

```python
def _compact_embedding(raw: torch.Tensor) -> torch.Tensor:
    return raw.detach().cpu().clone().contiguous()
```

Called at `embeddings.py:808` as:
```python
embedding = _compact_embedding(backbone(tensor).squeeze(0))
```

With `global_pool='token'` and `num_classes=0`, the EVA model returns the CLS-token embedding
as a `(1024,)` tensor. The `.clone()` breaks any storage link to the full token sequence
buffer. The `.contiguous()` ensures flat serialization. Storage size:
`1024 × 4 bytes = 4,096 bytes per sample` (W1 ratio = 1.0 guaranteed).

---

## 9. Token Count (B3-T gate detail)

At `img_size=224`, `patch_size=16`:
- Patch grid: 224 ÷ 16 = 14 → 14 × 14 = **196 patch tokens**
- Plus: **1 CLS token** + **4 register/storage tokens**
- **Total: 201 tokens**

`dynamic_img_size=True` in the timm factory means DINOv3 computes RoPE frequencies at
forward-pass time for the actual sequence length — no PE interpolation required at load time.
The original DINOv3 architecture uses RoPE with no learned positional embedding.

---

## 10. LVD-1689M Contamination (B3-C gate detail)

LVD-1689M is Meta's curated web-scale dataset of 1.689 billion natural images (ImageNet-1K,
ImageNet-22K, Google Landmarks, and curated web data). It contains no medical imaging data
and no retinal fundus photographs. No exclusion list applicable. Decision 029 (ODIR-in-any-form
exclusion) applies only to RETFound-Green and RETFound-MEH.

---

## 11. DINOv3 License Note

**Weights:** DINOv3 License (Meta AI). See:
`https://ai.meta.com/resources/models-and-libraries/dinov3-license/`  
**timm code:** Apache-2.0  
**Implication:** Downstream outputs (embeddings, model outputs) may be subject to DINOv3
license terms. Confirm compliance for research use before extraction. This is different from
DINOv2 (CC-BY-NC) and RETFound-Green (Apache-2.0). Do NOT store the signed download URL
in any report or document.

---

## 12. Action Items for B3 RUN

| # | Item | File | Notes |
|---|------|------|-------|
| AI-1 | Add `_load_dinov3_large()` | `src/retina_screen/embeddings.py` | Follow `_load_retfound_green` pattern; use `timm.create_model('vit_large_patch16_dinov3_qkvb', img_size=224, num_classes=0, global_pool='token', checkpoint_path=...)` |
| AI-2 | Add `'dinov3'` dispatch | `src/retina_screen/embeddings.py:437` | Add branch in `load_backbone()` after retfound_green branch |
| AI-3 | Update docstring model_types | `src/retina_screen/embeddings.py:36` | Add `'dinov3'` to supported model_type list |
| AI-4 | Create backbone config | `configs/backbone/dinov3_large.yaml` | `embedding_dim=1024, model_type=dinov3, input_size=224, global_pool=token, checkpoint_path=models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| AI-5 | Create preprocessing config | `configs/preprocessing/dinov3_large_224.yaml` | Standard 224px, ImageNet mean/std |
| AI-6 | Create MT experiment config | `configs/experiment/stage8d35_b3_brset_dinov3_large_multitask.yaml` | Follow stage8d35_b2 pattern |
| AI-7 | Create LP experiment config | `configs/experiment/stage8d35_b3_brset_dinov3_large_linearprobe.yaml` | Follow stage8d35_b2 LP pattern |
| AI-8 | Add targeted tests | `tests/test_backbone_dinov3.py` | Smoke test: create_model succeeds, output shape (1,1024), frozen=True |
| AI-9 | DINOv3 license compliance | Docs / Decision 030 | Confirm research-use compliance; record in Decision 030 |
| AI-10 | Set checkpoint path | Env var or YAML | `DINOV3_LARGE_CHECKPOINT=/path/to/...` or `checkpoint_path:` in backbone YAML |

---

## 13. Tests

```
pytest (excluding tests/test_odir_adapter.py):
  850 passed, 1 skipped — all green at preflight time
```

No test changes were made. No tests reference DINOv3 yet (expected — loader not yet added).

---

## 14. Git State

```
branch: main
last commit: f6dafa46 Stage 8D-3.5: fix RETFound-Green native392 avg-pool key name mismatch
working tree: clean (no staged changes at preflight time)
```

This preflight document is the only new file. No src/, configs/, tests/, or existing outputs/
were modified.

---

## 15. Summary

DINOv3-Large is technically ready for B3 RUN:
- Weights file verified (SHA-256 ✓, size ✓)
- timm 1.0.27 has native DINOv3 support (`vit_large_patch16_dinov3_qkvb`, EVA architecture)
- timm's `checkpoint_filter_fn` handles all key remaps automatically — simpler than RETFound-Green
- All 6 gates PASS; no blockers
- Embedding dimension (1024) and token count (201) confirmed without model instantiation
- 15/15 dimensions evaluated; 1 is an ACTION (add "dinov3" dispatch in B3 RUN)

**Overall verdict: READY_WITH_ACTION_ITEMS**  
B3 RUN prompt can proceed after user reviews this document and confirms DINOv3 license compliance.
