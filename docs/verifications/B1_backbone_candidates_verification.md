# B1 Backbone Candidate Verification — RETFound-Green Partial Closeout

**Stage:** 8D-3.5 B1  
**Date:** 2026-05-24  
**Scope:** RETFound-Green partial B1 closeout only. DINOv3-Large is explicitly out of scope for this prompt (handled in a separate later prompt; remains BLOCKED_ACCESS).

---

## §1 Verdict Summary

| Item | Verdict |
|------|---------|
| **Overall B1 (this prompt)** | PARTIAL — RETFound-Green closed out; DINOv3-Large deferred |
| **RETFound-Green** | `INCLUDE_WITH_WARNINGS` |
| **B2 RETFound-Green** | **GO** |
| **DINOv3-Large** | Out of scope — `BLOCKED_ACCESS` (HuggingFace gated; handled separately) |
| **B3 DINOv3-Large** | NO-GO (pending separate access-resolution prompt) |
| **FLAIR** | Out of scope — rejected (B4) |
| **DINOv2-Giant** | Out of scope — rejected (B5) |
| **RETFound-MEH** | Out of scope — access-pending/deferred |

---

## §2 Scope and Boundaries

This is a **partial B1 closeout for RETFound-Green only.** It covers:
- Primary-source verification of RETFound-Green (six B1 questions)
- Pretraining dataset caveat analysis
- timm loader path documentation
- Protocol mismatch analysis (native-392 vs matched-224)
- retfound.yaml mismatch documentation
- B2 forward-routing notes
- F10 framing-doc update
- Decision 029 appended to docs/decisions.md

**Not performed in this prompt:**
- DINOv3-Large verification (separate prompt)
- Embedding extraction
- Model training or evaluation
- A1 bootstrap CI rerun
- Any src/retina_screen/ modification
- Any tests/ or configs/ modification
- Weight download
- B2 or B3 execution

---

## §3 Primary Sources

| Type | Title | URL | Accessed | Used For |
|------|-------|-----|----------|----------|
| paper | Training a high-performance retinal foundation model with half-the-data and 400 times less compute | https://arxiv.org/abs/2405.00117 | 2026-05-24 | loader, embedding_dim, pretraining, BRSET numbers, resolution |
| paper | RETFound-Green (Nature Communications 2025) | https://www.nature.com/articles/s41467-025-62123-z | 2026-05-24 | BRSET numbers, pretraining confirmation |
| repo | RETFound_Green GitHub (Engelmann & Bernabeu) | https://github.com/justinengelmann/RETFound_Green | 2026-05-24 | weights URL, loader call, license |

---

## §4 RETFound-Green Verified Facts

### (a) Weights access

- **Weights URL:** https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
- **Access gate:** None. Public GitHub release, no login required, no approval required, no DUA.
- **License:** Apache 2.0
- **Stability:** v0.1 release tag on official author repository (Engelmann & Bernabeu). Stable for reproducibility.
- **Source:** GitHub releases page (https://github.com/justinengelmann/RETFound_Green)

### (b) Loader path

The official loader call documented in the GitHub README and arXiv 2405.00117:

```python
import timm

model = timm.create_model(
    "vit_small_patch14_reg4_dinov2",
    img_size=(392, 392),
    num_classes=0,
    checkpoint_path="retfoundgreen_statedict.pth"
)
model.global_pool = "avg"
model.eval()
```

**Not already supported in embeddings.py.** The current `src/retina_screen/embeddings.py` uses only:
- `torchvision` for ResNet-50 and ConvNeXt-Base
- `torch.hub` for DINOv2 variants

There is no `timm` import or timm loader branch. B2 must add timm as a project dependency and implement a new loader code path. See §5 for B2 loader requirements.

**timm availability check result (2026-05-24):** `timm` is not installed in the current project environment (`ModuleNotFoundError`). B2 must add timm to the project dependencies before extraction.

### (c) Embedding dimensionality

- **Embedding dim: 384** (ViT-Small architecture, patch size 14, register tokens 4)
- Model identifier: `vit_small_patch14_reg4_dinov2`
- Source: arXiv 2405.00117 architecture description; Nature Comms 2025

**Important mismatch:** The existing `configs/backbone/retfound.yaml` has `embedding_dim: 1024`, which corresponds to a ViT-Large variant (likely original RETFound or RETFound-MEH). RETFound-Green is 384-dim. Do NOT reuse `retfound.yaml` for RETFound-Green. See §7.

### (d) Input resolution and pooling

- **Native resolution:** 392×392 pixels, avg pooling over patch tokens
- **Matrix primary protocol (B2):** matched-224 / default_224 preprocessing / matrix pooling protocol
- **Subdecision resolved:** matched-224 is the B2 primary protocol. Native-392/avg-pool is conditionally deferred.

**timm positional-embedding interpolation note:** When loading weights trained at 392×392 and performing inference at 224×224, timm automatically interpolates the positional embeddings to match the new grid size. This is expected and correct behaviour for matched-224 protocol, but it introduces a mild positional-encoding mismatch relative to the native 392×392 training configuration. This does not invalidate matched-224 results — it is the same interpolation used for DINOv2/ViT fine-tuning at non-native resolutions — but it should be acknowledged when comparing B2 results to published RETFound-Green native-protocol numbers.

### (e) Pretraining-corpus contamination check

**Pretraining corpus:** ~75,000 publicly available retinal images  
**Named datasets:** AIROGS (53,327 images, random sample), DDR (full dataset), ODIR-2019 (full dataset)  
**Source:** arXiv 2405.00117 §2.1

| Dataset | In RETFound-Green pretraining? | Strict contamination list? | Verdict |
|---------|-------------------------------|---------------------------|---------|
| BRSET | No | Yes | ✅ clean |
| mBRSET | No | Yes | ✅ clean |
| IDRiD | No | Yes | ✅ clean |
| Messidor / Messidor-2 | No | Yes | ✅ clean |
| APTOS / APTOS-2019 | No | Yes | ✅ clean |
| RFMiD | No | Yes | ✅ clean |
| ODIR-2019 | **Yes** | No (not on strict list) | ⚠️ caveat |
| AIROGS | **Yes** | No (not on strict list) | ⚠️ caveat |
| DDR | **Yes** | No (not on strict list) | ⚠️ caveat |

**Contamination verdict for BRSET primary matrix: `not_blocked`**  
BRSET, mBRSET, and the six strict-list datasets are absent from RETFound-Green pretraining.

**Pretraining caveat — ODIR (critical):**  
RETFound-Green pretraining includes ODIR-2019. This project uses ODIR-5K, which is the same dataset family (same source/authors, same ophthalmological disease labelling scheme, overlapping population). RETFound-Green **must NOT** be used as a frozen backbone for external validation on ODIR in any form — ODIR-2019, ODIR-5K, or any ODIR-derived split — because this would constitute pretraining/evaluation overlap and invalidate any reported generalization claim.

**Pretraining caveat — AIROGS:**  
RETFound-Green pretraining includes AIROGS. If the project later adopts an AIROGS-based evaluation dataset (e.g., EyePACS-AIROGS-light-V2 is in the dataset inventory), RETFound-Green should not be evaluated on that dataset without an explicit contamination decision.

**Pretraining caveat — DDR:**  
DDR is in RETFound-Green pretraining. DDR is not currently listed as a project evaluation dataset. Do not claim DDR is a project dataset unless project docs explicitly say so.

### (f) Source paper BRSET evaluation

- **Paper reports BRSET:** Yes — extensive evaluation in Nature Communications 2025
- **Protocol:** native-392px input, avg-pool over patch tokens (native RETFound-Green protocol)
- **Reported performance:** 68/119 downstream task wins vs DERETFound; 68/119 wins vs RETFound-MEH
- **Comparability to matched-224 B2:** **Not directly comparable.** Published numbers use native-392/avg-pool; B2 uses matched-224/matrix-pooling. The protocol difference will likely affect absolute AUROC values. B2 results should be reported as "matched-224 protocol" and not compared to the paper's native-protocol numbers without explicit qualification.

---

## §5 timm Loader Requirements for B2

`timm` (PyTorch Image Models) is a separate Python library — distinct from `torchvision`, `torch.hub`, and HuggingFace `transformers`. The current project does not depend on timm and it is not installed.

B2 must implement all of the following before RETFound-Green embedding extraction can run:

1. **timm dependency**: Add `timm` to project dependencies (pyproject.toml / requirements). Verify the minimum version that includes `vit_small_patch14_reg4_dinov2`.
2. **state_dict loading**: The checkpoint is a raw state dict (not a timm checkpoint dict). Use `timm.create_model(..., checkpoint_path=...)` which handles this format, or load manually with `model.load_state_dict(torch.load(...), strict=True)`.
3. **Embedding dim enforcement**: Assert output shape `(384,)` after `model.global_pool = "avg"` and inference. If shape is wrong, halt — do not silently use wrong-dim embeddings.
4. **Input resolution enforcement**: Force `img_size=(224, 224)` for matched-224 B2 primary row. Do not default to 392×392 unless native-392 is explicitly requested.
5. **Pooling control**: For matched-224, use the default timm global pooling (avg or CLS via `model.global_pool`). Document exactly which pooling is active.
6. **Cache namespace**: New cache namespace must encode `retfound_green` + `matched224` + preprocessing hash. Must not collide with existing DINOv2 or ResNet caches.
7. **W1 storage/compaction re-verification**: Re-verify that `_compact_embedding()` is called on the RETFound-Green embedding to prevent the W1 storage-view bug (Decision 028).
8. **Shared MT+LP cache**: MultiTaskHead and LinearProbeHead must consume the same RETFound-Green cached embeddings. Cache must not be re-extracted per head.

---

## §6 Native-392 vs Matched-224 Protocol

| Property | Native protocol | Matched-224 (B2 primary) |
|----------|----------------|--------------------------|
| Input resolution | 392×392 | 224×224 |
| Pooling | avg over patch tokens | Matrix protocol (default) |
| timm config | img_size=(392,392) | img_size=(224,224) |
| Positional embeddings | Native (trained) | Interpolated from 392×392 |
| Published BRSET numbers | Available (68/119 wins) | Not yet available (B2 pending) |
| Comparability to matrix | Low — protocol differs | High — same protocol as DINOv2/ConvNeXt/ResNet |

**Research rationale for matched-224 as primary:** B2's scientific question is whether retinal-specific pretraining generalises under a strict cross-backbone matched protocol, not whether it performs best under its native configuration. Running at 224×224 with the same preprocessing as other matrix backbones isolates the pretraining-domain effect from the resolution and pooling effect. This is the harder test.

**Native-392 deferred:** Running native-392/avg-pool as a secondary row (after B2 matched-224 is complete) is conditionally permitted if user explicitly decides. It should be reported as a separate protocol row, not as the primary B2 result.

---

## §7 Existing retfound.yaml Mismatch

`configs/backbone/retfound.yaml` exists in the repository with `embedding_dim: 1024`. This value is incorrect for RETFound-Green, which is a ViT-Small model with embedding dim 384. The existing config likely corresponds to the original RETFound (Zhou et al., Nature 2023) or RETFound-MEH variant, both of which use ViT-Large (1024-dim).

**B2 action required:** Create a new config file, suggested name `configs/backbone/retfound_green_matched224.yaml`, with at minimum:
```yaml
name: retfound_green
model_type: retfound_green   # new type — requires new loader branch in embeddings.py
source: timm
model_identifier: vit_small_patch14_reg4_dinov2
version: retfound_green_v0.1
embedding_dim: 384
input_size: 224
frozen: true
mean: [0.5, 0.5, 0.5]   # RETFound-Green normalization (confirm against paper)
std: [0.5, 0.5, 0.5]
```

Do NOT modify or overwrite `configs/backbone/retfound.yaml`. It should remain as-is (deferred original RETFound config) to avoid confusion with future RETFound-MEH work.

---

## §8 B2 Forward-Routing Instructions

**B2 primary protocol:** matched-224 / default_224 / matrix protocol (same as DINOv2, ConvNeXt, ResNet)  
**B2 evaluation:** BRSET primary only (same split, same heads, same unweighted recipe as Decision 027)  
**B2 backbone config:** create new `configs/backbone/retfound_green_matched224.yaml` (see §7)  
**B2 loader:** add timm loader branch to `src/retina_screen/embeddings.py`; see §5 requirements  
**B2 external validation exclusion:** do NOT evaluate RETFound-Green on ODIR in any form (ODIR-5K, ODIR-2019, or any ODIR derivative); see §4(e)  
**B2 heads:** MultiTaskHead and LinearProbeHead (same as other matrix cells)  
**B2 cache:** new namespace; shared by MT and LP; W1 compaction verified  
**B2 native-392:** deferred; do not run by default  

---

## §9 DINOv3-Large — Out of Scope

DINOv3-Large (Siméoni et al., Meta, arXiv 2508.10104) remains `BLOCKED_ACCESS` due to HuggingFace gated access (login + contact-sharing agreement via Meta Privacy Policy required; not verified as satisfied). It was verified in the B1 exploration phase as technically suitable (1024-dim ViT-L, clean_with_caveat pretraining on LVD-1689M, no BRSET numbers). No F16 is created. B3 requires a separate prompt once the user confirms HuggingFace access.

---

## §10 Out-of-Scope Candidates (one-line each)

- **FLAIR**: Rejected (B4) — contamination with BRSET/evaluation-surface datasets.
- **DINOv2-Giant**: Rejected (B5) — internal scaling curve reasoning; DINOv3-Large is the next-generation comparator.
- **RETFound-MEH**: Access-pending / deferred — not rejected; not verified in this round.

---

## §11 Confirmation Checklist

- [x] No embedding extraction performed
- [x] No model training performed
- [x] No evaluation performed
- [x] No A1 bootstrap CI run or re-run
- [x] No src/retina_screen/ files modified
- [x] No tests/ files modified
- [x] No configs/ files modified
- [x] configs/backbone/retfound.yaml not modified
- [x] configs/backbone/retfound_green.yaml not created (B2 scope)
- [x] No backbone weights downloaded
- [x] No HuggingFace access gate accepted
- [x] No DINOv3 code/config/verification work performed
- [x] No F16 created
- [x] outputs/ artifacts not committed
- [x] Only committed files: docs/verifications/B1_backbone_candidates_verification.md, docs/paper_framing_and_findings.md (F10 only), docs/decisions.md (Decision 029 appended)
