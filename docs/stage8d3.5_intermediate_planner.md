

# SOME IMPORTANT POINTS TO CONSIDER WHILE IMPLEMENTATION:
### Verification result

**Confirmed true:**

RETFound-Green is openly released with no access gate. The weights are a direct download from a public GitHub release: `https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth`. No credentialed application, no waitlist, unlike RETFound-MEH. Published in Nature Communications 2025 (vol 16, article 6862); preprint arXiv 2405.00117.

It was evaluated on BRSET in its own paper, including diabetic retinopathy grading on BRSET specifically. The paper reports BRSET diverse-task results and BRSET DR grading (whole population and diabetic-only subset), using median-of-100-bootstrap-samples with Wilcoxon signed-rank significance testing — which is methodologically close to your pipeline's frozen-embedding + bootstrap-CI approach and is directly positionable against your matrix.

It is architecturally a DINOv2 variant. The load snippet is `timm.create_model('vit_small_patch14_reg4_dinov2', img_size=(392,392), num_classes=0, checkpoint_path='retfoundgreen_statedict.pth')` with `global_pool='avg'`. This is important and the summary document did not flag it: RETFound-Green is a ViT-Small with DINOv2-style architecture and registers, **trained at 392×392 with average pooling**, not CLS-token, not 224×224.

**Where the summary document was loose / where you need to be careful:**

The summary doc said RETFound-Green "outperformed RETFound-MEH on the BRSET diverse-task panel." The actual paper result is more nuanced: on BRSET diverse tasks, RETFound-Green had 5 significant wins vs DERETFound's 4; on BRSET DR grading (whole population) RETFound-Green 5 wins / RETFound-MEH 1. So it does generally beat RETFound-MEH on BRSET, but the headline is "generally best, followed by DERETFound," not a clean sweep. Don't carry the summary doc's slightly stronger phrasing into your decisions log — cite the actual win-counts.

Two integration consequences the summary document completely missed, and they matter for your planner:

First, **RETFound-Green is not a drop-in under your locked preprocessing.** Your entire matrix uses `default_224` (224×224, preprocessing_hash `92d0f40b94aea26c`). RETFound-Green's published configuration is 392×392 with average pooling. You have two choices and both need to be a recorded decision: (a) run RETFound-Green at 224 to match your matrix's controlled preprocessing — keeps the comparison clean but is _off_ the model's intended configuration and the numbers won't match its paper; or (b) run it at 392 — matches the model's design and its published BRSET numbers but introduces a preprocessing variable that breaks strict comparability with the other four backbones. This is a genuine methodological fork that the summary doc's "one-day insertion" framing obscured. It is not one day and it is not free; it's a deliberate decision with a real tradeoff.

Second, there is an external reference point you didn't know about: the official BRSET repository itself publishes precomputed embeddings for `dinov2_small/base/large/giant`, `convnext_base/large`, `vit_base/large`, and `retfound`. This is relevant for sanity-checking your own extraction (do your DINOv2-Large BRSET embeddings behave consistently with the reference?) and as a positioning citation, not as a substitute for your pipeline.

So B1 resolves as: **retinal-specific backbone is confirmed available and BRSET-relevant — RETFound-Green, openly downloadable.** But B2 carries a mandatory sub-decision (224 vs 392 / CLS vs avg-pool) that must be made and recorded before extraction, and the honest framing is "RETFound-Green at matched-224 as the controlled matrix cell, with its native-392 configuration as an optional secondary row if the comparison demands it" — not an unqualified one-day insertion.






**Status:** Active. **Position:** Inserted between Stage 8D-3E (complete, audited) and Stage 8D-4 (result lock) of the original planner. **Does not replace any original-planner stage.** **Purpose:** Establish statistical defensibility, complete the backbone matrix with a retinal-specific model, and correct the dr_grade framing — before the matrix is frozen at 8D-4 — so the controlled baseline that 8D-4 locks is the _correct_ baseline. All work here is post-8D-3E and pre-8D-4. No work here modifies the locked training recipe (Decision 027) for any matrix cell.

### Category A — Statistical defensibility (gates everything; no training)

**A1. Bootstrap confidence intervals across the full matrix.** Per-task and macro bootstrap CIs (stratified resampling of the test set, fixed seed) for all eight existing cells (4 backbones × 2 heads). Paired bootstrap (or DeLong for AUROC) for the head deltas (MT−LP per backbone) and the backbone deltas (per head). Output: a CI table that states, for every ordering claim currently in the matrix, whether the difference's CI excludes zero. Pure post-hoc analysis on existing prediction artifacts; no retraining, no cache changes, no recipe changes. **Exit criterion:** every matrix cell and every pairwise delta has a reported CI; the set of orderings that are statistically supported vs not is explicitly enumerated. **Why first:** if Base≈Large or MT>LP or DINOv2>ConvNeXt is not significant, that changes how the matrix is described at 8D-4 lock and in the paper. Doing this after the lock would mean locking claims that may not survive.

### Category B — Matrix completeness (one new backbone class; verification-gated)

**B1. Verify availability and BRSET-relevance of a retinal-specific foundation backbone.** Before any extraction: confirm, from primary sources, the actual release/access status and the published evaluation protocol of the candidate retinal-specific backbone(s) (RETFound-Green and FLAIR are the candidates; RETFound-MEH remains access-pending and is not the candidate unless weights arrive). Specifically verify: weights are genuinely downloadable without a gated process; the model's expected embedding dimensionality and loader path; whether the source paper itself reports BRSET linear-probe numbers we can position against. This is a read/verify task, not an assumption. **Exit criterion:** a written verification note stating which retinal-specific backbone(s) are confirmed-available with confirmed loader details, or a BLOCKED note if none are. **Deferral branch:** if no retinal-specific backbone is confirmable, B2 is deferred (not failed); reactivation condition = RETFound-MEH weights become available OR a retinal-specific backbone's open release is confirmed. The matrix is then locked at 8D-4 as a four-backbone _generic-vision_ matrix with an explicit stated limitation that no retinal-pretrained comparator was obtainable at lock time.

**B2. Add the verified retinal-specific backbone as the fifth matrix row, under the exact locked recipe.** Same four-prompt pattern as every prior cell (preflight → run → strict review → Codex audit). Same Decision 027 unweighted recipe, same preprocessing hash, same split, same final_test_result=false. MT and LP both. New cache namespace; W1 compaction applies (verify per the established storage gate). Contingent on B1 success. **Exit criterion:** fifth-row MT and LP cells produced, audited, and CI'd (re-run A1's procedure for the new row only). **Validation against original planner:** this extends the Stage 8D-3 matrix; it does not touch 8E or later. It is the only new extraction authorized by this planner.

### Category C — dr_grade reframing and findings extraction (no training; existing artifacts)

**C1. Recast dr_grade as referable-DR binary (grades 0–1 vs 2–4) as the primary DR endpoint.** Computed from existing predictions/labels on existing artifacts for all matrix cells (including B2's if it exists). Report referable-DR AUROC with the A1 bootstrap CI procedure applied. The 5-class result is retained but demoted to a documented-limitation table with accuracy, macro_f1, and balanced_accuracy always shown together. **Exit criterion:** referable-DR binary metric with CIs present for every matrix cell; 5-class limitation table present.

**C2. Document the monotone ordinal balanced-accuracy finding.** Write up the cross-backbone balanced-accuracy trend (improves with backbone quality despite classes 1/2 never being predicted) as a stated finding, with the per-backbone numbers and the interpretation (embeddings encode ordinal severity the argmax head cannot surface). Zero compute; this is a findings-documentation task. **Exit criterion:** finding recorded in the results documentation with supporting numbers and the explicit interpretation.

**C3. Document the MT−LP separability analysis including the non-monotone correction.** Record that the MT−LP gap is not monotone in backbone strength (ResNet +0.044, ConvNeXt +0.046, DINOv2-Base +0.024, DINOv2-Large +0.034) and that DINOv2-Base is a linear-separability sweet spot, scaling past which reduces linear accessibility. Explicitly retract the earlier "gap narrows monotonically" framing so it cannot survive into a draft. Zero compute. **Exit criterion:** corrected separability finding recorded; prior monotone claim explicitly marked retracted in the decisions/findings log.

### Category D — Deferred items (explicitly recorded, not silently dropped)

Each is recorded in the planner doc with reason and reactivation condition. None is executed in Stage 8D-3.5.

**D1. CORAL/ordinal loss for dr_grade (post-lock ablation).** Deferred because it is a recipe change and must not contaminate the locked matrix. Reactivation: after Stage 8D-4 lock, as a clearly-labelled separate ablation, only if C1's referable-DR result is strong enough that an ordinal 5-class secondary result adds value. Run exactly one ordinal method; do not also run focal loss (redundant).

**D2. dr_grade-only class weighting.** Deferred. Reactivation: only as a fallback if D1 is activated and underperforms; carries known precision-collapse risk; documented as fallback-of-fallback.

**D3. CLAHE/Graham preprocessing ablation.** Deferred. Reason: full re-extraction per backbone for a marginal expected gain (~0.01) concentrated on the deprioritized 5-class task. Reactivation: only if external validation (Stage 8E) shows a transfer gap plausibly attributable to illumination/contrast domain shift, in which case it becomes a targeted single-backbone diagnostic, not a matrix-wide re-extraction.

**D4. Patch-token pooling / multi-layer features.** Deferred. Reason: requires full re-extraction storing the full token sequence (re-confronts the W1 storage problem at larger scale) and a change to the embeddings-layer contract; this is a Stage-8E-or-later decision with its own decision-log entry, not a head experiment. Reactivation: explicit future-work decision only; recorded as a named limitation in the paper regardless.

**D5. Higher-resolution (384/448) extraction.** Deferred. Reason: full re-extraction; largest expected effect is on the deprioritized 5-class DR task. Reactivation: only jointly with D3 if a resolution-sensitivity question becomes paper-critical.

**D6. Standalone fine-tuning reference point.** Deferred as a standalone task and explicitly merged into the continual-learning component's design (original planner). Reason: a one-off fine-tune and the LoRA continual-learning experiment both adapt beyond frozen; running them separately is redundant. Reactivation: not as a separate task — the "cost of the frozen design" quantification becomes the frozen-baseline arm of the continual-learning experiment when that original-planner stage runs.

**D7. Bigger/deeper head, more epochs, optimizer tweaks.** Permanently declined (not merely deferred). Reason: matrix consistency across four backbones demonstrates training mechanics are not the bottleneck; both advisory documents independently concur. No reactivation condition.

### Category E — Lock handoff (routes into the original planner; does not duplicate it)

**E1. Stage 8D-3.5 closeout and handoff to Stage 8D-4.** Confirm A1, C1, C2, C3 complete and B-category resolved (either B2 done and audited, or B1's deferral note written). Update `docs/` surgically: this planner added as the intermediate deviation record; decisions log updated with the CI procedure, the referable-DR primary-endpoint decision, the dr_grade 5-class demotion, the retracted monotone claim, and every D-item deferral with its reactivation condition. No new decision supersedes Decision 027; the locked recipe is unchanged. **Exit criterion:** Stage 8D-4 (original planner) can now run with the matrix in its correct, CI-annotated, correctly-framed final form. Stage 8D-3.5 is closed.

**Validation against original planner — explicit non-overlap statement (must be recorded in the doc):** Stage 8D-3.5 does not perform external validation (owned by Stage 8E), does not implement the fairness audit (owned by its original-planner stage), does not implement LoRA/continual learning (owned by its original-planner stage), does not build the dashboard, and does not perform explainability work. It only: adds statistical rigor to the existing matrix, optionally completes the matrix with one retinal-specific row, corrects the dr_grade framing, and extracts findings already latent in existing artifacts. Stage 8E inherits one constraint from here: external results must be reported on the same CI basis (A1 procedure) and the same referable-DR primary endpoint (C1), so transfer is measured on the corrected framing, not the old one.

---

That's the skeleton. It's deliberately small: five execution tasks (A1, B1, B2, C1, C2/C3 grouped, E1), seven explicit deferrals with reactivation conditions, and a hard non-overlap statement against your original planner so nothing here duplicates 8E or the later components.