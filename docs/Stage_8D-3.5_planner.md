# Stage 8D-3.5 — Intermediate Planner (Updated)

**Status:** Active. **Position:** Inserted between Stage 8D-3E (complete, audited) and Stage 8D-4 (result lock and visual representation) of the original planner. **Does not replace any original-planner stage.** **Purpose:** Establish statistical defensibility, complete the backbone matrix with two retinal-relevant and one new-generation generic backbone, correct the dr_grade framing, and record the post-lock 2-D ablation grid plan — before the matrix is frozen at 8D-4 — so the controlled baseline that 8D-4 locks is the _correct_ baseline. All work in Categories A–E is post-8D-3E and pre-8D-4. Category F is recorded here but executed only after 8D-4 lock. No work here modifies the locked training recipe (Decision 027) for any matrix cell.

**Reasoning-capture rule (applies to every category below):** every decision recorded in this planner — including additions, rejections, deferrals, sub-decisions, and methodological forks — must be captured in `docs/paper_framing_and_findings.md` as a numbered entry at implementation time. The planner tells the implementer _what to do_; the framing-and-findings doc tells the paper _why_. The implementer's job is to copy the reasoning across as the action is taken, not to re-derive or re-debate it.

---

### Category A — Statistical defensibility (gates everything; no training)

**A1. Bootstrap confidence intervals across the full matrix.** Per-task and macro bootstrap CIs (stratified resampling of the test set, fixed seed, 1000 resamples minimum) for every existing matrix cell, and the same procedure re-run on every new cell B2/B3 produces. Paired bootstrap (or DeLong for AUROC where strictly appropriate) for the head deltas (MT−LP per backbone) and the backbone deltas (per head). Output: a CI table that states, for every ordering claim currently in the matrix, whether the difference's CI excludes zero. Pure post-hoc analysis on existing prediction artifacts; no retraining, no cache changes, no recipe changes. **Exit criterion:** every matrix cell and every pairwise delta has a reported CI; the set of orderings that are statistically supported vs not is explicitly enumerated. **Why first:** if Base≈Large or MT>LP or DINOv2>ConvNeXt is not significant, that changes how the matrix is described at 8D-4 lock and in the paper. Doing this after the lock would mean locking claims that may not survive. **Reasoning capture:** finding F8 in `paper_framing_and_findings.md` (statistical-rigor gate) already records the requirement; A1 satisfies it.

---

### Category B — Matrix completeness (verification-gated; up to two new backbone classes)

**B1. Verify availability, BRSET-relevance, and pretraining-set composition of every candidate retinal-relevant or new-generation backbone.** This is the gating verification before any extraction. Read primary sources; do not assume.

For each candidate, the verification must produce: (a) confirmed downloadable weights with no access gate, (b) confirmed loader path (timm/HF/custom), (c) confirmed expected embedding dimensionality, (d) confirmed expected input resolution and pooling strategy, (e) **explicit check of the model's pretraining corpus against BRSET, mBRSET, IDRiD, Messidor-2, APTOS, and RFMiD** — any overlap is a contamination flag that blocks inclusion as a frozen-feature matrix row (see B4 rejection rule below), and (f) whether the source paper itself reports BRSET numbers we can position against.

**Candidates confirmed in this planner round:**

- RETFound-Green (Engelmann & Bernabeu, Nat Comms 2025, arXiv 2405.00117). Confirmed open release, ViT-S DINOv2-architecture with reg4, published at 392×392 with avg pooling. Pretraining set: not contaminated against BRSET.
- DINOv3-Large (Siméoni et al., Meta, August 2025, arXiv 2508.10104). Confirmed open release on HF (`facebook/dinov3-vitl16-pretrain-lvd1689m`), 300M params (identical scale to DINOv2-Large), 16×16 patch, 1024-dim, RoPE positional embeddings, 4 register tokens. Pretraining set: LVD-1689M natural images, not contaminated against BRSET.

**Candidates explicitly rejected in this planner round:**

- FLAIR (Silva-Rodríguez et al., MedIA 2025, arXiv 2308.07898) — see B4.
- DINOv2-Giant — see B5.
- RETFound-MEH — access-pending; not rejected, but not in this planner round. Reactivation: if weights become available, add as a future planner round under the same B-pattern.

**Exit criterion:** written verification note for RETFound-Green and DINOv3-Large stating loader path, embedding dim, pretraining-set check result. Reasoning capture: finding F10 (RETFound-Green positioning) updated from DEFERRED-PENDING-SUBDECISION to ACTIVE; new finding F16 (DINOv3-Large positioning) created.

---

**B2. Add RETFound-Green as the fifth matrix row, under the exact locked recipe.** Same four-prompt pattern as every prior cell (preflight → run → strict review → Codex audit). Same Decision 027 unweighted recipe, same split, same `final_test_result=false`. MT and LP both. New cache namespace; W1 compaction applies (verify per the established storage gate).

**Mandatory preprocessing sub-decision (resolved here):** RETFound-Green's native configuration is 392×392 with avg-pool, but the matrix's locked preprocessing is `default_224` (`preprocessing_hash=92d0f40b94aea26c`) with CLS-style global pooling. The fork:

- _Matched-224/CLS as primary cell._ Preserves strict cross-backbone comparability with the existing four-row matrix. Numbers will not match RETFound-Green's published BRSET numbers (which are at 392/avg-pool). This is the cell that goes into Table 1.
- _Native-392/avg-pool as optional secondary cell._ Run only if Codex audit confirms the matched-224 cell's numbers diverge from RETFound-Green's published BRSET numbers by an amount large enough to need explanation. New preprocessing namespace, new cache, marked clearly as "off-protocol comparator-only cell" in any table it appears in.

**Decision: matched-224 as the primary B2 cell.** The native-392 secondary cell is deferred — run only on Codex-audit recommendation, not by default. The paper text must explicitly state the matched-protocol choice and the resulting expected gap vs published RETFound-Green BRSET numbers.

**Exit criterion:** fifth-row MT and LP cells produced at matched-224, audited, and CI'd (re-run A1 procedure for the new row). Reasoning capture: finding F10 updated with the matched-224 decision recorded and the native-392 fork explicitly marked as conditionally-deferred.

---

**B3. Add DINOv3-Large as the sixth matrix row, under the exact locked recipe.** Same four-prompt pattern, same locked recipe. Run after B2 or in parallel; both are independent. MT and LP both. New cache namespace; W1 compaction applies.

DINOV3-L/16 weights have been manually downloaded and are present in the path: C:\retinal_fundus_to_systemic_screening\models\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth and this was done after this planner was made so make adjustments accordingly i dont need any clash in the retrieval of weights

The B3 implementation should load the local file rather than going through `AutoModel.from_pretrained` — which means the loader in `embeddings.py` should be either a `torch.load` of the state-dict directly into an instantiated ViT, or `timm.create_model('vit_large_patch16_dinov3', ..., checkpoint_path=<local path>)` once timm has DINOv3 support. (Timm did add DINOv3 model entries shortly after the August 2025 release; whether your project's timm version is recent enough is something the B3 preflight must check.)

**Preprocessing note (no fork required):** DINOv3-Large is designed to operate at multiple resolutions including 224×224 with patch-16, which yields 196 patch tokens + 1 CLS + 4 register tokens = 201 tokens. This is its native configuration at 224, not an off-protocol downscaling. The 16×16 patch (vs DINOv2's 14×14) is recorded as a preprocessing-namespace note in the cache provenance but does not require a methodological fork — both DINOv2-Large and DINOv3-Large are being evaluated in their respective native 224 configurations under the same `default_224` preprocessing hash.

**Expected extraction cost:** roughly equivalent to or slightly cheaper than DINOv2-Large (~3 hours CPU). DINOv3-L has fewer tokens at 224 than DINOv2-L (201 vs 261), and attention scales quadratically in token count, so per-image cost is comparable or slightly lower despite identical parameter count. Do not over-budget for B3.

**Specific scientific question B3 answers:** does the _pretraining-data-and-method advance_ (1.7B images, Gram anchoring, RoPE) outperform _parameter-scale increases within an older paradigm_ (DINOv2-Large at the same 300M params)? Zhou et al. (arXiv 2509.03421, Sep 2025) report DINOv3-Large beats DINOv2-Giant on retinal fine-tuning tasks despite being 3.5× smaller; B3 replicates the comparable observation on BRSET under frozen extraction with multi-task heads — a different protocol on the same architectural question.

**Exit criterion:** sixth-row MT and LP cells produced at 224 native, audited, and CI'd. Reasoning capture: new finding F16 (DINOv3-Large positioning and rationale) created.

---

**B4. FLAIR — explicitly REJECTED from the matrix (and from any frozen-feature evaluation in this project).** This rejection is recorded permanently with no reactivation condition for frozen-feature use.

**Reason:** FLAIR's pretraining corpus is an assembly of 38 open fundus datasets. The published dataset list (`github.com/jusiro/FLAIR` README) explicitly includes BRSET (dataset 27), IDRiD (03), Messidor (02), APTOS (15), and RFMiD (04). Every dataset in this project's evaluation scope — BRSET as the primary, and IDRiD/Messidor-2/APTOS/RFMiD as the external-validation arm — was seen by FLAIR during pretraining, paired with category labels overlapping this project's task labels.

**Consequence:** FLAIR has no held-out evaluation surface anywhere in this project's data scope. Including it as a matrix row would compare a model on its training distribution against models on a true held-out test set, which is not a meaningful comparison. Including it on the external-validation arm would not test generalization — those datasets are also in its training set.

**Conditional reactivation (narrow and separate):** FLAIR may appear in the paper _only_ as a zero-shot text-prediction baseline in a clearly-fenced "vision-language models" experiment, with explicit language stating BRSET and all external-validation datasets are in FLAIR's pretraining set. This is a separate experiment, not a matrix row, and not part of this planner. Decision on whether to run it at all is deferred to post-lock.

**Reasoning capture:** new finding F17 (FLAIR rejection with contamination evidence) created in `paper_framing_and_findings.md` so the rejection is locked in and cannot re-enter a draft from stale notes. The reasoning must be stored verbatim — every external-validation dataset name listed — so that any future advisor proposing to add FLAIR is met with the specific contamination list.

---

**B5. DINOv2-Giant — explicitly REJECTED from the matrix.** Recorded permanently.

**Reason:** Two independent lines of evidence converge on the same conclusion. First, the internal DINOv2 Base→Large result on BRSET already showed the scaling curve within the DINOv2 family is flat at Large (+0.0058 MT macro-AUROC, LP regresses; the gain is not statistically established). Scaling further to Giant within the same family, same pretraining objective, same LVD-142M corpus is predicted to yield a similar near-null result at substantially higher cost. Second, Zhou et al. (arXiv 2509.03421) directly compared DINOv2-Giant (1.1B params) against DINOv3-Large (300M) on retinal fine-tuning tasks and report DINOv3-Large outperforms DINOv2-Giant despite being 3.5× smaller, while DINOv2-Giant required ~3.5× more GPU memory and 2× the training time.

**Consequence:** B3's addition of DINOv3-Large already answers the "what happens past DINOv2-Large?" question better than DINOv2-Giant would, at lower cost. Adding DINOv2-Giant would consume ~10–14 hours of CPU extraction and produce a 4096-dim embedding store for a predicted-inferior result.

**Reasoning capture:** new finding F18 (DINOv2-Giant rejection with scaling-already-flat and DINOv3-L-superior evidence) created.

---

### Category C — dr_grade reframing and findings extraction (no training; existing artifacts)

**C1. Recast dr_grade as referable-DR binary (grades 0–1 vs 2–4) as the primary DR endpoint.** Computed from existing predictions and labels on existing artifacts for _all six_ matrix cells once B2 and B3 are complete. Report referable-DR AUROC with the A1 bootstrap CI procedure applied. The 5-class result is retained but demoted to a documented-limitation table with accuracy, macro_f1, and balanced_accuracy always reported together. **Exit criterion:** referable-DR binary metric with CIs present for every matrix cell (six rows × two heads); 5-class limitation table present. Reasoning capture: finding F4 already records the decision.

**C2. Document the monotone ordinal balanced-accuracy finding.** Cross-backbone balanced-accuracy trend (improves with backbone quality despite classes 1 and 2 never being predicted) as a stated finding, with per-backbone numbers and the interpretation (embeddings encode ordinal severity the unweighted argmax head cannot surface as class predictions). Extend the trend across the new B2 and B3 rows once they exist — does the monotone improvement continue, plateau, or break with retinal-specific (RETFound-Green) and next-generation generic (DINOv3-L) backbones? This is an additional finding C2 produces beyond what F5 currently records. Zero compute. **Exit criterion:** finding recorded in `paper_framing_and_findings.md` with the extended cross-six-backbones trend.

**C3. Document the MT−LP separability analysis including the non-monotone correction.** Existing finding (ResNet +0.044, ConvNeXt +0.046, DINOv2-Base +0.024, DINOv2-Large +0.034) extended across B2 and B3 cells. Whether DINOv2-Base remains the linear-separability sweet spot when RETFound-Green and DINOv3-L enter the picture is itself a finding; record whichever direction the new data shows. Explicitly retract the earlier "gap narrows monotonically" framing so it cannot survive into a draft. Zero compute. **Exit criterion:** corrected separability finding recorded; prior monotone claim explicitly marked retracted in finding F6.

---

### Category D — Pre-lock deferred items (recorded, not silently dropped)

Each is recorded in the planner with reason and reactivation condition. None is executed in Stage 8D-3.5.

**D1. CORAL/ordinal loss for dr_grade.** Deferred to post-lock (Category F). Reason: recipe change, must not contaminate the locked matrix. Reactivation: F3 in the post-lock ablation track.

**D2. dr_grade-only class weighting.** Deferred. Reactivation: only as a fallback if F3 (CORAL) is run and underperforms; carries known precision-collapse risk; documented as fallback-of-fallback.

**D3. CLAHE/Graham preprocessing ablation.** Deferred. Reason: full re-extraction per backbone for a marginal expected gain (~0.01) concentrated on the deprioritized 5-class task. Reactivation: only if Stage 8E (external validation) shows a transfer gap plausibly attributable to illumination/contrast domain shift, in which case it becomes a targeted single-backbone diagnostic, not a matrix-wide re-extraction.

**D4. Patch-token pooling / multi-layer features.** Deferred. Reason: requires full re-extraction storing the full token sequence (re-confronts the W1 storage problem at larger scale) and a change to the embeddings-layer contract; this is a Stage-8E-or-later decision with its own decision-log entry, not a head experiment. Reactivation: explicit future-work decision only; recorded as a named limitation in the paper regardless.

**D5. Higher-resolution (384/448) extraction.** Deferred. Reason: full re-extraction; largest expected effect is on the deprioritized 5-class DR task. Reactivation: only jointly with D3 if a resolution-sensitivity question becomes paper-critical, OR as the conditional native-392 RETFound-Green secondary cell described in B2.

**D6. Standalone fine-tuning reference point.** Deferred as a standalone task and explicitly merged into the continual-learning component's design (original planner). Reason: a one-off fine-tune and the LoRA continual-learning experiment both adapt beyond frozen; running them separately is redundant. Reactivation: not as a separate task — the "cost of the frozen design" quantification becomes the frozen-baseline arm of the continual-learning experiment when that original-planner stage runs. Reasoning capture: finding F12.

**D7. Bigger/deeper head, more epochs, optimizer tweaks.** Permanently declined (not merely deferred). Reason: matrix consistency across four backbones demonstrates training mechanics are not the bottleneck. No reactivation condition.

---

### Category E — Lock handoff (routes into the original planner; does not duplicate it)

**E1. Stage 8D-3.5 closeout and handoff to Stage 8D-4.** Confirm A1, B2, B3, C1, C2, C3 complete and audited. Update `docs/` surgically: this planner added as the intermediate deviation record; `docs/decisions.md` updated with the CI procedure decision, the referable-DR primary-endpoint decision, the dr_grade 5-class demotion, the retracted monotone claim, the matched-224 RETFound-Green decision, the DINOv3-Large 224-native decision, and every D-item deferral and B4/B5 rejection with its reasoning. `docs/paper_framing_and_findings.md` updated per the reasoning-capture references throughout this planner (F4 active, F5 extended, F6 extended-and-retraction-locked, F8 satisfied by A1, F10 updated, F16 created, F17 created, F18 created). No new decision supersedes Decision 027; the locked recipe is unchanged.

**Exit criterion:** Stage 8D-4 (original planner) can now run with the matrix in its correct six-row, CI-annotated, correctly-framed final form. Stage 8D-3.5 is closed.

**Validation against original planner — explicit non-overlap statement (must be recorded in the doc):** Stage 8D-3.5 does not perform external validation (owned by Stage 8E), does not implement the fairness audit (owned by its original-planner stage), does not implement LoRA/continual learning (owned by its original-planner stage), does not build the dashboard, and does not perform explainability work. It only: adds statistical rigor to the existing matrix, extends the matrix with RETFound-Green and DINOv3-Large under the locked recipe, corrects the dr_grade framing, extracts findings already latent in existing artifacts, and records the post-lock ablation plan. Stage 8E inherits one constraint from here: external results must be reported on the same CI basis (A1 procedure) and the same referable-DR primary endpoint (C1), so transfer is measured on the corrected framing, not the old one.

---

### Category F — Post-lock ablation track (recorded here, executed after 8D-4 lock)

**Structural framing:** the locked six-row matrix (4 generic + RETFound-Green + DINOv3-L) × 2 heads (LP, MT) is the controlled spine. Each F-item adds a head-axis variant — a new column in a 2-D ablation grid (backbone × head architecture). The spine is never modified. Each new column is run on the strongest 2–4 backbones only (DINOv2-Base, DINOv2-Large, RETFound-Green, DINOv3-Large — the "top backbones" set, defined and frozen at 8D-4 lock based on locked-matrix AUROC ordering; ResNet-50 and ConvNeXt-Base are not re-ablated). New head-config files are created with the exact same base architecture as the corresponding LP or MT head; no modification to existing head code, configs, training scripts, or trained checkpoints. Original matrix cells stand untouched as the comparison anchor.

**Execution discipline (same as every prior cell):** preflight → run → strict review → Codex audit. New cache namespaces only if a backbone-level change is involved (none of F1–F3 require this; only F4 might). Each F-item must produce a one-sentence finding statement that goes into `paper_framing_and_findings.md`; if it doesn't, it shouldn't be run.

---

**F1. Per-task single-task heads.** Highest priority post-lock ablation.

_What it is:_ one independently-trained head per task, on the same frozen embeddings as the locked matrix uses. Two head variants worth running: (a) `singletask_linearprobe_perTask` — seven independent linear probes (one per task), each trained on the same backbone's cached embeddings with the same Decision 027 recipe; (b) optionally `singletask_mlp_perTask` — same MT trunk architecture but with only one task's head wired at a time, trained per-task. Variant (a) is the minimum and the published-protocol comparison row; variant (b) is the cleaner controlled comparison against MT.

_What it tests:_ the negative-transfer hypothesis. Three possible outcomes, all publishable: MT > single-task on most tasks (positive transfer dominates), single-task > MT on dr_grade specifically (negative transfer is the dr_grade weakness, not representation), mixed results with task-feature interpretation.

_Expected direction (Doc 56 prediction, more grounded than neutral):_ MT ≥ single-task on most cells, with the sparse-positive tasks (amd 22 positives, hypertensive_retinopathy 30, macular_edema 33) most likely to _degrade_ under single-task because they lose the auxiliary regularization the binary-task suite provides to the shared trunk. The strongest single-task wins, if any, will be on the densest tasks (diabetes, drusen) where the shared trunk's auxiliary benefit is smallest. The dr_grade outcome is the key uncertainty — if single-task wins on dr_grade, that's the cleanest diagnostic available; if it doesn't, dr_grade weakness is confirmed as a representation/imbalance issue, not negative transfer.

_Scope:_ run on all four top backbones × both variants. Cost: minutes per cell on cached embeddings.

_Reasoning capture:_ new finding F19 (per-task single-task heads — purpose, prediction, finding) created.


we can also use custom settings for every head we train according to the task that we assign them like using coral or corn ordinal loss for ordinal dr based tasks and using gradnorm or kendall for the other binary tasks and taking inspiration from the foundation models that are working best for every task individually and then modify the individual heads to use a similar configuration instead of making them all the exact same simple mlp's ok nevermind i didn't realize that we are just training individual heads for each task and just placing them onto the foundation model(i think we should just use these individual task heads for the foundation model that is performing the best already) or should we train entire completely new neural networks individually for each task but that would mean training on all 16k images about 7 times that is not computationally feasible. or we can just train individual heads on top of our strongest model and then use a different configuration(different parameters and hyperparameters and different overall architecture like layers used config used loss used whatever else) for every single head instead of making them all the same i think this one is the best one but i could be wrong correct me if i am.

# since dinov3 doesnt has almost no papers in it being used for retinal data through our specific methodology which i am planning to make like extraction of embeddings by frozen weights and then training a separate mlp for every single task our results will become the benchmark for dinov3 on brset at least with our configuration.

---

**F2. GradNorm MT head variant.** Second priority post-lock ablation.

_What it is:_ a new MT head config (`multitask_gradnorm`) that swaps the current Kendall homoscedastic-uncertainty weighting (Decision 015) for GradNorm gradient-magnitude balancing. The trunk architecture, task heads, and Decision 027 training recipe are identical to the locked MT head; only the across-task loss-weighting mechanism changes.

_What it tests:_ whether the choice of multi-task weighting scheme materially affects outcomes on this task mix. Two outcomes, both publishable: GradNorm matches Kendall within CIs (loss-weighting-choice invariance — useful negative finding for the literature) or GradNorm changes the per-task balance (positive finding, with the specific shift documented).

_What it will not do:_ fix the dr_grade class 1/2 collapse. That's a within-task class imbalance problem; GradNorm is an across-task weighting mechanism. Do not over-frame F2 as a dr_grade fix.

_Scope:_ run on all four top backbones. MT head only (LP has no multi-task weighting to ablate). Cost: minutes per cell on cached embeddings.

_Reasoning capture:_ new finding F20 (GradNorm-vs-Kendall — purpose, scope, prediction caveat) created.

---

**F3. CORAL ordinal loss for dr_grade.** Third priority post-lock ablation.

_What it is:_ a new head config that replaces the standard cross-entropy classification head for dr_grade with a CORAL (Consistent Rank Logits) ordinal regression head, leaving all other tasks' heads unchanged. Trains under Decision 027 recipe on cached embeddings.

_What it tests:_ whether using the ordinal structure of DR grade (0 < 1 < 2 < 3 < 4) at the loss level rescues the 5-class performance enough to be reported as a secondary result. The referable-DR binary (C1) remains the primary endpoint either way.

_Scope:_ dr_grade only, on the top two backbones (DINOv2-Large + whichever of DINOv3-L / RETFound-Green wins on referable-DR at lock). Do not also run focal loss; the F3 finding is "ordinal modeling for dr_grade" and adding focal would muddle it. Do not run on every backbone; the question is whether ordinal modeling moves a strong cell, not a survey of every cell.

_Activation gate:_ run only if C1's referable-DR result at lock is strong enough that a credible 5-class secondary result would add value to the paper. If referable-DR is the only DR story the paper can credibly tell, F3 is skipped.

_Reasoning capture:_ new finding F21 (CORAL ordinal dr_grade — purpose, activation gate, scope) created.

also consider whether we should use coral or corn loss?

---

**F4. Feature ensembling across top backbones.** Fourth priority, exploratory only.

_What it is:_ concatenate or weight-combine cached embeddings across two or three top backbones (e.g., DINOv2-Large || RETFound-Green || DINOv3-Large → 3072-dim combined feature), then train a new MT head on the combined representation under Decision 027 recipe. No re-extraction required; existing caches are simply concatenated at head-input time.

_What it tests:_ whether different pretraining paradigms (generic-SSL, retinal-specific-SSL, next-generation-generic-SSL) produce _complementary_ features that combine usefully, or whether they are largely redundant. A real finding either way.

_Real risk to flag in the paper:_ combined feature dimension grows (768 + 1024 + 1024 = 2816 or 1024×3 = 3072) without sample size growing, so overfitting risk on sparse-positive tasks is non-trivial. Mitigate with dropout / weight decay in the head; report with appropriate caveats; do not lead the paper with this number. This is the one F-item where the overfitting concern is legitimate (unlike F1 where it was misplaced).

_Scope:_ one or two combinations max — start with DINOv2-Large || RETFound-Green (generic vs retinal-specific paradigm), add DINOv3-L if the first combination shows complementarity. MT head only.

_Reasoning capture:_ new finding F22 (feature ensembling — purpose, scope, overfitting-risk caveat) created.

---

**F5. Pipeline distillation for deployment.** Deferred at this planner, no activation in F-track without a specific trigger.

_What it would be (so the option is concretely scoped if reactivated):_ distill the full pipeline (frozen winning backbone + trained head) into a small end-to-end student model (e.g., a compact ConvNeXt-Tiny or MobileViT) trained on images directly, learning to mimic the full pipeline's task predictions. The goal is a deployable single-model artifact that runs at smartphone-feasible inference latencies without requiring the large backbone at inference time.

_Reactivation condition:_ only if the dashboard component (later original-planner stage) hits inference-latency constraints that make per-image large-backbone extraction infeasible at deployment. Until then, this is premature optimization and the locked-matrix-plus-stored-embeddings architecture is the deployment artifact.

_Reasoning capture:_ new finding F23 (pipeline distillation — deferred with dashboard-latency reactivation condition) created.

---

**F6. Permanently rejected at this planner round:** ensemble averaging / test-time augmentation / threshold tuning as primary contributions. Reason: these are leaderboard-polish techniques, not findings. They may appear briefly as final-result reporting (e.g., calibrated thresholds on the reliability split for the dashboard) but they are not paper-claim items and do not get a Category F slot. No reactivation condition. Reasoning capture: brief note in finding F11 (integration novelty) reinforcing that polish-techniques are not contributions.

---

### Execution order summary

Pre-lock, in order: A1 → B1 → (B2 and B3 in parallel or sequence) → re-run A1 on the new rows → C1, C2, C3 in parallel → E1.

Post-lock, in priority order: F1 → F2 → F3 (if activation gate passes) → F4 (exploratory, optional).

Deferred indefinitely with named reactivation conditions: D1–D6, F5.

Permanently rejected: D7, F6, B4 (FLAIR), B5 (DINOv2-Giant).

---

### Reasoning-capture audit (must be completed at E1)

The implementer's checklist at E1 closeout — every entry below must exist in `docs/paper_framing_and_findings.md` before Stage 8D-3.5 is closed:

- F4 ACTIVE, six rows now in scope (existing entry, extended scope confirmed).
- F5 ACTIVE, extended across six backbones (existing entry, extended).
- F6 ACTIVE with retraction locked (existing entry, status confirmed).
- F8 status: GATE SATISFIED by A1 (existing entry, status updated).
- F10 ACTIVE, matched-224 decision recorded, native-392 fork conditionally-deferred (existing entry, updated from DEFERRED-PENDING-SUBDECISION).
- F16 NEW: DINOv3-Large positioning, rationale, 224-native protocol decision.
When F16 is created (the DINOv3-Large entry — _after_  we confirm download + access):

- The PhysioNet DINOv3-S/B BRSET/mBRSET embedding release exists; record it as both a positioning reference (we evaluate L, they released S and B) and as a potential cross-comparator (their embeddings could be re-used in a follow-up scaling-within-DINOv3 analysis without re-extraction).
- DINOv3-Large is 300M params (same as DINOv2-Large); the "lower compute than DINOv2-Giant" claim is true, the "lower compute than DINOv2-Large" claim is not.
- Patch-16 vs DINOv2's patch-14 is a preprocessing-namespace note, not a methodological fork — both run natively at 224.
- The download path is the Meta direct release, not the HuggingFace gated mirror; this avoids the access-gate issue the B1 exploration flagged.
- The PhysioNet release of DINOv3-S/B BRSET embeddings does NOT undermine our novelty because (a) different scale (L not S/B), (b) different protocol (we run frozen extraction with MultiTaskHead + LinearProbeHead, not just CSV embeddings), (c) we evaluate across the full BRSET task panel and link to mBRSET / external validation / fairness / continual learning.


- F17 NEW: FLAIR rejection with full external-validation-dataset contamination list, conditional zero-shot reactivation noted.
- F18 NEW: DINOv2-Giant rejection with internal-scaling-flat and Zhou-et-al-DINOv3-L-superior evidence.
- F19 NEW: per-task single-task heads (F1) — purpose, scope, Doc-56 prediction framing.
- F20 NEW: GradNorm-vs-Kendall (F2) — purpose, scope, dr_grade-not-fix caveat.
- F21 NEW: CORAL ordinal dr_grade (F3) — purpose, activation gate.
- F22 NEW: feature ensembling (F4) — purpose, scope, overfitting caveat.
- F23 NEW: pipeline distillation (F5) — deferred, reactivation condition.

Stage 8D-3.5 closeout is not complete until this checklist is satisfied. The planner's reasoning does not survive into the manuscript unless it is captured in the framing-and-findings doc at the moment the action is taken.