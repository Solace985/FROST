# Paper Framing and Findings Record

## Purpose

This document records manuscript-oriented framing decisions, result-backed research inferences,
claims-to-make, and claims-to-avoid for use during manuscript writing. It is distinct from
`docs/decisions.md`, which records protocol and architecture decisions.

**Relationship to decisions.md:**
`docs/decisions.md` = protocol/architecture (training recipes, guardrail locks, infrastructure).
This document = manuscript framing and findings (what the results say, how to present them,
what claims are supported or unsupported by the current evidence).

**Discipline:**
Every entry has: ID, date added, the framing/finding, supporting evidence with exact file path
or decision number, rationale, and status label (ACTIVE / SUPERSEDED / DEFERRED / UNVERIFIED /
EXTERNAL-LIT).

Entries are numbered F1–F15. Do not add entries beyond F15 without explicit user instruction.

---

> **STANDING WARNING — read before drafting any section of the manuscript:**
> Nothing in this document is a final or clinical claim. All numbers in this document
> are preliminary, internal, non-paper-final results (`final_test_result=false` in all
> evaluation summary files). **All cross-backbone orderings are PROVISIONAL pending bootstrap
> confidence intervals (entry F8).** Do not assert any ordering in a manuscript draft until
> F8 is satisfied.

---

## Entries

---

### F1 — Stage 8D-3 backbone × head matrix: complete verified result table

**Date:** 2026-05-19

**Finding:**
The Stage 8D-3 matrix covers four backbones × two heads (MultiTaskHead and LinearProbeHead)
on the BRSET test split (n=1,623). All runs use Decision 027 locked unweighted recipe,
frozen backbone, patient-level 60/15/15/10 split, seed=42.
`final_test_result=false` in all eight evaluation_summary.json files.

Binary macro-AUROC = mean AUROC over 6 binary tasks: macular_edema, hypertensive_retinopathy,
amd, drusen, other_ocular, diabetes. dr_grade is excluded (ordinal, not binary).

**Source-verified values (all individual AUROCs read from overall_metrics.json; macro computed
by summing and dividing by 6; arithmetic shown):**

#### ResNet-50 MT
Source: `outputs/evaluation/20260512_192437/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9559176672384220 |
| hypertensive_retinopathy | 0.6905628792634443 |
| amd | 0.8747373800465619 |
| drusen | 0.7292267962934833 |
| other_ocular | 0.8119589869589869 |
| diabetes | 0.8468990917319518 |
Sum = 4.9093028015328502; **binary macro-AUROC = 0.8182171335888083**

#### ResNet-50 LP
Source: `outputs/evaluation/20260515_005309/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9045073375262054 |
| hypertensive_retinopathy | 0.7170119271814187 |
| amd | 0.7438816648685480 |
| drusen | 0.6627649219932374 |
| other_ocular | 0.7850737100737101 |
| diabetes | 0.8306282967633198 |
Sum = 4.6438678584064394; **binary macro-AUROC = 0.7739779764010732**

#### ConvNeXt-Base MT
Source: `outputs/evaluation/20260515_152007/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9738136077758719 |
| hypertensive_retinopathy | 0.6276940782590500 |
| amd | 0.9199364033842485 |
| drusen | 0.8343010335263107 |
| other_ocular | 0.8312275562275563 |
| diabetes | 0.8873810044008863 |
Sum = 5.0743536835739237; **binary macro-AUROC = 0.8457256139289873**

#### ConvNeXt-Base LP
Source: `outputs/evaluation/20260515_152223/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9165618448637317 |
| hypertensive_retinopathy | 0.6696798493408663 |
| amd | 0.8284027028561695 |
| drusen | 0.7719406327183935 |
| other_ocular | 0.7800037800037800 |
| diabetes | 0.8322887730578358 |
Sum = 4.7988775828407768; **binary macro-AUROC = 0.7998129304734628**

#### DINOv2-Base MT
Source: `outputs/evaluation/20260515_235231/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9870211549456833 |
| hypertensive_retinopathy | 0.7539652646997279 |
| amd | 0.9262108909204475 |
| drusen | 0.8274314873889537 |
| other_ocular | 0.8725193725193726 |
| diabetes | 0.8798464371547179 |
Sum = 5.2469946076289029; **binary macro-AUROC = 0.8744991012714838**

#### DINOv2-Base LP
Source: `outputs/evaluation/20260515_235605/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9776634267200306 |
| hypertensive_retinopathy | 0.7921322452395899 |
| amd | 0.8632956674805520 |
| drusen | 0.7750490882801379 |
| other_ocular | 0.8522254772254771 |
| diabetes | 0.8434095945566340 |
Sum = 5.1037754995024215; **binary macro-AUROC = 0.8506292499170702**

#### DINOv2-Large MT
Source: `outputs/evaluation/20260518_233207/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9837430912902612 |
| hypertensive_retinopathy | 0.7865243774848294 |
| amd | 0.9047186417579922 |
| drusen | 0.8477081821301782 |
| other_ocular | 0.8896191646191647 |
| diabetes | 0.8694903086862886 |
Sum = 5.2818037659687143; **binary macro-AUROC = 0.8803006276614524**

#### DINOv2-Large LP
Source: `outputs/evaluation/20260518_233445/overall_metrics.json`
| Task | AUROC |
|------|-------|
| macular_edema | 0.9741185439298647 |
| hypertensive_retinopathy | 0.7978656622724419 |
| amd | 0.7961785247856453 |
| drusen | 0.8075626895313968 |
| other_ocular | 0.8544415044415046 |
| diabetes | 0.8506819813352477 |
Sum = 5.0808489063961010; **binary macro-AUROC = 0.8468081510493502**

**Summary table (rounded to 4 d.p.):**
| Backbone | Params (n_params) | embed_dim | MT AUROC | LP AUROC |
|----------|-------------------|-----------|----------|----------|
| ResNet-50 | — | — | 0.8182 | 0.7740 |
| ConvNeXt-Base | — | — | 0.8457 | 0.7998 |
| DINOv2-Base | 175 | 768 | 0.8745 | 0.8506 |
| DINOv2-Large | 343 | 1024 | 0.8803 | 0.8468 |

Note: n_params for DINOv2 backbones sourced from backbone verification artifacts
(`outputs/backbone_verification/20260515_114711/dinov2_base_verification.json`,
`outputs/backbone_verification/20260518_050150/dinov2_large_verification.json`);
unit not explicitly stated in the artifact — confirm unit (likely millions) against
verification script before manuscript use. ResNet-50 and ConvNeXt-Base param counts
not verified here; mark [UNVERIFIED — not in eval artifacts] for those two.

RETFound: TBD (deferred — Decision 028, item 6).

**Status:** ACTIVE — all numbers source-verified from priority-1 artifacts.
Orderings provisional pending F8.

---

### F2 — Primary cross-backbone finding: pretraining paradigm matters more than backbone scale

**Date:** 2026-05-19

**Finding:**
The CNN-to-self-supervised-ViT jump (ConvNeXt-Base MT → DINOv2-Base MT) is nearly 5× larger
than the DINOv2-Base → DINOv2-Large scale-up, under identical frozen-extraction protocol.

**Computed deltas (from F1 source-verified values):**

ConvNeXt-Base MT → DINOv2-Base MT:
0.8744991012714838 − 0.8457256139289873 = **+0.0287734873424965 ≈ +0.0288**

DINOv2-Base MT → DINOv2-Large MT:
0.8803006276614524 − 0.8744991012714838 = **+0.0058015263899686 ≈ +0.0058**

Ratio: 0.0288 / 0.0058 ≈ 4.97×.

**Claim-to-make (provisional):**
Under frozen linear-evaluation protocol, switching pretraining paradigm (supervised CNN →
self-supervised ViT) yields a larger accuracy gain than scaling within the same paradigm
(DINOv2-Base → DINOv2-Large).

**Claim-to-avoid:**
Do not assert this as a general principle without F8 bootstrap CIs. The Base→Large delta
(+0.0058) is smaller than any plausible CI width given the sparse task positives in the macro.

**Supporting evidence:** F1 verified values. Decision 027 (locked recipe).

**Status:** ACTIVE — arithmetic verified. Ordering provisional pending F8.

---

### F3 — DINOv2 Base ≈ Large under frozen extraction; LP direction reverses

**Date:** 2026-05-19

**Finding:**
The MT delta Base→Large (+0.0058) is a macro over six binary tasks, three of which have
22–33 positive examples (amd n_pos=22, macular_edema n_pos=33, hypertensive_retinopathy
n_pos=30 — see F7). No confidence intervals have been computed. The LP direction reverses:
DINOv2-Large LP is *below* DINOv2-Base LP.

**Source-verified LP values (from F1):**
DINOv2-Base LP: 0.8506292499170702 (source: `outputs/evaluation/20260515_235605/overall_metrics.json`)
DINOv2-Large LP: 0.8468081510493502 (source: `outputs/evaluation/20260518_233445/overall_metrics.json`)
LP delta: 0.8468082 − 0.8506292 = **−0.0038210 (Large LP < Base LP)**

**Claim-to-make:**
Scaling from 175M to 343M parameters (as reported in backbone verification artifacts) bought
no measurable improvement under frozen linear evaluation; the LP direction reversal suggests
DINOv2-Base is the better linear-separability operating point for this domain.
Frame as: "The marginal MT gain (+0.0058) is within the expected noise of a 6-task macro with
sparse positives; LP reversal (−0.0038) corroborates that scale does not help beyond DINOv2-Base
under frozen extraction."

**Claim-to-avoid:**
Do not call DINOv2-Large "the matrix leader" in the manuscript. The +0.0058 point estimate
does not constitute a significant finding without F8 CIs. The constructive recommendation
(F15) is to lead with DINOv2-Base as the operating point; this entry is its defensive complement.

**Status:** ACTIVE — LP reversal source-verified. MT delta provisional pending F8.

---

### F4 — dr_grade: reframe primary DR endpoint as referable-DR binary

**Date:** 2026-05-19

**Finding:**
The BRSET test split dr_grade class distribution (same across all eight runs; n=1,623 total)
is heavily imbalanced, making 5-class ordinal prediction a misleading primary endpoint.
The appropriate manuscript primary for DR is a referable-DR binary (grades 0–1 vs 2–4).

**Source-verified dr_grade class distribution:**
Source: any overall_metrics.json `per_class_support` field — consistent across all 8 artifacts.
Verified from `outputs/evaluation/20260518_233207/overall_metrics.json` (DINOv2-Large MT):
- Class 0 (no DR): 1,517 / 1,623 = 93.47%
- Class 1 (mild DR): 33 / 1,623 = 2.03%
- Class 2 (moderate DR): 11 / 1,623 = 0.68%
- Class 3 (severe DR): 22 / 1,623 = 1.36%
- Class 4 (proliferative DR): 40 / 1,623 = 2.47%

**Claim-to-make:**
Report accuracy, macro_f1, and balanced_accuracy together for the 5-class task (do not omit
any of the three). Introduce referable-DR binary (grades ≥ 2) as the clinically meaningful
primary endpoint alongside the 5-class results; state that the 5-class protocol is retained
as a documented limitation.

**Claim-to-avoid:**
Do not headline 5-class dr_grade accuracy alone (misleading: always exceeds 93% by predicting
all-no-DR). Do not drop any of the three ordinal metrics (accuracy, macro_f1, balanced_accuracy)
from reporting.

**Status:** ACTIVE — class distribution source-verified from priority-1 artifacts.

---

### F5 — dr_grade balanced_accuracy shows monotone backbone ordering

**Date:** 2026-05-19

**Finding:**
Despite all four backbones failing to predict dr_grade classes 1 and 2 reliably
(see Decision 027 and Decision 028 item 3), the balanced_accuracy metric still increases
monotonically with backbone quality, suggesting that embeddings encode latent ordinal structure
that the unweighted argmax head cannot fully surface.

**Source-verified balanced_accuracy values (MultiTaskHead only):**

ResNet-50 MT: **0.38439983220471025**
Source: `outputs/evaluation/20260512_192437/overall_metrics.json`

ConvNeXt-Base MT: **0.40030892311380110**
Source: `outputs/evaluation/20260515_152007/overall_metrics.json`

DINOv2-Base MT: **0.43627704200874930**
Source: `outputs/evaluation/20260515_235231/overall_metrics.json`

DINOv2-Large MT: **0.47368160843770600**
Source: `outputs/evaluation/20260518_233207/overall_metrics.json`

Ordering: 0.3844 < 0.4003 < 0.4363 < 0.4737 — strictly monotone increasing.

**Interpretation:** Embeddings from stronger backbones carry more ordinal DR-grade signal,
even though the unweighted cross-entropy head trained on 93.5% class-0 data cannot translate
that signal into correct class 1/2 predictions. The signal is latent in the representation,
not absent.

**Note:** LP balanced_accuracy values not tabulated here; these four are MT-only.
LP balanced_accuracy comparison is a future entry if needed.

**Status:** ACTIVE — all four values source-verified. Ordering provisional pending F8.

---

### F6 — MT−LP separability gap is NOT monotone; corrected framing locked

**Date:** 2026-05-19

**Finding:**
The MT−LP macro-AUROC gap computed from F1 source-verified values is not monotone across
backbone strength. DINOv2-Base shows the *smallest* gap (features are most linearly separable),
and DINOv2-Large shows a wider gap than Base — suggesting the Large model encodes more
entangled representations for out-of-domain medical tasks.

**Computed MT−LP deltas (showing arithmetic, all values from F1):**

ResNet-50:
0.8182171335888083 − 0.7739779764010732 = **+0.0442391571877351**

ConvNeXt-Base:
0.8457256139289873 − 0.7998129304734628 = **+0.0459126834555245**

DINOv2-Base:
0.8744991012714838 − 0.8506292499170702 = **+0.0238698513544136**

DINOv2-Large:
0.8803006276614524 − 0.8468081510493502 = **+0.0334924766121022**

Gap sequence across backbone quality: 0.0442 → 0.0459 → 0.0239 → 0.0335.
This is NOT monotone. The gap does not narrow with increasing backbone quality.
DINOv2-Base has the narrowest gap; DINOv2-Large widens the gap relative to Base.

**Locked corrected claim (verbatim):**
"The MT−LP gap is not monotone in backbone strength. DINOv2-Base is a linear-separability
sweet spot; scaling to DINOv2-Large reduces linear accessibility for fine-grained
out-of-domain medical tasks."

**Retraction notice:**
The earlier framing that appeared in conversational analysis — "separability improves
monotonically with backbone quality" and "gap narrows monotonically" — was prior
conversational framing, never committed to a project document, and is **RETRACTED here
before it can enter a draft**. This entry is the authoritative project record on MT−LP
separability. Do not search for or edit any prior document to apply this retraction.

**Status:** ACTIVE — all arithmetic source-verified from F1 priority-1 artifact values.
Ordering provisional pending F8.

---

### F7 — Sparse-positive and label-validity caveats for binary tasks

**Date:** 2026-05-19

**Finding:**
Several binary tasks have very few positive examples in the test set (n=1,623), implying
wide implicit confidence intervals on per-task AUROC. Two tasks have additional label
validity concerns.

**Source-verified positive counts (test set):**
Source: `positives` field, `outputs/evaluation/20260518_233207/overall_metrics.json`
(consistent across all 8 evaluation artifacts — same test split):

| Task | n_positives | n_negatives | Caveat |
|------|-------------|-------------|--------|
| amd | 22 | 1,601 | Sparse — AUROC unreliable; wide implicit CI |
| macular_edema | 33 | 1,590 | Sparse |
| hypertensive_retinopathy | 30 | 1,593 | Sparse; also retinal finding ≠ systemic hypertension (Decision 026 item 3) |
| drusen | 261 | 1,362 | Adequate n |
| other_ocular | 143 | 1,480 | Adequate n; however, heterogeneous composite category |
| diabetes | 230 | 1,393 | Adequate n; however, clinical proxy label — PROXY quality (Decision 026 item 4) |

**Specific caveats:**
- **amd (n_pos=22):** Single-digit effective positive count after split; AUROC metric has
  very wide effective CI. Treat all amd AUROC comparisons across backbones with extreme
  caution (see F3: amd AUROC varies substantially across backbones, e.g. DINOv2-Base LP
  amd=0.8633 vs DINOv2-Large LP amd=0.7962 — a 0.067 drop consistent with pure noise at n=22).
- **diabetes:** Label source is clinical/medical record, not fundus-derived. Decision 026
  explicitly marks diabetes as PROXY quality. Do not overclaim as "retinal-derived systemic
  diabetes detection."
- **other_ocular:** Heterogeneous composite; AUROC represents a mix of conditions that may
  not be clinically coherent as a single endpoint.
- **hypertensive_retinopathy:** Direct ophthalmologist-graded retinal finding; must NOT be
  equated with systemic hypertension diagnosis (Decision 026 item 3).

**Status:** ACTIVE — positive counts source-verified. Label-quality notes from Decision 026.

---

### F8 — Statistical rigor requirement: bootstrap CIs required before any ordering claim

**Date:** 2026-05-19

**Finding:**
All backbone orderings, head comparisons, and task-level differences in this document are
point estimates computed on n=1,623 test samples. No confidence intervals have been computed.
Multiple entries (F2, F3, F5, F6) carry explicit provisional flags pending this requirement.

**What is required before manuscript submission:**
1. Bootstrap CIs (≥1,000 iterations, stratified by task) on each backbone's binary macro-AUROC.
2. Paired bootstrap test or DeLong test for each pairwise AUROC comparison:
   - ConvNeXt-Base MT vs DINOv2-Base MT (+0.0288 delta)
   - DINOv2-Base MT vs DINOv2-Large MT (+0.0058 delta)
   - All MT vs LP pairs for each backbone
   - Any per-task comparison cited in text
3. Multiple-comparison correction (e.g. Bonferroni or Holm) if testing more than 2 orderings
   in a single family.
4. Pre-registration of the exact comparison set before running bootstrap (Decision 013 obligation).

**Status:** GATE-SATISFIED — 2026-05-24.
Bootstrap CIs computed (Stage 8D-3.5 A1, 2000 resamples, patient-level, seed=42).
Report: `outputs/analysis/A1_bootstrap_ci/20260523T204156Z/a1_bootstrap_ci_report.md`
All 4 execution gates passed (M, P, C, D). 102 supported orderings, 106 not-supported.
DeLong not implemented; paired percentile bootstrap is the uniform A1 delta method.
F2/F3/F5/F6 ordering claims remain provisional until manuscript-stage review of §11
(not-supported orderings); softening recommendations are in §11 of the A1 report.

---

### F9 — Frozen-extraction methodological justification

**Date:** 2026-05-19

**Finding:**
Frozen backbone evaluation is the correct design choice for the matrix comparison runs and is
justifiable in the manuscript on three grounds.

**Grounds:**
1. **Hardware feasibility:** Full fine-tuning of a 307–343M parameter ViT on BRSET requires
   GPU with substantial VRAM; frozen extraction with caching is CPU-feasible and reproducible
   on commodity hardware.
2. **Clean comparability:** Frozen extraction ensures that any observed performance difference
   between backbones is attributable to the pretrained representations, not to downstream
   fine-tuning variance. This is the standard linear-evaluation protocol.
3. **Reproducibility:** Cached embeddings enable exact reproduction of downstream experiments
   without re-running the backbone forward pass.

**Supporting evidence:**
Decision 009 (`docs/decisions.md`, lines 269–291) — "Foundation Backbones Stay Frozen by
Default." Status: locked. Rationale quoted: "Frozen backbones reduce compute, improve
reproducibility, allow caching, and avoid unnecessary fine-tuning risk."

**Claim-to-avoid until F12 is satisfied:**
Any defense of the frozen design that lacks a concrete quantification of the accuracy cost.
The manuscript must acknowledge that freezing imposes an accuracy ceiling; F12 is the
experiment that provides the specific number.

**Status:** ACTIVE — Decision 009 verified at `docs/decisions.md`.

---

### F10 — RETFound-Green positioning

**Date:** 2026-05-19

**Finding:**
RETFound (Zhou et al., Nature 2023; "RETFound-Green" CFP variant) is the most important
missing backbone in the Stage 8D-3 matrix. It is the domain-specific baseline the manuscript
needs to answer whether retinal-specific pretraining outperforms general self-supervised
pretraining (DINOv2) under frozen extraction.

[EXTERNAL-LIT — not artifact-verified, confirm against paper before manuscript use]
RETFound public weights are openly released with no access gate.
RETFound has been benchmarked on BRSET in its own publication.
Architecture and published performance specifics must be confirmed against the paper before
manuscript use — do not assert them as settled here.

**Unresolved methodological fork (DEFERRED-PENDING-SUBDECISION):**
Two valid evaluation modes for RETFound exist:
- **Matrix-matched mode:** resize to 224px, extract CLS token — maximally comparable to
  DINOv2/ConvNeXt/ResNet in the Stage 8D-3 matrix.
- **Native mode:** use RETFound's native 392px input, average-pool over patch tokens —
  matches the design described in the RETFound paper and gives the most favorable
  comparison against its published numbers.

These may give different results. The choice must be a locked decision before extraction.
Record as DEFERRED-PENDING-SUBDECISION until explicit user decision.

**Status:** EXTERNAL-LIT for all parameter/performance specifics. DEFERRED-PENDING-SUBDECISION
for evaluation mode fork.

---

### F11 — Integration components are the scientific contribution; bake-off AUROC is Table 1

**Date:** 2026-05-19

**Finding:**
The backbone × head AUROC matrix (F1) is Table 1 of the manuscript. It is a necessary
prerequisite for the actual contribution, not the contribution itself.

**The actual research contribution is the integration:**
- Fairness stratification: per-subgroup AUROCs stratified by sex, age band, acquisition
  device, source population — not available in the published BRSET benchmarks
- Device-invariance natural experiment: mBRSET smartphone versus BRSET clinical-camera
  transfer evaluation
- LoRA continual-learning protocol: safe update with subgroup-balanced replay and
  out-of-distribution gating
- Subgroup-conditional reliability dashboard: per-subgroup reliability annotations at
  inference time
- Cross-population external validation: IDRiD, Messidor-2, APTOS 2019, EyePACS

**Supporting evidence:** `docs/project_specification.md` §1 (Executive Summary) —
"The research contribution is the integration. No single component is unprecedented;
their combination on reproducible public data, with the fairness and continual-learning
scaffolding as first-class outputs rather than discussion-section asides, is genuinely
under-served in the published literature."

**Manuscript structuring implication:**
Do not allow AUROC-improvement ablations (class weighting, referable-DR binary, fine-tuning)
to crowd out the integration components. The integration components must appear in the Methods
and Results as primary contributions, not as future work. Table 1 should be positioned as
"backbone selection" scaffolding for the integration experiments, not as the headline.

**Status:** ACTIVE.

---

### F12 — Frozen-vs-fine-tuned reference point: one controlled datapoint required

**Date:** 2026-05-19

**Finding:**
The frozen design (F9) requires a concrete quantification of the accuracy cost before
the paper can defend it without qualification.

**Planned experiment (DEFERRED-PENDING-EXPERIMENT):**
One controlled fine-tuning datapoint: one backbone (DINOv2-Base, recommended operating
point per F15), one dense endpoint (referable-DR binary, grades ≥ 2), reported as:
"Frozen extraction costs X AUROC points relative to fine-tuned, at a cost of Y GPU-hours
vs Z GPU-hours for extraction+head training."

**Claim-to-avoid until this experiment is complete:**
Any defense of the frozen design that does not include this number. In particular, do not
write "frozen extraction is competitive with fine-tuning" without the evidence.

**Status:** DEFERRED-PENDING-EXPERIMENT.

---

### F13 — Comparability-to-published-work caveat

**Date:** 2026-05-19

**Finding:**
Our evaluation protocol differs from the two most closely related BRSET papers in multiple
protocol dimensions. Direct AUROC comparisons without explicit protocol disclosure would
be misleading.

**Protocol differences (our work vs published work):**

[EXTERNAL-LIT — not artifact-verified, confirm against papers before manuscript use]

vs. Nakayama et al. (PLOS Digital Health 2024 — BRSET dataset paper):
- Their protocol: image-level split, fine-tuned ConvNeXt V2, single-task
- Our protocol: patient-level split, frozen backbone, multi-task Kendall-weighted head

vs. Aghabeigi Alooghareh et al. (Bioengineering 2025 12(8):840 — see F14 for author correction):
- Their protocol: patient-level split (5-fold cross-validation), fine-tuned, single-task per
  model, inverse-frequency class weighting
- Our protocol: single patient-level train/val/reliability/test split, frozen backbone,
  multi-task, no class weighting (Decision 027)

**Claim-to-avoid:**
Any "we match/beat X" statement without explicitly stating the protocol differences in the
same sentence. In particular: (a) our AUROC numbers are not directly comparable to Nakayama
due to image-level vs patient-level split and fine-tuned vs frozen; (b) our numbers are not
directly comparable to Aghabeigi due to 5-fold vs single split, weighted vs unweighted, and
fine-tuned vs frozen.

**Status:** EXTERNAL-LIT for the referenced papers' protocol details.

---

### F14 — Citation and number corrections: Aghabeigi et al. / not "Khan et al."; HR metric correction

**Date:** 2026-05-19

**Purpose:** Prevent incorrect citation and a specific misquoted number from entering any
manuscript draft.

**Correction 1 — Author attribution:**
The BRSET benchmark paper published in *Bioengineering* 2025, 12(8):840 is authored by
**Aghabeigi Alooghareh, Sheikhey, Sahafi, Pirnejad, and Naemi** — NOT "Khan et al."
The incorrect attribution "Khan et al." appears in both `docs/project_specification.md`
§4.3 and `docs/decisions.md` Decision 027 reference list. Both instances must be corrected
to "Aghabeigi Alooghareh et al." before submission.

[EXTERNAL-LIT — author attribution not artifact-verified; confirm against paper title page
before manuscript use]

**Correction 2 — HR metric identity:**
The value "0.79" cited in `docs/decisions.md` Decision 027 as "Khan Hypertension AUC ~0.79"
refers to a hypertension weighted-F1 score from the Aghabeigi paper, **not an AUROC**.

[EXTERNAL-LIT — confirm metric type and value against the paper before manuscript use]
Published best hypertensive retinopathy AUROC in Aghabeigi et al. is approximately:
CSWin-B ≈ 0.82, Swin-L ≈ 0.81.

Do not cite "0.79" as an HR AUROC in any manuscript section. Do not use "Khan" as the
author attribution. Both errors must be corrected before any comparison statement is written.

**Status:** ACTIVE — corrections locked. All values tagged [EXTERNAL-LIT]; confirm against
paper before use.

---

### F15 — Constructive efficiency claim: DINOv2-Base as recommended operating point

**Date:** 2026-05-19

**Finding:**
DINOv2-Base is the recommended operating point for the manuscript's primary backbone
recommendation. The manuscript should lead with this constructive framing; F3 is its
defensive complement.

**Source-verified supporting evidence:**

Accuracy (MT): DINOv2-Base = 0.8744991012714838 (source: F1,
`outputs/evaluation/20260515_235231/overall_metrics.json`)

Accuracy (LP): DINOv2-Base LP = 0.8506292499170702 > DINOv2-Large LP = 0.8468081510493502
(source: F1 — DINOv2-Base LP is *higher* than DINOv2-Large LP)

Parameter counts:
- DINOv2-Base: n_params = 175 (source: `outputs/backbone_verification/20260515_114711/dinov2_base_verification.json`)
- DINOv2-Large: n_params = 343 (source: `outputs/backbone_verification/20260518_050150/dinov2_large_verification.json`)
- Ratio: 175 / 343 = **0.5102 ≈ 51%** (DINOv2-Base is roughly half the parameter count of Large)
- Note: unit of n_params is not explicitly stated in the artifact (integer value, likely millions);
  confirm unit against the verification script before manuscript use.

Embedding dimension:
- DINOv2-Base: 768 (source: backbone verification JSON above)
- DINOv2-Large: 1024 (source: backbone verification JSON above)
- Smaller embedding store = 75% of Large's embedding size per sample; relevant for cache
  storage and inference latency.

MT−LP gap: DINOv2-Base gap = 0.0239 (smallest of all four — most linearly separable per F6)

**Constructive framing:**
DINOv2-Base delivers near-identical MT AUROC to DINOv2-Large (within 0.0058, likely within
CI), higher LP AUROC than Large, roughly half the parameters, and smaller embedding store.
For downstream integration experiments (fairness, cross-site, continual learning), DINOv2-Base
is the efficient primary backbone.

**Status:** ACTIVE — all values source-verified. Recommendation provisional pending F8 CIs.

---

## Unverified and Deferred Entry Summary

| Entry | Tag | Reason |
|-------|-----|--------|
| F8 | GATE-SATISFIED (2026-05-24) | Bootstrap CIs complete — see A1 report outputs/analysis/A1_bootstrap_ci/20260523T204156Z/ |
| F10 | EXTERNAL-LIT + DEFERRED-PENDING-SUBDECISION | RETFound weights/protocol; evaluation mode fork not decided |
| F12 | DEFERRED-PENDING-EXPERIMENT | Fine-tuning reference experiment not run |
| F13 (protocol comparisons) | EXTERNAL-LIT | Nakayama and Aghabeigi protocol details from papers, not our artifacts |
| F14 | EXTERNAL-LIT | Author names, HR AUROC values — must be confirmed against paper |
| ResNet-50 / ConvNeXt n_params | UNVERIFIED | Not in priority-1 or priority-2 sources; backbone verification JSON not checked for these two |
| F15 n_params unit | Needs confirmation | Unit ("millions") inferred but not explicit in backbone verification JSON |
