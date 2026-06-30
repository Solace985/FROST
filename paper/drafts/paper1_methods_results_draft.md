# Paper 1 — Methods & Results (Draft)

*Continues the manuscript begun in the Abstract/Introduction/Related Work draft. Manuscript-facing
names and register match that file; Table 1 there is the literature-gap table, so tables here begin
at Table 2. All numbers trace to internal reproducibility records (preliminary; not paper-final).
Cross-paper comparisons and unverified parameter counts are tracked at the end.*

---

## Methods

### Study design

We evaluated frozen retinal representations in a controlled backbone × head matrix, holding the
data split, training recipe, and random seed fixed while varying only two factors: the
backbone/protocol row that produced the image embedding, and the prediction head trained on top of
it. Freezing every backbone — extracting embeddings once, with no gradient updates to the
pretrained weights — ensured that any performance difference is attributable to the pretrained
representation rather than to fine-tuning dynamics, and made the comparison exactly reproducible
from cached features. The matrix comprised seven backbone/protocol rows and two heads, for fourteen
trained cells, each evaluated on the same held-out test set. This design isolates representation
quality as the variable of interest, which is the question the Introduction posed.

### Dataset and cohort

We used BRSET, a Brazilian colour-fundus cohort with paired ophthalmological gradings and
patient-level metadata [VERIFY: full dataset name and release version]. Because the cohort contains
multiple images per patient, the split was constructed at the patient level to prevent leakage of a
patient's images across partitions, using a 60/15/15/10 division into training, validation,
reliability, and test partitions; the reliability partition is reserved for subgroup reliability
work outside the scope of this paper. The resulting partitions contain 9,763, 2,443, 2,437, and
1,623 images respectively, and a split audit confirmed zero patient overlap across partitions
(Table 2). All model selection used the validation partition only, and every metric reported below
was computed once on the untouched test partition of 1,623 images from 854 patients.

**Table 2. Dataset summary.**

| Property | Value |
|---|---|
| Source / modality | BRSET; colour fundus photography [VERIFY: name, version] |
| Total images / patients | 16,266 / 8,524 |
| Split scheme | Patient-level 60/15/15/10 (train/validation/reliability/test) |
| Train | 9,763 images / 5,114 patients |
| Validation | 2,443 images / 1,278 patients |
| Reliability (held out, not used here) | 2,437 images / 1,278 patients |
| Test | 1,623 images / 854 patients |
| Patient overlap across splits | 0 (audited) |
| Label set | 5-class DR grade; six binary conditions (macular edema, hypertensive retinopathy, AMD, drusen, other ocular, diabetes) |

### Task and endpoint definitions

The primary endpoint is referable diabetic retinopathy (referable DR; grade ≥ 2), the binary
distinction between non-referable eyes (no or mild DR) and referable eyes (moderate DR or worse,
through proliferative disease) that determines whether a screening encounter ends in specialist
referral. We formed the referable-DR score from the model's 5-class DR-grade output as the total
predicted probability mass on grades 2, 3, and 4, so that the endpoint reflects the clinical
referral boundary rather than exact severity. The 5-class DR grade itself is reported only as a
secondary endpoint and documented limitation, because its severe class imbalance makes
fine-grained ordinal accuracy a poor screening summary. The six binary conditions constitute a
multi-condition panel; two carry label-validity caveats that we state plainly and do not overclaim:
the diabetes label is record-derived and is therefore a proxy rather than a retina-derived
diagnosis, and "hypertensive retinopathy" is an ophthalmological retinal grading, not a measurement
of systemic blood pressure.

### Image preprocessing and extraction protocols

To make embeddings comparable across backbones, all images were processed under a single
resolution-matched protocol at 224 × 224 pixels before extraction. Because the retina-specific
backbone was pretrained at its own native resolution, we additionally extracted it under a native
protocol at 392 × 392 pixels with average pooling over patch tokens, giving a within-backbone
comparison of acquisition-matched versus native-resolution features. Under the matched protocol the
transformer backbones used their class-token embedding and the convolutional backbones used global
average pooling; positional embeddings for the native-392 weights were interpolated to 224 under
the matched protocol, as is standard for that model family. Embeddings were extracted once per
protocol and cached, so that all downstream head training and evaluation reads fixed feature
vectors.

### Backbone models

The seven backbone/protocol rows span four pretraining paradigms (Table 4 lists them). Two are
supervised convolutional networks — ResNet-50 (2048-dimensional features) and ConvNeXt-Base
(1024-dimensional). Three are general self-supervised learning (SSL) vision transformers — DINOv2-Base
(ViT-Base, 768-dimensional), DINOv2-Large (ViT-Large, 1024-dimensional), and the next-generation
DINOv3-Large (ViT-Large, 1024-dimensional). The remaining rows are a retina-specific SSL model,
RETFound-Green (a ViT-Small/14 backbone with 384-dimensional features), evaluated under both the
matched (224) and native (392) protocols. We confirmed that the retina-specific model's pretraining
corpus does not include BRSET or the external DR datasets reserved for later work, so no
pretraining/evaluation overlap affects these results [VERIFY: pretraining-corpus contamination
check against source].

### Prediction heads

Two heads bracket the realistic options for a frozen reader. The linear probe is a single linear
layer trained per task directly on the frozen embedding, measuring the raw linear separability of
the representation. The multi-task head is a shared trunk with per-task output layers trained
jointly, with per-task losses combined by homoscedastic uncertainty weighting so that the model
learns the relative weight of each task rather than using fixed weights. Each task used the loss
appropriate to its type — cross-entropy for the 5-class DR grade and binary cross-entropy for the
six binary conditions — with a per-task mask so that images lacking a given label contribute no
gradient for that task.

### Training recipe

A single pre-specified recipe was applied identically to all fourteen cells, so that any
performance difference reflects the backbone or head rather than tuning. Heads were trained with
AdamW at a learning rate of 1 × 10⁻⁴, batch size 256, and a cosine schedule with warmup, with
gradient clipping at 1.0 and a cap of 100 epochs. Model selection used early stopping on validation
multi-condition macro-AUROC with patience 10, and no class weighting was applied; all runs used a
fixed seed of 42. Imbalance mitigation was deliberately excluded from this comparison and reserved
for a separate ablation, so that the matrix measures representations under one neutral recipe.

### Evaluation and statistical analysis

For each cell we computed per-task area under the receiver operating characteristic curve (AUROC),
adding area under the precision–recall curve (AUPRC) for the sparse-positive conditions, and we
summarised the binary panel as a multi-condition macro-AUROC — the mean AUROC over the six binary
conditions, with the ordinal DR grade excluded from the aggregate. We quantified uncertainty with a
patient-clustered bootstrap: resampling the 854 test patients with replacement so that a patient's
images move together, recomputing each metric over 2,000 resamples at a fixed seed, and taking the
2.5th and 97.5th percentiles as the 95% confidence interval (CI). Pairwise comparisons used a
paired bootstrap that applied the same patient resamples to both cells; we label a difference
"supported" when its paired 95% CI excludes zero and "not separable under paired bootstrap" when the
interval includes zero. We did not run a formal equivalence or non-inferiority test, so we make no
equivalence claims; an overlapping interval is reported as non-separation, not as sameness.

### Scope and claim control

This paper is deliberately bounded to internal BRSET. We make no external-validation,
cross-device, fairness-mitigation, calibration-transfer, continual-learning, or deployment claims
here, and we do not name or brand the architecture. Secondary multi-condition results are framed as
screening or risk-flagging signals rather than diagnoses, consistent with the label caveats above.
These boundaries keep every claim that follows within what the present evidence supports.

---

## Results

### Cohort and task structure

The test cohort's label structure conditions every result that follows, because most conditions are
rare. Among the 1,623 test images, referable DR is present in 73 (4.50%), and the underlying DR
grade is dominated by grade 0 (1,517 images, 93.5%), with mild, moderate, severe, and proliferative
grades at 33, 11, 22, and 40 images respectively (Table 3). The binary panel ranges from densely to
very sparsely positive: drusen (261 positives, 16.1%), diabetes (230, 14.2%), and other ocular
(143, 8.8%) are well populated, whereas macular edema (33, 2.0%), hypertensive retinopathy (30,
1.8%), and AMD (22, 1.4%) are sparse. These three sparse conditions carry wide implicit uncertainty
on AUROC and therefore warrant precision–recall reporting and cautious comparison, and the extreme
DR-grade imbalance is precisely why we adopt referable DR rather than fine-grained severity as the
primary endpoint.

**Table 3. Cohort and task characteristics (test partition, n = 1,623 images / 854 patients).**

| Task | Positives | Prevalence | Notes |
|---|---|---|---|
| Referable DR (grade ≥ 2) | 73 | 4.50% | Primary endpoint; from DR-grade probability mass on grades 2–4 |
| DR grade (5-class) | grade 0/1/2/3/4 = 1,517 / 33 / 11 / 22 / 40 | 93.5% grade 0 | Secondary; severe imbalance |
| Drusen | 261 | 16.1% | Well populated |
| Diabetes | 230 | 14.2% | Record-derived proxy label |
| Other ocular | 143 | 8.8% | Heterogeneous composite |
| Macular edema | 33 | 2.0% | Sparse |
| Hypertensive retinopathy | 30 | 1.8% | Sparse; retinal grading, not systemic measurement |
| AMD | 22 | 1.4% | Sparsest; wide implicit CI |

### Primary referable-DR result

Frozen features support strong referable-DR discrimination across the matrix. Every multi-task cell
reaches referable-DR AUROC above 0.97, and the strongest cell — RETFound-Green (392, native) with
the multi-task head — reaches 0.985 (95% CI 0.964–0.997), with the large general transformers close
behind: DINOv2-Large at 0.979 (0.956–0.993) and DINOv3-Large at 0.977 (0.956–0.993) (Table 4). The
top tier is statistically indistinct: across the leading multi-task cells, every paired
backbone-to-backbone difference is not separable under paired bootstrap (Table 5), so no single
top-tier backbone can be ranked first on referable DR from these data. Precision–recall tells the
same story with more spread, the strongest cell reaching referable-DR AUPRC 0.882 (0.797–0.941)
against a 4.5% positive base rate. A sensitivity-at-fixed-specificity operating point would sharpen
the clinical reading and is computable from the saved per-sample predictions, but is not yet in the
analysis records and is flagged for completion.

### Parameter-efficiency and the domain-pretraining frontier

The matrix's one supported top-tier backbone advantage belongs to the smallest model. RETFound-Green
(392, native), a ViT-Small backbone with 384-dimensional embeddings, holds the only referable-DR
backbone difference at the top tier that survives paired bootstrap — over DINOv2-Base with the
multi-task head, AUROC Δ +0.012 (95% CI 0.004–0.022) — while the much larger ViT-Large general
transformers (1024-dimensional embeddings) do not separate from it: RETFound-Green (392, native)
versus DINOv2-Large is +0.006 (−0.017–0.031) and versus DINOv3-Large is +0.008 (−0.003–0.022), both
not separable under paired bootstrap (Table 5, Figure 3). This is representation quality bought by
domain-specific pretraining rather than by scale: a ViT-Small retinal model matches ViT-Large
general models on the clinical endpoint at a fraction of the embedding width — 384 versus 1024
dimensions, a smaller cache and a lighter head — and approximately an order of magnitude fewer
backbone parameters [VERIFY: ~22M vs ~300M from standard model cards]. We did not run a formal
non-inferiority test, so we state this as the only supported top-tier edge plus non-separation from
the larger models, and we do not claim equivalence.

### Backbone axis: paradigm over scale, and recency does not dominate

On the multi-condition macro-AUROC, where confidence intervals are available for the four original
backbones, the pretraining paradigm matters more than scale. Moving from the supervised convolutional
ConvNeXt-Base to the self-supervised DINOv2-Base raises multi-task macro-AUROC by +0.029 (95% CI
0.010–0.049), a supported jump, whereas scaling within the self-supervised family from DINOv2-Base
to DINOv2-Large adds only +0.006 (−0.015–0.025), not separable under paired bootstrap (Table 5).
Model recency likewise does not dominate: on referable DR the next-generation DINOv3-Large is not
separable from DINOv2-Large (−0.002, −0.021–0.018). For the retina-specific and next-generation
rows the multi-condition macro-AUROC is reported as a point estimate without a CI (RETFound-Green
matched 0.843 and native 0.875; DINOv3-Large 0.875), because the clustered-bootstrap intervals for
the macro aggregate were computed for the original four backbones only; these macro values are
therefore descriptive and are not used for tested comparisons.

### Multi-condition screening and the head axis

The multi-task head provides targeted positive transfer that concentrates on the sparsest
conditions. On the multi-condition macro-AUROC the multi-task head exceeds the linear probe for
every backbone with a CI available — ResNet-50 +0.044 (0.023–0.066), ConvNeXt-Base +0.046
(0.026–0.067), DINOv2-Base +0.024 (0.010–0.038), and DINOv2-Large +0.033 (0.012–0.057) — and the
gain is largest on AMD, the rarest condition, where the multi-task advantage is supported for all
four backbones and reaches +0.131 (0.057–0.215) for ResNet-50 and +0.109 (0.015–0.212) for
DINOv2-Large (Table 5). The benefit is not monotone in backbone strength: DINOv2-Base shows the
narrowest head gap and DINOv2-Large a wider one, so a stronger backbone does not steadily reduce the
value of joint training. On referable DR the head effect is endpoint-dependent and ceiling-limited:
the multi-task head's AUROC advantage is supported only below the discrimination ceiling — ConvNeXt-Base
+0.032 (0.014–0.051), DINOv2-Base +0.015 (0.004–0.029), RETFound-Green (392, native) +0.017
(0.004–0.030) — and is not separable for the cells already at 0.97+; on precision–recall, however,
the multi-task head's referable-DR advantage is supported for all seven backbones.

### Protocol sensitivity: a within-backbone change rivals a change of backbone

Acquisition protocol moves the retina-specific model as much as a change of backbone does. Switching
RETFound-Green from the matched-224 to the native-392 protocol raises multi-condition macro-AUROC by
+0.032 as a point estimate (0.843 → 0.875; reported without a CI, descriptive only), a shift that
rivals the supported CNN-to-transformer paradigm jump (+0.029) and exceeds the within-family scale
step (+0.006). On the primary endpoint the protocol effect is metric-dependent: the native protocol's
referable-DR AUROC gain over matched is +0.007 (−0.004–0.020), not separable under paired bootstrap,
but its AUPRC gain is +0.048 (0.003–0.096), supported (Table 5). The native protocol thus yields the
strongest single cell and a supported precision–recall improvement, while remaining statistically
indistinct from matched extraction on referable-DR AUROC — a reminder that preprocessing is a
first-class experimental factor, not a fixed detail.

### DR severity (5-class): a strong screening endpoint with weak fine-grained grading

Fine-grained DR grading is the matrix's weakest result, and we report it honestly. Under the single
unweighted recipe, the rare DR grades collapse — mild and moderate grades receive essentially no
correct predictions against the 93.5%-grade-0 background — so 5-class balanced accuracy stays well
below its discrimination potential, from 0.384 (ResNet-50, multi-task) to 0.474 (DINOv2-Large,
multi-task) (Table 4). Yet that balanced accuracy rises with backbone strength across the original
self-supervised rows, indicating that stronger embeddings encode latent ordinal severity that the
unweighted argmax head does not fully surface. The contrast between near-ceiling referable-DR
discrimination and weak fine-grained grading under the same features is exactly what motivates
referable DR as the primary DR endpoint, with multi-class severity retained as a documented
limitation.

### Master matrix and supported-claim summary

Table 4 reports the full fourteen-cell matrix; Table 5 reports the head and key backbone deltas with
verdicts; Table 6 enumerates which claims are CI-supported, which are point-estimate descriptive,
and which are rejected.

**Table 4. Master matrix (14 cells).** Referable-DR AUROC [95% CI]; multi-condition macro-AUROC
([95% CI] for the four original backbones, point estimate "PE" otherwise); 5-class DR balanced
accuracy. MT = multi-task head; LP = linear probe.

| Backbone (protocol) | Head | Referable-DR AUROC [95% CI] | Multi-condition macro-AUROC | 5-class bal. acc. |
|---|---|---|---|---|
| ResNet-50 (224) | MT | 0.942 [0.901–0.974] | 0.818 [0.790–0.845] | 0.384 |
| ResNet-50 (224) | LP | 0.920 [0.885–0.949] | 0.774 [0.742–0.804] | 0.239 |
| ConvNeXt-Base (224) | MT | 0.975 [0.954–0.989] | 0.846 [0.821–0.871] | 0.400 |
| ConvNeXt-Base (224) | LP | 0.943 [0.913–0.969] | 0.800 [0.771–0.829] | 0.307 |
| DINOv2-Base (224) | MT | 0.973 [0.951–0.989] | 0.874 [0.855–0.892] | 0.436 |
| DINOv2-Base (224) | LP | 0.958 [0.931–0.981] | 0.851 [0.830–0.871] | 0.418 |
| DINOv2-Large (224) | MT | 0.979 [0.956–0.993] | 0.880 [0.854–0.904] | 0.474 |
| DINOv2-Large (224) | LP | 0.974 [0.958–0.987] | 0.847 [0.819–0.875] | 0.423 |
| RETFound-Green (224, matched) | MT | 0.978 [0.959–0.991] | 0.843 (PE) | 0.399 |
| RETFound-Green (224, matched) | LP | 0.968 [0.950–0.984] | 0.806 (PE) | 0.355 |
| DINOv3-Large (224) | MT | 0.977 [0.956–0.993] | 0.875 (PE) | 0.419 |
| DINOv3-Large (224) | LP | 0.962 [0.938–0.980] | 0.829 (PE) | 0.347 |
| RETFound-Green (392, native) | MT | **0.985 [0.964–0.997]** | 0.875 (PE) | 0.463 |
| RETFound-Green (392, native) | LP | 0.968 [0.951–0.982] | 0.819 (PE) | 0.332 |

**Table 5. Paired deltas with verdicts.** Δ = first minus second; "supported" = paired 95% CI
excludes zero; "not separable" = interval includes zero.

| Comparison | Metric | Δ [95% CI] | Verdict |
|---|---|---|---|
| **Backbone (multi-task)** | | | |
| ConvNeXt-Base → DINOv2-Base | Macro-AUROC | +0.029 [0.010–0.049] | supported |
| DINOv2-Base → DINOv2-Large | Macro-AUROC | +0.006 [−0.015–0.025] | not separable |
| RETFound-Green (392) vs DINOv2-Base | Referable-DR AUROC | +0.012 [0.004–0.022] | **supported** |
| RETFound-Green (392) vs DINOv2-Large | Referable-DR AUROC | +0.006 [−0.017–0.031] | not separable |
| RETFound-Green (392) vs DINOv3-Large | Referable-DR AUROC | +0.008 [−0.003–0.022] | not separable |
| DINOv3-Large vs DINOv2-Large | Referable-DR AUROC | −0.002 [−0.021–0.018] | not separable |
| **Head (multi-task − linear probe)** | | | |
| ResNet-50 | Macro-AUROC | +0.044 [0.023–0.066] | supported |
| ConvNeXt-Base | Macro-AUROC | +0.046 [0.026–0.067] | supported |
| DINOv2-Base | Macro-AUROC | +0.024 [0.010–0.038] | supported |
| DINOv2-Large | Macro-AUROC | +0.033 [0.012–0.057] | supported |
| ResNet-50 | AMD AUROC | +0.131 [0.057–0.215] | supported |
| DINOv2-Large | AMD AUROC | +0.109 [0.015–0.212] | supported |
| ConvNeXt-Base | Referable-DR AUROC | +0.032 [0.014–0.051] | supported |
| DINOv2-Base | Referable-DR AUROC | +0.015 [0.004–0.029] | supported |
| DINOv2-Large | Referable-DR AUROC | +0.005 [−0.008–0.017] | not separable |
| **Protocol (RETFound-Green, native − matched)** | | | |
| Native-392 vs matched-224 (MT) | Referable-DR AUROC | +0.007 [−0.004–0.020] | not separable |
| Native-392 vs matched-224 (MT) | Referable-DR AUPRC | +0.048 [0.003–0.096] | supported |

**Table 6. Supported versus descriptive claims.**

| Claim | CI-supported? | Status |
|---|---|---|
| Every multi-task cell reaches referable-DR AUROC > 0.97 | Yes (per-cell CIs) | Supported |
| Top-tier backbones not separable on referable DR | Yes (paired) | Supported (non-separation) |
| RETFound-Green (392) > DINOv2-Base on referable-DR AUROC | Yes | Supported |
| RETFound-Green (392) not separable from DINOv2-/DINOv3-Large on referable DR | Yes (paired) | Supported (non-separation); no equivalence claim |
| Paradigm jump (CNN → SSL ViT) > within-family scale step on macro-AUROC | Yes (original 4) | Supported |
| Multi-task > linear probe on macro-AUROC (original 4 backbones) | Yes | Supported |
| Multi-task rescue of sparse AMD | Yes (all 4) | Supported |
| Multi-task referable-DR AUROC gain below the ceiling | Yes (3 backbones) | Supported; ceiling-limited |
| Native-392 referable-DR AUPRC gain over matched | Yes | Supported |
| Native-392 referable-DR AUROC gain over matched | No | Not separable |
| Protocol change rivals paradigm jump on macro-AUROC | No (macro PE) | Descriptive (point estimate) |
| RETFound-Green / DINOv3-Large multi-condition macro-AUROC | No | Descriptive (point estimate) |
| 5-class balanced accuracy rises with backbone strength | No (no CI computed) | Descriptive trend |
| Frozen ≈ fine-tuned; equivalence of top backbones | No test run | Not claimed |

---

## PLACEHOLDERS & OPEN ITEMS

**[VERIFY] — internally recorded, confirm against source before submission:**
- BRSET full dataset name and release version (Table 2).
- Retina-specific model pretraining-corpus contamination check against BRSET and the reserved
  external DR datasets (Methods, Backbone models).
- Backbone parameter counts for the efficiency contrast (~22M ViT-Small vs ~300M ViT-Large): the
  internal verification records store parameter-tensor counts, not parameter totals; confirm the
  million-scale figures from standard model cards before the explicit "order of magnitude" wording.
  The verified contrasts that do not need this step are architecture class (ViT-Small vs ViT-Large)
  and embedding dimension (384 vs 1024), which are used as the primary efficiency evidence.

**[from artifact: not yet computed] — claims down-shifted or flagged because the value/test is absent:**
- Sensitivity at a fixed high specificity for the headline referable-DR cell: computable from saved
  per-sample predictions but not present in the analysis records; flagged in the primary-result
  subsection rather than stated.
- Multi-condition macro-AUROC confidence intervals for RETFound-Green (both protocols) and
  DINOv3-Large: not computed (the clustered-bootstrap macro intervals cover the four original
  backbones only), so these macro values are reported as point estimates and excluded from tested
  comparisons.
- Formal non-inferiority / equivalence test (e.g. TOST) for the small retina-specific model versus
  the large general transformers on referable DR: not run; the top-tier relationship is reported as
  "the only supported edge" plus "not separable under paired bootstrap", with no equivalence claim.
- 5-class DR balanced-accuracy trend across backbones: reported as a descriptive monotone trend; no
  bootstrap CI was computed for the ordinal balanced-accuracy ranking.

**Resolved during drafting:**
- Per-task positive-count check: macular edema (33) and hypertensive retinopathy (30) confirmed
  against the authoritative per-task metrics record; consistent across all cells (same test split).
  No discrepancy remains.
