# Paper 1 — Front Sections (Draft)

*Controlled frozen-backbone × head evaluation on BRSET with a referable-DR primary endpoint.
Target venue: npj Digital Medicine. This file contains the Abstract, Introduction, and a
standalone Related Work section, plus a placeholders list. All numbers trace to internal
reproducibility records; cross-paper comparisons are marked for verification.*

---

## Abstract

**Background.** Diabetic retinopathy (DR) is a leading cause of preventable vision loss, and fundus
screening turns on one decision: whether an image is referable and warrants specialist review. How
frozen foundation-model features from different pretraining paradigms compare for this decision,
under a matched and statistically rigorous protocol, is unclear.

**Methods.** On BRSET with a patient-level 60/15/15/10 split (test: 1,623 images, 854 patients), we
extract frozen cached embeddings from seven backbone/protocol rows — two supervised (ResNet-50,
ConvNeXt-Base), three general self-supervised learning (SSL; DINOv2-Base, DINOv2-Large,
DINOv3-Large), and one retina-specific SSL (RETFound-Green) at matched and native resolution — and
train two heads each: a shared multi-task head (MT) and a linear probe (LP). The primary endpoint
is referable DR (grade ≥ 2), with a six-task multi-condition macro-AUROC secondary; we report
patient-clustered bootstrap 95% confidence intervals (CIs; 2,000 resamples) for AUROC and AUPRC.

**Results.** Every MT row reaches referable-DR AUROC above 0.97, the strongest 0.985 (95% CI
0.964–0.997). Across general backbones, scale (Base to Large) and recency (DINOv2 to DINOv3) are not
separable under paired bootstrap. The only surviving top-tier backbone advantage is held by the
smallest, retina-specifically pretrained model over a general SSL transformer (AUROC Δ +0.012, 95%
CI 0.004–0.022). Head and protocol effects are endpoint-dependent: MT exceeds LP on AUPRC for every
backbone but on AUROC only below the discrimination ceiling; native and matched resolution are not
separable on AUROC, though native shows a supported AUPRC gain.

**Conclusion.** Frozen features provide a strong, reproducible referable-DR screening baseline on
internal BRSET, and representation quality — obtained cheaply through domain pretraining — and
acquisition protocol matter as much as scale or recency. Rankings shift with endpoint, head,
preprocessing protocol, and statistical uncertainty.

**Keywords:** diabetic retinopathy screening; retinal foundation models; frozen feature
extraction; referable DR; bootstrap confidence intervals; multi-task learning; BRSET

---

## Introduction

Diabetic retinopathy is a leading cause of preventable blindness, and the clinical value of
fundus screening rests on one binary triage decision rather than on fine-grained severity
grading. A screening encounter sorts each patient into referable disease, which warrants
ophthalmologist review and possible treatment, or non-referable, which can be safely returned
to routine follow-up. This referable/non-referable boundary — conventionally moderate DR or
worse — is the operating point that determines whether a screening programme catches sight-
threatening disease before it progresses. The burden falls hardest where ophthalmologists are
scarce: in low- and middle-income settings, the number of people with diabetes far exceeds the
specialist capacity to examine them, so an accurate, low-cost triage that can be run by a
technician is the rate-limiting clinical need. A fundus photograph, acquired in seconds without
contrast or pupil dilation and increasingly capturable on portable cameras, is the natural
substrate for that triage, which motivates automated readers that make the referable decision
reliably and at scale.

The retina also offers more than a view of eye-specific disease, because it is the one site
where the living microvasculature is imaged directly. Vascular calibre, tortuosity, and
neural-layer changes visible in a fundus image carry screening signal for systemic conditions
whose pathology is partly vascular, and a single photograph can therefore flag multiple ocular
findings alongside risk proxies for metabolic and cardiovascular processes. We treat these
secondary signals strictly as screening or risk-flagging endpoints, not as diagnoses: a retinal
finding labelled "hypertensive retinopathy" is an ophthalmological grading, not a measurement of
systemic blood pressure, and a record-derived diabetes label is a proxy, not a retina-derived
diagnosis. Framed this way, a fundus reader is a multi-condition triage instrument, and the
question becomes which image representation best supports that panel of decisions.

Foundation models reshape how that representation is obtained, by converting an image into a
reusable embedding that a lightweight head can map to many tasks. Three pretraining paradigms now
compete for this role on fundus images: supervised convolutional networks trained on natural
images, general self-supervised vision transformers such as the DINOv2 and DINOv3 families, and
retina-specific self-supervised models such as RETFound-Green trained on large unlabelled fundus
corpora. Each embodies a different bet about where transferable retinal signal comes from —
label supervision, general visual statistics at scale, or domain-matched pretraining. How these
representations behave specifically when the backbone is frozen and only a small head is trained
is the open question, because freezing isolates the quality of the pretrained features from the
confound of downstream fine-tuning and makes the comparison reproducible on commodity hardware.

Existing comparisons rarely isolate that question, because they vary along many axes at once.
Reported retinal-model rankings differ in dataset, image preprocessing and input resolution,
endpoint definition, whether the split is image-level or patient-level, whether the backbone is
fine-tuned or frozen, whether any uncertainty is reported, which model families are included, and
which head architecture sits on top. When several of these differ between two studies, a headline
AUROC gap cannot be attributed to the representation alone, and image-level splits in particular
can inflate metrics through patient leakage when a cohort contains multiple images per patient.
What is missing is a controlled frozen comparison on a patient-level split, evaluated on a
clinically meaningful screening endpoint, with confidence intervals that say which differences are
real. We supply exactly that.

This paper reports a controlled frozen backbone × head evaluation on BRSET. We hold the data
split, training recipe, and seed fixed, and vary only the backbone/protocol row — ResNet-50,
ConvNeXt-Base, DINOv2-Base, DINOv2-Large, DINOv3-Large, and RETFound-Green at resolution-matched
and native resolution — and the prediction head, a shared multi-task head versus a per-task linear
probe. Our contributions are: (1) a fully patient-split, frozen-extraction matrix that attributes
performance differences to the pretrained representation rather than to fine-tuning variance; (2)
referable DR (grade ≥ 2) as the primary DR endpoint, reported with patient-clustered bootstrap
CIs for every cell; (3) the finding that top-tier frozen multi-task models all exceed referable-DR
AUROC 0.97 and are not separable under paired bootstrap across general-backbone scale and recency;
(4) the finding that the only supported top-tier backbone advantage is held by the smallest,
retina-specifically pretrained model over a general transformer, indicating representation quality
over scale; (5) an endpoint-dependent head and protocol analysis, in which the multi-task head
beats the linear probe on AUPRC for every backbone but only below the AUROC ceiling, and native
resolution separates from matched resolution on AUPRC but not AUROC; and (6) a reproducible
protocol built on cached frozen embeddings, a fixed seed, and a pre-specified comparison set. The
remainder of the paper details the cohort and matrix, the statistical procedure, and the per-cell
results that support these claims.

---

## Related Work

*Venue note (for the author, not for the manuscript): for an npj Digital Medicine submission this
content is normally folded into the Introduction. It is written here as a standalone section so it
can be kept or merged.*

**Retinal AI and DR screening.** Automated DR detection is among the most mature applications of
medical imaging AI, with multiple readers reporting high discrimination for referable disease on
standard datasets and several reaching regulatory deployment [CITE: landmark automated DR
screening / regulatory reader studies]. This maturity has shifted the frontier away from "can a
model detect referable DR at all" toward whether reported performance is obtained under
conditions that transfer to practice — patient-level evaluation, calibrated uncertainty, and
robustness across acquisition devices and populations [CITE: reviews on clinical readiness of DR
AI]. Taken together, the DR-screening literature establishes referable DR as the clinically
meaningful endpoint while leaving the methodological question of how to compare representations
fairly largely open.

**BRSET and public fundus datasets with systemic labels.** BRSET is a Brazilian multi-label
colour-fundus cohort [VERIFY: full dataset name and release version] that pairs images with both
ophthalmological gradings and patient-level metadata, which makes it well suited to multi-condition
screening and to subgroup-aware evaluation; its smartphone companion supports cross-device study
[CITE: BRSET dataset paper; CITE: smartphone-fundus companion dataset]. Its value for our purpose
is the combination of a DR severity grade with several binary ocular labels on the same images;
its limitations are a heavily imbalanced DR grade distribution and label sources that vary in
proximity to the retina, so that some targets are proxies rather than retina-derived diagnoses.
These properties make BRSET a strong internal substrate but also explain why a raw multi-class DR
metric is a poor headline, motivating the referable-DR recast we adopt.

**Retinal and general vision foundation models.** Three model families anchor current comparisons.
Retina-specific self-supervised models, beginning with the RETFound line, pretrain on large
unlabelled fundus corpora and report transfer to ocular and systemic endpoints [CITE: RETFound and
RETFound-Green]. General self-supervised transformers, notably the DINOv2 and DINOv3 families,
provide strong label-free representations trained on natural images and are increasingly applied
to medical tasks [CITE: DINOv2; CITE: DINOv3]. Supervised convolutional baselines such as
ResNet-50 and ConvNeXt remain standard reference points and underpin the original BRSET benchmarks
[CITE: ConvNeXt; CITE: ResNet]. Prior evaluations report that domain-specific and general models
each lead on some tasks and that scaling does not uniformly help in medical imaging [VERIFY:
generalist-vs-specialist ocular benchmark], which indicates that representation quality is task-
and domain-dependent rather than a monotone function of size or recency — precisely the
hypothesis our frozen matrix is designed to test.

**Frozen embeddings and lightweight heads.** Freezing the backbone and training only a small head
is the standard linear-evaluation protocol for representation quality, and it carries three
practical advantages for a controlled comparison: any performance difference is attributable to
the pretrained features rather than to fine-tuning dynamics, embeddings can be cached for exact
reproduction, and the procedure runs on commodity hardware [CITE: linear-evaluation protocol for
SSL]. Public frozen-embedding releases for fundus cohorts, including precomputed DINOv3 features
for BRSET at smaller scales, show that the community already treats frozen extraction as a
first-class research mode [VERIFY: PhysioNet BRSET/mBRSET embedding release; model sizes]. What
that line of work largely omits is a head-to-head, statistically tested comparison of paradigms at
a fixed protocol on a clinical endpoint, which is the gap we occupy.

**Multi-task heads versus linear probes.** Two head designs bracket the realistic options for a
frozen reader. A linear probe measures the raw linear separability of the frozen features, whereas
a shared multi-task head lets dense binary tasks provide auxiliary supervision that can regularise
sparse targets, a mechanism expected to matter most when positive counts are small [CITE:
multi-task learning with auxiliary tasks; CITE: small-sample multi-task regularisation]. Because
the two heads can reorder backbones, studying both is necessary to separate "which representation
is best" from "which representation is most linearly accessible," and our results show this
separation is endpoint-dependent rather than uniform.

**Statistical uncertainty in model comparison.** Point-estimate leaderboards are insufficient when
test sets are modest and positives are sparse, because differences smaller than the sampling noise
are read as rankings. Resampling-based confidence intervals address this, and when a cohort
contains multiple images per patient the resampling unit must be the patient, not the image, to
avoid understating uncertainty [CITE: bootstrap CIs for AUC; CITE: clustered/patient-level
resampling]. Few retinal foundation-model comparisons report paired, patient-clustered intervals,
so reported orderings are hard to interpret; we make these intervals the basis for every claim and
report differences as "supported" only when the paired interval excludes zero.

### Gap table

| Prior direction | What prior work typically does | Remaining gap | What this paper adds |
|---|---|---|---|
| BRSET baseline studies | Supervised CNNs (e.g. ConvNeXt-family), often image-level split, single endpoint focus [VERIFY] | No frozen multi-paradigm comparison; limited uncertainty reporting | Frozen backbone × head matrix on a patient-level split with bootstrap CIs |
| Retinal foundation-model papers | Fine-tune a domain-specific backbone; report transfer across tasks [CITE] | Frozen behaviour and matched-protocol comparison to general models under-reported | Frozen RETFound-Green at matched and native resolution, side-by-side with general backbones |
| General SSL ViT papers | Demonstrate DINOv2/DINOv3 transfer, often on natural-image or mixed medical benchmarks [CITE] | DINOv3-Large at this frozen multi-task fundus protocol not characterised [VERIFY] | DINOv3-Large evaluated under identical frozen extraction and heads on BRSET |
| DR screening studies | Report referable-DR discrimination, frequently fine-tuned, often without CIs [CITE] | Frozen-feature referable-DR with patient-clustered CIs scarce | Referable-DR AUROC/AUPRC with patient-clustered bootstrap CIs for every cell |
| Multi-task retinal models | Train joint heads, report aggregate metrics [CITE] | Head axis rarely contrasted against linear probes under fixed features | MT vs LP contrast on identical frozen embeddings, per endpoint |
| Foundation-model benchmarking practice | Point-estimate leaderboards across backbones [CITE] | Differences within sampling noise reported as rankings | Paired bootstrap with explicit "not separable" outcomes |
| External-validation studies | Cross-population/cross-device transfer for DR [CITE] | — (out of scope here) | This paper restricts claims to internal BRSET; external transfer is deferred |

The gap this paper fills is narrow and specific: a controlled, frozen backbone × head comparison
on a patient-split fundus cohort, evaluated on referable DR as the primary endpoint with
patient-clustered bootstrap confidence intervals, so that paradigm, scale, recency, head, and
acquisition protocol can each be assessed against the uncertainty that determines whether an
apparent difference is real.

---

## PLACEHOLDERS & OPEN ITEMS

**[CITE] — references to insert (do not invent; locate and verify each):**
- Landmark automated DR screening / regulatory reader studies.
- Reviews on clinical readiness of DR AI (patient-level evaluation, calibration, robustness).
- BRSET dataset paper; smartphone-fundus companion dataset paper.
- RETFound and RETFound-Green model papers.
- DINOv2 model paper; DINOv3 model paper.
- ConvNeXt paper; ResNet paper.
- Linear-evaluation protocol for self-supervised learning.
- Multi-task learning with auxiliary tasks; small-sample multi-task regularisation.
- Bootstrap CIs for AUC; clustered / patient-level resampling methodology.
- Retinal foundation-model fine-tuning transfer studies; DR screening (fine-tuned) studies;
  multi-task retinal models; external-validation (cross-population/cross-device) DR studies.

**[VERIFY] — comparator/specifics recorded internally but not yet confirmed against source:**
- BRSET full dataset name and release version.
- Generalist-vs-specialist ocular foundation-model benchmark (claim that scaling does not
  uniformly help in medical imaging; which model wins on average).
- PhysioNet BRSET/mBRSET frozen-embedding release and the exact DINOv3 model sizes it covers
  (used to soften any "first DINOv3 on BRSET" framing — prior DINOv3-S/B frozen embeddings exist).
- Whether DINOv3-Large specifically has been characterised on BRSET under this frozen multi-task
  protocol (internal notes suggest not, but published DINOv3-on-BRSET work exists).
- Original BRSET baseline split type (image-level vs patient-level) and head/endpoint specifics.
- External published referable-DR / DR AUROC numbers (e.g. fine-tuned DINOv3, ConvNeXt V2
  baseline, frozen-encoder external-validation benchmark): all currently unverified; confirm
  values, endpoint definitions, and author attributions before citing. Per internal correction,
  one BRSET benchmark paper must be attributed to "Aghabeigi Alooghareh et al." (not "Khan
  et al."), and a "0.79" hypertension figure is a weighted-F1, not an AUROC.

**[from artifact] — numbers used in the draft, with source records (all internal,
final_test_result=false; preliminary):**
- Split / cohort: BRSET test 1,623 images, 854 patients; patient-level 60/15/15/10; 0% patient
  overlap — split audit record.
- Referable DR: grade ≥ 2; 73 positives / 1,623 (4.50%); per-cell AUROC/AUPRC with
  patient-clustered bootstrap 95% CIs (2,000 resamples, seed 42) — referable-DR analysis record.
- Headline cell: RETFound-Green native-392 MT referable-DR AUROC 0.985 (95% CI 0.964–0.997).
- Only supported top-tier backbone AUROC edge: RETFound-Green native-392 MT vs DINOv2-Base MT,
  Δ +0.012 (95% CI 0.004–0.022).
- Multi-condition macro-AUROC point estimates (six binary tasks): confidence intervals available
  for the four original backbones only; RETFound and DINOv3 rows must be cited as point estimates.

**Open framing items:**
- The frozen-vs-fine-tuned accuracy-cost datapoint is not yet run; do not claim frozen features
  equal fine-tuned state of the art, and do not assert cross-paper equivalence with externally
  reported fine-tuned numbers.
- Do not name or brand the architecture; not supported by evidence.
- Keep all secondary (systemic/multi-condition) endpoints as screening/risk-flag language; no
  whole-body-diagnosis or early-detection overreach; no external-validation, fairness, continual-
  learning, or deployment claims in these sections.
