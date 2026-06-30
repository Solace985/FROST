# Paper 1 — Discussion & Conclusion (Draft)

*Continues the manuscript begun in the Abstract/Introduction/Related Work and Methods/Results
drafts. Manuscript-facing names and register match those files. The Discussion introduces no new
numbers or findings; it interprets the results already reported.*

---

## Discussion

Frozen retinal foundation-model features support strong and reproducible referable-DR screening on
BRSET. Across the matrix, a lightweight head trained on cached embeddings — without any update to
the backbone — distinguishes referable from non-referable eyes with discrimination that reaches
0.985 (95% CI 0.964–0.997) in the strongest cell, and exceeds 0.97 in every multi-task
configuration. The practical reading is that the expensive component, the backbone, can be reused
as-is: a screening triage head is cheap to fit and cheap to refit, and the entire comparison is
exactly reproducible because the features are computed once and stored. This shifts the design
question for a frozen fundus reader away from "which backbone, fine-tuned how" toward "which frozen
representation, with which head, for which endpoint."

The clearest answer to that question is that representation quality, not parameter scale, separates
the top of the matrix. The single supported top-tier backbone advantage on referable DR belongs to
the smallest model — a ViT-Small retina-specific backbone with 384-dimensional embeddings — which
edges a general transformer of larger width and is not separable from the ViT-Large general models
above it. Domain-specific pretraining therefore buys a representation that parameter scale and model
recency did not match here under frozen extraction. This is the finding with the most direct
deployment consequence: a smaller embedding means a smaller feature cache, a lighter head, and
cheaper inference, so the domain-pretrained representation is not only competitive on the clinical
endpoint but cheaper to serve and to store.

That result also carries a caution for how the field compares foundation models. Because the
pretraining paradigm and the operating point moved performance more than size or generation did,
"newest or largest wins" is not a safe default when selecting a frozen backbone for fundus
screening. A next-generation general transformer did not separate from its predecessor on referable
DR, and a small domain model held its own against both. Backbone selection for a frozen reader
should therefore be settled empirically on the target endpoint, not inferred from a model's scale or
release date.

The head axis is best read as a task-structure decision rather than a generic upgrade. The
multi-task head's benefit concentrated on the sparsest conditions — most visibly on the rarest,
AMD — where joint training with denser tasks supplies auxiliary regularization that a per-task
linear probe lacks, and its advantage was not monotone in backbone strength. On referable DR, an
endpoint already near the discrimination ceiling, the multi-task head added little additional
ranking accuracy but still improved precision–recall. Joint training is thus worth adopting when
rare targets need the regularization of a shared trunk, and is otherwise a marginal choice on an
endpoint where discrimination is already saturated.

Acquisition protocol emerged as a first-class experimental axis, which is a methodological
contribution as much as a result. Changing only the resolution and pooling of the retina-specific
model — matched-224 versus native-392 — shifted its multi-condition macro by a margin that, on the
descriptive point estimates, appears comparable to a change of backbone, and produced a supported
precision–recall gain on referable DR even though the referable-DR AUROC difference was not
separable. Read suggestively, since the macro-scale comparison is a point estimate rather than a
tested difference, this implies that cross-paper "which retinal model is best" claims are confounded
whenever resolution and pooling are not held constant. Preprocessing belongs in the comparison
protocol, not in a footnote.

The endpoint structure of the study is internally consistent and disciplines what should be
headlined. Strong referable-DR discrimination coexisted with weak five-class grading from the same
frozen features, which is expected: separating referable from non-referable eyes is a genuinely
easier problem than ranking all five ordinal grades on a distribution dominated by a single class.
This coexistence justifies referable DR as the primary DR endpoint and argues against headlining
multi-class accuracy on a heavily imbalanced grade distribution, where a high aggregate number can
hide collapse on the rare, clinically important grades.

Finally, paired uncertainty is part of what the study contributes to how these comparisons should be
read. Many apparent backbone differences fell within sampling noise once intervals were computed at
the patient level, and reporting them as "not separable" rather than as rankings prevents
point-estimate ordering from being mistaken for evidence. Against the closest prior work — a frozen
embedding release, an adaptation-focused model comparison, and point-estimate leaderboards — this
paper adds more backbones and pretraining paradigms, an explicit head axis, a preprocessing axis,
and patient-clustered paired testing, without asserting equivalence to any external result or
superiority over fine-tuned systems. Its contribution is a controlled, uncertainty-aware reading of
frozen representations on a clinical screening endpoint, not a new leaderboard entry.

### Limitations

This study is bounded to internal BRSET, so it establishes no cross-population or cross-device
robustness. That matters because screening readers meet distribution shift in deployment, and a
single-cohort result cannot speak to it; we therefore restrict every claim to BRSET and frame
nothing as generalization. External and smartphone-cohort validation, carrying the same endpoint and
uncertainty machinery forward, is the direction that resolves this.

The design is frozen-only, so the accuracy cost of declining to fine-tune is not quantified here.
This matters because fine-tuning can lift performance and a reader may want that headroom; we made
the choice deliberately, to isolate representation quality from optimization variance and to keep
the comparison reproducible from cached features. A controlled frozen-versus-fine-tuned datapoint on
the referable-DR endpoint is the natural follow-up.

Fine-grained five-class severity is weak, collapsing on the rare grades under the single unweighted
recipe, and referable DR is itself a thresholding of that same head. This matters because severity
staging has clinical uses beyond the referral decision; we mitigated it by demoting severity to a
secondary, explicitly limited endpoint rather than headlining it. Ordinal-aware modelling and
imbalance-sensitive training are the route to a stronger grading result.

The rarest conditions carry wide implicit uncertainty, with AMD, hypertensive retinopathy, and
macular edema each having few positives in the test partition. This matters because per-task AUROC
is unstable at such counts; we mitigated it by reporting precision–recall alongside AUROC and by
treating sparse-task comparisons cautiously. Larger cohorts with more positives are needed to narrow
these intervals.

Two labels are not retina-derived diagnoses: the diabetes label is record-derived, and "hypertensive
retinopathy" is an ophthalmological retinal grading rather than a systemic measurement. This matters
because a strong AUROC on a proxy label can be over-read as systemic diagnosis; we framed these
results as screening or risk-flagging signals throughout. Confirmatory, retina-anchored labelling
would let these endpoints carry diagnostic weight.

Uncertainty coverage is uneven and no equivalence test was run: multi-condition macro intervals were
finalized for the original backbones, the remaining macro values are reported descriptively, and the
top-tier relationship is stated as non-separation rather than tested equivalence. This matters
because a descriptive value should not be read as a tested one; we restricted every comparative claim
accordingly. Extending interval coverage to all rows and adding a pre-specified non-inferiority test
would convert these descriptive readings into formal ones.

A single neutral recipe was applied to every cell, excluding imbalance mitigation and per-backbone
tuning. This matters because tuned or rebalanced training could reorder cells; we excluded it by
design so that differences reflect the backbone and head rather than optimization effort. Its effect
belongs in a dedicated ablation rather than in this controlled comparison.

### Future work

Future work extends this controlled reading outward without loosening its claims. External validation
on independent populations and a smartphone-acquired cohort would test whether the frozen referable-DR
result holds under distribution and device shift, carrying the same referable-DR endpoint and
patient-clustered uncertainty forward as the basis for comparison. A frozen-versus-fine-tuned datapoint
and ordinal-aware severity modelling would close the two methodological gaps left open here. The
integration components this pipeline is built for — calibration, subgroup fairness auditing, and
continual adaptation — are reserved for subsequent work and are not claimed here.

---

## Conclusion

We reported a controlled, patient-level, frozen backbone × head evaluation on BRSET, with referable
DR (grade ≥ 2) as the primary endpoint and patient-clustered bootstrap confidence intervals for
every cell. Frozen foundation-model features provide a strong, reproducible referable-DR screening
baseline that a lightweight head can realise without fine-tuning the backbone. The instructive result
is not which model won but that none clearly did: the top tier was not separable under paired
bootstrap, and the only supported top-tier advantage was held by the smallest, domain-pretrained
model — so representation quality bought cheaply through domain-specific pretraining can matter as
much as parameter scale, model recency, or head complexity, and acquisition protocol can rival the
choice of architecture as a lever on measured performance. Read together, these findings show that a
frozen fundus reader's ranking is not a fixed property of its backbone but a joint function of the
endpoint, the head, the preprocessing protocol, and the statistical uncertainty against which it is
judged. Establishing whether this strong internal screening baseline holds under population and
device shift is the next step.
