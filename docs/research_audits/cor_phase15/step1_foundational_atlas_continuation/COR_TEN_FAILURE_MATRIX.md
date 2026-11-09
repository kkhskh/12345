# COR: the ten tracked failures (F1–F7, F25, F29, F30), the O/D question, and candidate decisions (mandate §13, §14)

**Historical verdicts are unchanged.**
- `OBSTRUCTION_READY = NO`
- `PHASE_2 = NO GO`
- all Step 1 verdicts
- the atlas verdicts, including the PP2 and PP5 failures

This document adds evidence from the continuation only.

Revised after review round R2 (`reviews/ADJUDICATION_R2.md`); corrected claims are marked [R].

**Labels.**
- [T] theorem.
- [N] exact numerical check.
- [E] learned-model evidence.
- [C] prospective confirmation.
- [PH] post hoc.

## 1. Candidate decisions (§13): what reaches the logits, in which direction, and whether it helps

Fixed pairs: (c1, c2) are the base top-2 candidates at q. The population is the frozen PP1 edits with NB = 800, giving 50,130 S1 edits (`scripts/decision_ledger.py`, `logs/decision_ledger.json`). Models: TF s10 (class A′), TF s12 (C), DeepSets s0 (A).

Class comparisons rest on **one model per class**, so no class effect can be tested [R].

| question | finding | type |
|---|---|---|
| margin direction under intervention | sign agreement with Bayes 0.996–1.000 on filtered PP1 edits, and 0.963–0.980 unfiltered on the high-ratio family (all failures at \|Δm\| < 1). [R] This is explained empirically by pointwise margin errors being smaller than the effects (C1 coverage 98.6–99.7%). It is **not** explained by C4, which is vacuous at the observed ε | [E] |
| ranking reversals | 3,562 edits flip the Bayes argmax. After the edit the models' argmax equals the Bayes argmax in 0.82–0.83 of them. The models flip their *own* argmax in 0.59. [R] No matched reference rate was computed (for example, agreement on non-reversal endpoints matched for margin), so 0.82 is uninterpreted | [E] |
| confidence versus preference | 11,238 edits change Bayes confidence by > 0.1 without changing its argmax. The models change their argmax in 0.13 of them. [R] The "near-tie" explanation is a hypothesis; it was not stratified | [E] |
| candidate-specific support ("more agreement helps the group's own candidate") | agrees with the Bayes sign in **1.000** of interior edits but only **0.111** at the count boundary (5 ↔ 6). n = 16,832. The interior result is very probably a theorem (THEOREMS §7, Proposition SR, sketch). The boundary failure is the paradox of unanimity (Gunn et al. 2016) | [N, P-sketch] |
| decision under misleading relations | on p_c = 0 episodes in which an *honest* group agrees in all 6 questions, the models disagree with the true answer in 0.63–0.65 of questions. [R] **The T4-Bayes plug-in itself disagrees in 0.587.** So the models behave as T4-Bayes does: this is prior mismatch, not a network-specific failure. Elsewhere on p_c = 0 the T4 plug-in disagrees in 0.41, against s10 0.07, s12 0.37 and DeepSets 0.47 | [E, N] |

**The six-level distinction asked for in §13**, filled in per mechanism:

| level | A′ (tag-route) | C (reread) | A (pool) |
|---|---|---|---|
| relation internally represented | yes: 2-D tag, contrast-identified; mean-ablation 0.54–0.68 | yes: tag *and* other-question content | yes: the pool decodes the colluder at 0.97 |
| an operator responds to it | routing pattern (pattern patch 0.84–0.85) | routing (head 3) and rereading (head 0) | ρ-MLP (not dissected) |
| the operator has a causal effect | yes (freeze, patch) | yes (heads 0+3: IIA 0.76) | not tested internally |
| the effect reaches the candidate logits | yes (attention-output site IIA 0.87–0.94) | yes (all heads 0.88) | yes (behaviour) |
| the effect has a predictable direction | **only given the generator semantics**: sign = Bayes Δm (D, C4). The generator-free "support" rule is right in the interior and wrong at the boundary (0.11) | same | same |
| the effect improves correctness | on support: yes, accuracy 0.739–0.742 against own 0.576. Under a misleading relation (p_c = 0, honest perfect group): **no**, 0.63–0.65 disagreement with the truth. That matches the T4-Bayes plug-in (0.587), so the failure belongs to the training prior, not to the implementation | same | same |

**The frozen quantities that determine the sign.**
- The sign of a relation-conditioned margin change is fixed by:
  - the generator's role semantics (a_r and the configuration prior);
  - the candidate-conditioned configuration posterior (Theorem D).
- Both are computable without the network. They are independently checkable and do not encode the logit change.
- Where those semantics are *not* given, no signed implication exists:
  - D2, plus the boundary result above (0.11);
  - the OD1 characterisation (§2).
- **Strongest valid conditional statement.** Under T4, for edits that keep the edited group below full agreement, the generator-free support rule has the Bayes sign. Proposition SR gives a proof sketch; it holds empirically on 1.000 of 16,832 edits.
- **The counterexample at its limit** is the full-agreement boundary (0.111), which is the paradox of unanimity.

## 2. O/D revisited (§14) [R: corrected]

**What holds.**
- **OD1 (the "if" direction).** The unweighted discrepancy ranking equals the Bayes ranking when sources are conditionally independent, equally accurate and symmetric, and Y is uniform.
- **OD1′ (proposed by reviewer NV; the "only if" part is unverified).** For independent symmetric sources, D never strictly reverses the Bayes order iff the log-odds weights satisfy the weighted-versus-simple-majority coincidence condition. Exact ranking equality needs equal weights [R3-F11]. A sufficient condition is a weight ratio below 3/2 for six sources (Nitzan–Paroush 1982; Ben-Yashar–Paroush 2001).

**Withdrawn [R].** The pre-review "the bridge holds iff the source semantics are trivial". Unequal accuracies inside the band, and the package's own correlated-pair case (1.000), contradict it.

**Where the bridge breaks.**

| case | consistency |
|---|---|
| unequal accuracies outside the band (weight ratio 3.23) | 0.989 |
| biased errors | 0.980 |
| T1 roles: noise has weight 0 | 0.900 |
| T4 roles: a copy counts once, and the colluder's value is uninformative | 0.723 |

Under T4, min-discrepancy accuracy (0.552) is below own-question Bayes (0.576).

**Determination.** O/D's unweighted discrepancy coincides with Bayes when the semantics induce nearly equal log-odds weights. It fails exactly where the latent-reliability family puts its relations: zero-weight sources and dependent (copying) sources. The repaired form is weighted voting with semantics-derived weights and copy discounting, which is established Bayesian truth discovery, not a new obstruction theory.

## 3. The ten failures

| id | failure | continuation evidence | status change from the atlas |
|---|---|---|---|
| **F1** | missing noncircular causal bridge | the relation → decision bridge is *characterised*:<br>• on the filtered PP1 edits, C1 coverage is 98.6–99.7% [R: the C4 bound is vacuous];<br>• on the high-ratio family, every \|Δm\| ≥ 1 sign is correct (coverage not computed);<br>• internally, the bridge runs through the tag subspace (ablation moves outputs toward the own-question posterior in every Transformer seed tested) and class-dependent operators (§COMPUTATIONAL_CLASSES) | **advanced, but re-scoped**: the bridge is real, and its implementation-independent part is pointwise Bayes-closeness. Nothing COR-specific survives as the bridge |
| **F2** | lower discrepancy can worsen a candidate | Proposition SR (interior: the sign is guaranteed, sketch) and the boundary failure (0.11; the paradox of unanimity). U-scope counterexample: the union-role posterior is non-quasi-concave in 45/49 contexts under a mixture over α [T-sketch, N] | **characterised further**: a guaranteed-sign region with its boundary, plus a second reversal mechanism (mixtures) |
| **F3** | consistency ranking ≠ preference | OD1 (if) + OD1′ (the exact weighted-voting condition) [T]; consistency 0.72 under T4 [N] | **sharpened** [R: the pre-review "iff trivial semantics" is withdrawn]: consistency = preference exactly under the OD1′ weight condition, which zero-weight or copying sources violate |
| **F4** | no distinctive prospective prediction | P-A: a frozen *internal* predictor of the extrapolation direction **failed** on held-out seeds. Its MAE was 0.126 against 0.091 for the architecture baseline, and it ordered the two held-out Transformers backwards. The behavioural proxy A2 passed (MAE 0.026) but is near-tautological. P-B (PP3 features): B1 passed, B3 failed, B4 passed [C/F] | **unchanged**: no distinctive *internal* prediction survived. P-B1 and P-B4 are confirmed but restricted (one seed per arm) |
| **F5** | representation and map non-identifiability | R1: a split-invariant attribution (the Owen quotient game; the gauge part is vacuous at this site), with limit L1. Seed 12: necessary-but-not-sufficient tag with at least four channels [T, E] | **partially repaired, by known means**: class-level invariance under exact splits only. Variable-level identification is still alignment-class-relative (I7) |
| **F6** | output optimisation hidden in repairs | all new subspaces from input contrasts; counterexample models constructed, not fitted | procedural |
| **F7** | outcome-dependent tuning | P-A frozen before any held-out model existed. Compute-budget cuts were logged *before* training (`logs/COMPUTE_LOG.txt`). Bugs found and fixed: the projector union and inf − inf (by the authors); a NaN-argmax artefact and a hash-order dependence (by review R2) | procedural |
| **F25** | same consistency, different decisions | the same perfect-agreement pattern is correctly discounted under T4 and wrongly discounted under p_c = 0, both by every model and by the T4-Bayes plug-in (0.587) [E, N]. The I2 role-swap line gives identical episode laws with opposite correct decisions [T] | **sharpened**: exact at ρ* |
| **F29** | no advantage over existing methods | BASELINES_AND_NOVELTY: no new mathematical theory.<br>• V1 is weighted voting with copy discounting (Nitzan–Paroush; Dong et al.);<br>• OD1/OD1′ is weighted-vs-simple-majority coincidence;<br>• R1 is the Owen quotient game | **unchanged** |
| **F30** | excessive unfrozen choices | remaining conventions are listed:<br>• linear-subspace alignment class;<br>• sites;<br>• τ for pattern classes;<br>• probe classes for the PP2/PP3 signatures;<br>• NB = 800 for the ledger | partial |
