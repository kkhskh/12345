# COR: the ten tracked failures (F1–F7, F25, F29, F30), the O/D question, and candidate decisions (mandate §13, §14)

**Historical verdicts are unchanged.**
- `OBSTRUCTION_READY = NO`
- `PHASE_2 = NO GO`
- all Step 1 verdicts
- the atlas verdicts, including the PP2 and PP5 failures

This document adds evidence from the continuation only.

**Labels.**
- [T] theorem.
- [N] exact numerical check.
- [E] learned-model evidence.
- [C] prospective confirmation.
- [PH] post hoc.

## 1. Candidate decisions (§13): what reaches the logits, in which direction, and whether it helps

Fixed pairs: (c1, c2) are the base top-2 candidates at q. The population is the frozen PP1 edits with NB = 800, giving 50,130 S1 edits (`scripts/decision_ledger.py`, `logs/decision_ledger.json`). Models: TF s10 (class A′), TF s12 (C), DeepSets s0 (A).

| question | finding | type |
|---|---|---|
| margin direction under intervention | sign agreement with Bayes 0.996–1.000 for all classes. This is a consequence of natural KL ≈ 10⁻³ with density ratio W ≈ 1 (C4) | [E, T] |
| ranking reversals | 3,562 edits flip the Bayes argmax. After the edit, the models' argmax equals the Bayes argmax in 0.82–0.83 of them, for all three classes alike | [E] |
| confidence versus preference | 11,238 edits change Bayes confidence by > 0.1 without changing its argmax. The models change their argmax in 0.13 of them, again the same for all classes. These are near-tie effects of the KL error, not a class signature | [E] |
| candidate-specific support ("more agreement helps the group's own candidate") | agrees with the Bayes sign in **1.000** of interior edits but only **0.111** at the count boundary (total count 5 ↔ 6). n = 16,832 | [N] |
| decision failure under misleading relations | on p_c = 0 episodes in which an *honest* group agrees in all 6 questions, all three T4 models disagree with the true Bayes answer in 0.63–0.65 of questions. They treat the honest group as a colluder. Elsewhere on p_c = 0: s10 0.07, s12 0.37, DeepSets 0.47 | [E] |

**The six-level distinction asked for in §13**, filled in per mechanism:

| level | A′ (tag-route) | C (reread) | A (pool) |
|---|---|---|---|
| relation internally represented | yes: 2-D tag, contrast-identified; mean-ablation 0.54–0.68 | yes: tag *and* other-question content | yes: the pool decodes the colluder at 0.97 |
| an operator responds to it | routing pattern (pattern patch 0.84–0.85) | routing (head 3) and rereading (head 0) | ρ-MLP (not dissected) |
| the operator has a causal effect | yes (freeze, patch) | yes (heads 0+3: IIA 0.76) | not tested internally |
| the effect reaches the candidate logits | yes (attention-output site IIA 0.87–0.94) | yes (all heads 0.88) | yes (behaviour) |
| the effect has a predictable direction | **only given the generator semantics**: sign = Bayes Δm (D, C4). The generator-free "support" rule is right in the interior and wrong at the boundary (0.11) | same | same |
| the effect improves correctness | on support: yes, accuracy 0.739–0.742 against own 0.576. Under a misleading relation (p_c = 0, honest perfect group): **no**, 0.63–0.65 disagreement with the truth | same | same |

**The frozen quantities that determine the sign.**
- The sign of a relation-conditioned margin change is fixed by:
  - the generator's role semantics (a_r and the configuration prior);
  - the candidate-conditioned configuration posterior (Theorem D).
- Both are computable without the network. They are independently checkable and do not encode the logit change.
- Where those semantics are *not* given, no signed implication exists:
  - D2, plus the boundary result above (0.11);
  - the OD1 characterisation (§2).
- **Strongest valid conditional statement.** Inside the interior of the T4 count range, the support rule's sign holds for every scored edit (1.000). This is a restricted, independently checkable rule.
- **The counterexample at its limit** is the boundary crossing.

## 2. O/D revisited (§14)

### OD1 (THEOREMS §5)

The unweighted discrepancy ranking equals the Bayes ranking *exactly* when sources are conditionally independent, equally accurate and symmetric in their errors [T, N: consistency 1.000].

When an assumption fails, so does the bridge:

| assumption broken | consistency |
|---|---|
| unequal accuracies | 0.989 |
| biased errors | 0.980 |
| T1 roles | 0.900 |
| T4 roles | 0.723 |

Under T4, min-discrepancy accuracy (0.552) falls *below* own-question Bayes (0.576).

### Determination (sharpened from the atlas)

O/D is the Bayes statistic in exactly the case where there is no relation to infer. Every relation in the latent-reliability family (reliability heterogeneity, copying, collusion) is what O/D lacks.

The repaired form is a weighted discrepancy with semantics-derived weights: Dawid–Skene, or Dong et al. 2009 vote discounting. That is established Bayesian truth discovery, not a new obstruction theory.

**Scope.** The "if" direction is proved in general. The "only if" direction is shown only on the families tested.

## 3. The ten failures

| id | failure | continuation evidence | status change from the atlas |
|---|---|---|---|
| **F1** | missing noncircular causal bridge | the relation → decision bridge is now *explained*:<br>• PP1 sign agreement follows from KL-closeness under W ≈ 1 (C4);<br>• all failures are certified-margin failures (C1 never violated in 10 models);<br>• internally, the bridge runs through the tag subspace (necessary in all four Transformer seeds tested) and class-dependent operators (§COMPUTATIONAL_CLASSES) | **advanced, but re-scoped**: the bridge is real, and its *implementation-independent* part is exactly Bayes-closeness. Nothing COR-specific survives as the bridge |
| **F2** | lower discrepancy can worsen a candidate | the boundary result above: 0.11 support-rule agreement exactly at the colluder boundary; U-scope counterexample (two peaks with a mixture over α) [T, N] | **characterised further**: the reversal region is known, and a second reversal mechanism exists (mixtures) |
| **F3** | consistency ranking ≠ preference | OD1 characterisation [T, N] | **sharpened into an iff-style statement** (the "only if" direction tested, not proved) |
| **F4** | no distinctive prospective prediction | P-A, a frozen *internal* predictor of the extrapolation direction tested on held-out seeds. Outcome: see PROSPECTIVE_PREDICTIONS.md | **pending**: F4 changes only if P-A1 passes |
| **F5** | representation and map non-identifiability | R1: a split- and gauge-invariant attribution exists (quotient game), with limit L1 (pattern-changing reparametrisations). Seed 12: necessary-but-not-sufficient tag [T, E] | **partially repaired**: there is an invariant attribution at class level. Variable-level identification is still alignment-class-relative (I7) |
| **F6** | output optimisation hidden in repairs | all new subspaces from input contrasts; counterexample models constructed, not fitted | procedural |
| **F7** | outcome-dependent tuning | P-A frozen before any held-out model existed. Compute-budget cuts were logged *before* training (`logs/COMPUTE_LOG.txt`). Two bugs found and fixed: the projector union, and inf − inf | procedural |
| **F25** | same consistency, different decisions | misleading-relation result: the same perfect-agreement pattern is correctly discounted under T4 and wrongly discounted under p_c = 0 by every model [E]. The I2 role-swap line gives identical episode laws with opposite correct decisions [T, N] | **sharpened**: exact at ρ* |
| **F29** | no advantage over existing methods | BASELINES_AND_NOVELTY: no new mathematical theory. V1 is Dong et al.'s vote discounting; OD1 is majority-vote optimality | **unchanged** |
| **F30** | excessive unfrozen choices | remaining conventions are listed:<br>• linear-subspace alignment class;<br>• sites;<br>• τ for pattern classes;<br>• probe classes for the PP2/PP3 signatures;<br>• NB = 800 for the ledger | partial |
