# R7 — Scientific defense review

Package: `step1_mechanism_identification_continuation` at commit `d1aa97d`.
Role: find legitimate restricted results that the package undersells because they are conditional or use known mathematics, without inflating them. Mandate references: §13 (signed frozen weights are not automatically circular), §14, §17 (negatives must be scoped), §22 (what counts as advance).

Two new exploratory checks were run for this review. Scripts are in `/tmp/r7/` (outside the package) and use only package code and checkpoints:

- `indep2.py`: an ordinal-additivity (independence / cancellation) test on C′'s closed-form margin.
- `learned_indep.py`: the same test on learned checkpoints.

The test works as follows. If `m(a,b) = h(f(a) + g(b))` with h strictly increasing, then `sign(m(a1,b) − m(a2,b))` must be the same for every b. A robust reversal (both differences larger than a tolerance) therefore rules out **every** model of that form. The test is invariant to any monotone transform of the output, so it applies equally to logits, probabilities and confidence scores.

---

## 1. T-RF is a decision-level theorem; the package states its consequences as if it covered all behaviour (oversold negative)

**(a)** T-RF is proved for `F(k_A, k_B, p)`, a function of discrete per-source support (sign or argmax). The relation-free twin (`src/relation_free_equivalent.py`) is checked only on `sign m`. But the package draws conclusions about behaviour in general:

- `MAIN_RESEARCH_RESULT.md` §14 says "Behaviour cannot identify it."
- The F4 row says T-RF "rules out" a behavioural prediction.
- Synthesis Q15 says "behaviour cannot do it".

The margin magnitude, C′'s confidence, is behaviour too, and T-RF says nothing about it.

**(b) Evidence** (new; C′ closed form, `src/cprime.py`). All four points lie outside C′'s dead bands:

| (a, b) | m |
|---|---|
| (1, 1) | +1.890 |
| (−1, 1) | −2.631 |
| (1, −1) | −2.631 |
| (−1, −1) | −1.890 |

- Raising `a` from −1 to 1 **raises** m by 4.52 when b = 1.
- It **lowers** m by 0.74 when b = −1.

No model of the form `h(f(a) + g(b) + const)` with h increasing can do this. That form includes the T-RF twin: its margin is `(φ_A + φ_B − τ)/4` through a monotone LN read. On the 60 × 60 grid, 1,854 of 3,540 ordered a-pairs show reversals larger than 0.5.

So C′'s graded output certifies, **from input–output behaviour alone**, that some cross-source interaction is used. The twin matches C′'s decisions but not C′'s confidence ordering.

There is also a C′-specific, checkable signature. When both sources agree on c′, the circuit gives **less** confidence in c′ than when the sources conflict and it defers (−1.890 vs −2.631). That follows from C′1*, because the prior's value mass dominates under conflict routing.

**Limits (do not inflate).**

- The test certifies a cross-source interaction. It does not certify which relation is used. A generic `ab` interaction (reader R3 in A-matchS) would also fail ordinal additivity.
- B-MLP still matches everything. Ordinal non-additivity is not a predictive advantage over a matched-capacity nonlinear model (F29 stands).
- The test applies only when graded outputs are observable.

**(c) Proposed wording.**

- For `MAIN_RESEARCH_RESULT.md`: "For candidate-directed rules, *decision* behaviour cannot identify relation use (T-RF). Graded output (margin or confidence) can certify that a cross-source interaction is used, when it violates ordinal additivity, as C′'s does. It cannot certify that the interaction is the agreement relation; only internal interventions (X4, C2) do that."
- For the F4 row: "sharpened negative at the **decision** level; partial at the graded-output level (ordinal non-additivity, exploratory) and at the internal level."

## 2. The learned-model negative, scoped correctly, is *stronger* at the margin level than the package claims, and narrower in scope

**(a)** The package says T3 and T4 margins are "largely additive" (`R²_add` ≈ 0.65–0.70). That reading leaves room for a 30–35% non-additive, possibly relational, magnitude component. In the other direction, "the learned models do not develop relation computation for reconciliation tasks" generalises past the tested family.

**(b) Evidence** (new; `learned_indep.py`, 50 × 50 grid on [−3, 3]², tolerance 0.1 SD of the margin, seeds 0 and 3):

| task / arch | robust order reversals | largest reversal |
|---|---|---|
| T3 / TF1, both p | 0 / 1225 | 0 |
| T3 / TF2 | 0 / 1225 in 3 of 4 cases; 84 / 1225 for s0, p = −1 | 0.10 SD |
| T4 / TF1 | 0 / 1225 | 0 |
| T2 / TF1 | 0 / 1225 | 0 |
| **T1 / TF1** (positive control) | **625 / 1225** | **≈ 2.0 SD** |

So the non-additive part of the T3 and T4 margins is almost entirely a **monotone distortion of an additive score**. It is not an interaction.

A caveat on what the training objective constrains. T3 and T4 use hard targets (`semantics.py`: `soft = (y+1)/2`), so the objective never constrains margin magnitude. The tasks therefore gave no incentive for graded relational confidence.

**Scope of the tested family.**

- pre-norm, `d = 16`, 2 heads, GELU MLP width 32, 1–2 layers;
- 3,000 steps, hard targets, one data mixture;
- transfer (`logs/transfer.stdout`) covered **only T1 and T2** at d = 8, d = 32 and 1 head. It did not cover T3, T4, T5a or T5b, nor head count 4 or the MLP-width variants.

So the reconciliation negative has had no transfer test at all.

**(c) Proposed wording.** "In the tested family (pre-norm Transformers with d = 16, 1–2 layers, 3,000 steps, hard targets; no transfer variants for T3–T5b), models trained on candidate-directed reconciliation tasks (T3, T4, T5b) solve them without a causally used relation. Their binary margins are ordinally additive in the sources (exploratory check, R7). This is expected from T-RF: those tasks do not demand a relation. It is a reproducible failure-to-emerge (§22, last item). It is not evidence about tasks whose targets *demand* relation-dependent strength or confidence, which were not tested."

Also: delete "They solve them additively" as a general statement, and keep it attached to the tested family.

## 3. T3–T5b were, by the package's own theorem, not tests of reconciliation; say so as a design finding, not a model finding

**(a)** `LEARNED_SYNTHETIC_MECHANISMS.md` §1.3 does say "because those tasks do not need it". But the main result still lists the T3–T5b outcome among the negatives as if it were a finding about learned models. The informative content is a property of the **task class**: the mandate's T3 and T5 specifications are threshold-representable. That is a substantive design result for §9 itself.

**(b) Evidence:**

- the T-RF proof;
- the LP feasibility checks (`logs/additive_lp.json`: T3 and T4 feasible, T1 infeasible);
- B-add reaches 0.997 on T3 and 0.991 on T5b.

**(c) Proposed wording.** "T-RF shows that §9's T3 and T5 specifications, as instantiated, are threshold-representable, so they cannot test whether learned models develop reconciliation computation. The learned T3–T5b results confirm this design limit; they do not show that learned models avoid relational reconciliation when it is needed. Obligation 2 (strength-dependent, non-threshold rules) is the test that would."

## 4. B's confidence modulation is an on-manifold, relation-caused, theorem-predicted effect; "only a confidence change" undersells it

**(a)** The package classes B as "a confidence modulator, not a decision switch" (X6, synthesis Q12), in a tone that suggests a lesser result. But:

- §13 explicitly lists "agreement-versus-conflict response differences" as a theorem target;
- X6 uses exactly the X4 criterion: a matched relation-blind pathway with both members real inputs;
- B-TM gives the sign of the routing effect on the total margin from inspectable frozen quantities (`w_B − w̄_{−B}`), and it is correct 3,000 / 3,000.

**(b) Evidence.**

- X6 (`logs/matched.txt`): (1, 1) and (3.5, −0.055) share `m_uniform = +2.391`, but their margins are +5.192 and +2.405.
- (1.5, 0.7) and (3.5, −0.090) share `m_uniform = +2.353`, but their margins are +4.231 and +2.388.
- Matching the uniform-routing pathway, not the raw sum, avoids the LN confound that `MLP_NORMALIZATION_THEOREMS.md` (N1, PA9) identifies.

**Limits that must stay attached.**

- Only 2 matched pairs.
- Both matched partners have near-zero b, so a magnitude-of-b explanation is not fully separated from a sign-of-b one.
- For K = 3, the gate tracks the inner product, not agreement (T-B-IMP, B4 equal-dot-product pair).
- A competing writer at kC = 3 cancels the contrast (B5).

**(c) Proposed wording.** "B (K = 2 instantiation): an on-manifold, relation-caused confidence effect. At equal relation-blind (uniform-routing) margin, agreement roughly doubles the margin (2 matched pairs). The direction is predicted from frozen quantities by B-TM (3,000 / 3,000) within B-TM's class: no competing writer, no rereader, final LN with zero bias. It is not a decision switch in the tested pairs. It is not candidate agreement for K ≥ 3."

## 5. X4 + C2 + C′1* constitute a complete bridge *within the engineered class*; F1 "partial" is right overall but hides that

**(a)** The F1 row is rated partial because there is no pretrained or non-relational-objective evidence. That is correct. But for the engineered C′ the chain is complete and noncircular in the §13 sense:

- the frozen signs (`λ, μ, wv, π > 0`) are inspectable before any response is evaluated;
- the thresholds are derived;
- the regions are certified;
- the on-manifold contrast (X4) uses real inputs only.

§22 items 1, 4 and 7 are each met for C′.

**(b) Evidence:**

- X4: three matched pairs, each with equal `m_blind` and opposite decisions, e.g. +1.890 vs −2.630;
- C2: `q_R` carries 100% of the effect and the values 1.8%;
- C′1*: 0 violations at 40 fresh parameter settings.

**(c) Proposed wording for the F1 row.** "Complete for the engineered C′ class: noncircular in the §13 sense, on-manifold within T-R's reachable set. Partial for learned models (T1 only: an explicitly relational objective, which §9 designates as a positive control). Absent for pretrained models."

## 6. Prospective internal predictions deserve explicit F4 credit, but labelled with their exact status

**(a)** The F4 row's "partial at the internal level" is fair. But it does not list the one prediction that was genuinely made from frozen weights before measurement and checked on held-out seeds: the step-6 sign prediction. This is the kind of prediction §13 asks for.

**(b) Evidence:**

- T1/TF1: 61 / 71 correct on dev seeds and 33 / 35 on held-out seeds (preregistered, analysis code frozen at `c4bc553`);
- T3: at chance, which is a correct *negative* prediction that those units are not the mechanism;
- C′1* thresholds checked on fresh points.

X4 itself was computed, not preregistered, so it is checkable but not shown to be prospective. Do not call X4 "prospective" without a preregistered instance.

**(c) Proposed wording.** "Internal predictions with prospective status: (i) the preregistered frozen-weight sign prediction (T1/TF1: 94% on held-out seeds); (ii) C′1* regions validated on fresh parameters. X4- and C2-type contrasts are checkable predictions but were not preregistered."

## 7. The T3 unit-criterion failure is a §22 positive item, not just a caveat

**(a)** The package lists it fifth among its negatives. Mandate §22 explicitly counts "a strong counterexample showing why a proposed identification procedure cannot work" as real progress.

**(b) Evidence.**

- The T3 units pass `I_agree ≥ 0.2` and `D_sum ≥ 0.2`.
- Ablating them leaves accuracy ≥ 0.998.
- The sign prediction is at chance, and the probe interchange transfers 0%.
- The preregistered pipeline itself caught this, at the ablation and interchange stages.

**(c) Proposed wording.** "Identification-procedure result (§22): activation-level relation criteria, even margin-free and preregistered, select unused units in additively solved tasks (T3). The interventional stages of the same preregistered pipeline correctly reject them. A use-implying criterion needs a generator-only interventional step (Obligation 5)."

## 8. Places where the package should *not* be defended (guarding against inflation)

1. **T1 positives.** §9 designates T1 as a positive learnability control. "Identified, causally used, gauge-robust" is legitimate *as a validation of the identification pipeline*, not as evidence that relational computation emerges. The package mostly respects this, but synthesis Q18.3 should add "positive control".
2. **The "precise placement" claim.** This is standard linear-threshold / XOR theory. The defensible novelty is the application: §9's reconciliation tasks fall on the threshold side. Do not call T-RF new mathematics.
3. **The continuous case is open.** T-RF and the twin are checked only outside dead bands (|a|, |b| ≥ 0.1). Inside the bands, C′'s decision depends on relation and magnitude jointly (`CPRIME_…` line 145). Whether that dependence is threshold-representable is untested (Obligation 3). Neither side should claim it.
4. **The R7 checks are exploratory.** They come from a post hoc reviewer analysis. They should be rerun inside the package (into `src/`) and labelled exploratory before they enter any main claim.

## 9. Strongest defensible restricted positive claims (suggested final list)

1. **C′1*** (proved; certified regions). In conditional rereading with inspectable positive frozen signs, the choice between sources and prior is fixed by `sign G = sign(ab)` together with `|G|` relative to the log value-mass ratio. It holds outside the dead bands, which T-CO proves unavoidable.
2. **X4 + C2.** In the engineered C′, two reachable inputs with identical relation-blind computation receive opposite decisions. The relation's effect is carried 100% by the block-2 query route. This is a complete noncircular bridge within the engineered class and within T-R's reachable set.
3. **Graded-output signature (exploratory, R7).** C′'s margin violates ordinal additivity outside dead bands (strongest reversal: both differences ≥ 2.26 logits; the (±1, ±1) example: +4.52 vs −0.74). So C′ is behaviourally distinguishable from its decision-equivalent relation-free twin at the confidence level. Behaviour establishes a cross-source interaction; it does not establish agreement specificity.
4. **B (K = 2).** An on-manifold, relation-caused confidence modulation whose direction B-TM predicts from frozen quantities (3,000 / 3,000), within B-TM's class.
5. **Learned positive control (T1).** The identification pipeline recovers a causally used relation with frozen-weight sign prediction (94% on held-out seeds) and set-level gauge invariance. Its interventional stages correctly reject activation-only false positives (T3).
6. **Scoped learned negative.** In the tested family, threshold-representable reconciliation tasks are solved without a used relation, and their margins are ordinally additive (exploratory). This is a design finding about §9's task class (T-RF), not evidence against relational computation where the target demands it.
