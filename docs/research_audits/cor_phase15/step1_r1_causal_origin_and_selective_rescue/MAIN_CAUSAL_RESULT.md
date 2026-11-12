# Main causal result: Investigation VII

> **Post-review [R1–R5].** Revised after five independent reviews (REVIEWS_AND_ADJUDICATION.md):
> - "Bayes level" is withdrawn.
> - The corollary is corrected.
> - The rescue is described as a behavioural rescue by a precomputed conjunction.
> - Post-review development controls are added: a generic ingredient adapter and 3× training.
> - The layer-2 route description is corrected.
> - Outcome F is set to undetermined for budget.
>
> Pre-review text: commit 76a9d60.

**Question.** Why do P-count-pretrained Transformers make the wrong candidate update in the R1 perfect-agreement configuration?
Can the error be corrected selectively?

**Answer.** The correct R1 update requires a cross-group interaction.
- **What it needs.** In the exact rule this is the eligibility factor m_g. It counts the other groups that are perfect *and*
  agree at q. We prove that any mixture-form weighting that depends only on each group's own count gives R1, R2 and R3 the same
  sign. Getting the full Bayes pattern therefore needs information about other groups: their counts or their votes.
- **What the P-count states provide.** P-count frozen states present each group's count separately and accessibly, but not the
  cross-group relation. The one-block transfer learner therefore settles on a near-separable perfect-agreement discount: correct
  in R2/R3, wrong in R1.
- **The correction.** Supplying exactly that conjunction, computed from the input, raises the R1 response on all 8 new
  independent lineages (frozen protocol). Mean |E_R1| falls from 0.72 to 0.30 (58 %). Three parents overshoot Bayes, and the raw
  sign is corrected in 5/8. The other decisions and average quality are preserved or improved. Controls:
  - supplying the counts (which contain the ingredients) makes R1 worse (7/8);
  - random features do nothing;
  - data balancing fixes R1 only by damaging R2/R3;
  - post-review development controls: giving all ingredients (counts plus q-agreement) does not help, and tripling the
    training budget leaves the sign wrong.
- **What is not identified.** The internal operation of the original models is identified only as a route: a readout-token
  query, direct and MLP pathway. It is not a unique circuit, and blocking that route does not repair the decision.

## 1. Mathematics (R1_INFORMATION_REQUIREMENTS.md, FORMAL_THEOREMS_AND_LIMITS.md)

- **Proposition 1 (closed form) [P].** p*(y) ∝ Σ_g RHO^{n_g} m_g L_g(y), where m_g counts the *other* groups that are perfect
  and agree at q. The rule is vote-blind and mixture-form.
- **Explaining away, exactly.** At 4→5, h* (the existing perfect group) goes from "certain colluder" (m = 0) to "possibly
  reliable" (m = 1). In R1 both possibilities support c, so the c-supporting mass rises from 0.926 to 0.976. In R3, h*'s
  candidate becomes an equal competitor (0.917 → 0.486).
- **Theorem 1 [P].** For separable weights, sign(R1) = sign(R2) = sign(R3) = sign(w(5) − w(4)). So any rule that gets the
  Bayes signs needs cross-group weights.
- **Theorem 2 [P].** In the eligibility family with v(0) = 0, the R1 sign is decided by one balance:
  sign(w(5)/w(4) − v(2)/(2v(1))).
- **Status of the theorems.** They constrain mixture-form computations, not Transformers in general.

## 2. Discriminating H-REP, H-READ, H-LEARN and H-INT (development; 16 Investigation VI parents)

**H-READ (inherited readout or LN) is not the cause.**
- The P-count inherited W_A is its random initialisation.
- A fresh LN changes nothing.
- A fresh W_A reduces the P-count R1 error by about 0.29, at a cost to KL.
- With a fully fresh downstream stack the curriculum difference persists: P80 −0.60 vs T70 −0.02.

**H-LEARN in its "wrong supervision" form is not the cause.** Exact-Bayes soft targets do not fix R1 under the original
architecture (−0.74).

**H-REP in its information-loss form is not shown.** High-capacity MLP readers on P-count states approach Bayes (P80: Δm_R1 ≈ −0.06) or over-credit (P81: E +0.45).

**What is shown is architecture-limited accessibility of the cross-group relation, interacting with learning (H-INT):**
- **Probes.** "Another group is perfect" is linearly decodable at the q-source tokens of T1 states (0.98–0.997). In P-count
  states it is at or near the majority rate (0.912–0.952); confirmed on new lineages, 8/8 at 0.913–0.932.
- **Not absence.** A class-balanced probe gives AUC 0.73, and per-group "perfect" indicators decode well. What is weakly
  accessible is the *conjunction*.
- **Identical readers.** The same MLP reader gives R1 +0.42 on T1 parent 70's states and −0.29 on P-count parent 80's. On P-count parent 81, which has more accessible states, it gives +0.45. The original one-block learner fails on both P-count parents.
- **Oracle readers.** An MLP given all counts but not m reaches KL 0.002 while still failing R1 at 4000 steps. It needs about
  16,000 steps.

**Routes (exact input-copy factorial).**
- T1 implements the boundary discount through the *keys of the source tokens* (Ks ≈ −1.5).
- P-count implements it through the *readout token's own count-bearing state*, mainly a query head, amplified 3–5× by the MLP.
  That route is nearly regime-blind. Its R1 contribution is negative in 8/8 parents. The magnitude depends on the baseline
  (−0.74 from before, −0.37 from after); it is a hybrid-input quantity.
- Confirmed on new lineages: 8/8 negative (mean −0.535).
- The route is not necessary for the error. Replacing the readout-token input (`_rtemb`) leaves R1 wrong (−0.60 / −0.75).

**Dynamics.**
- The error is absent at step 0 and appears within 250–500 steps.
- It deepens as KL falls, along the readout-token route.
- T1 learns the discount through the source tokens from the start.

## 3. Selective rescue (frozen, independent; INDEPENDENT_CONFIRMATION.md, SELECTIVE_RESCUE.md)

The cohort is 8 new P-count lineages, each trained under the original transfer setup with matched streams. Values are means.

| | E_R1 | \|E\| R2∪R3 | KL | acc |
|---|---|---|---|---|
| original | −0.724 | 1.228 | 0.070 | 0.725 |
| **+ m (repair)** | **−0.029** | **0.514** | **0.039** | **0.731** |
| + n (count control) | −1.361 | 0.746 | 0.034 | 0.733 |
| + random (capacity control) | −0.743 | 1.227 | 0.070 | 0.725 |
| W_bal (generic balancing) | +0.176 | 1.311 | 0.149 | 0.691 |

**Frozen predictions.**

| id | prediction | result | verdict |
|---|---|---|---|
| P1 | m raises R1 | +0.695, sign-flip p = 0.0039, CI [0.48, 0.88], 8/8 | confirmed |
| P2 | specific vs capacity | +0.714, p = 0.0039 | confirmed |
| P3 | selective success ≥ 6/8 | **7/8** | confirmed |
| P4 | n worsens R1 | p = 0.012, 7/8 | confirmed |
| P5 | original reversal replicates | 8/8 | confirmed |
| P6 | low H2 accessibility | 8/8 | confirmed |
| P7 | negative readout-token route | 8/8 | confirmed |

**Limits of the rescue.**
- The repaired R1 sign is positive in 5/8 parents and slightly negative in 3/8.
- One parent (yP96) improved by only +0.15.
- P3 counts overshoot as success; a distance-to-Bayes version would pass 5/8.
- The hypothesis was written after injection had worked on two development parents, but before any confirmation run.

**Post-review development controls** (not frozen):

| control | parent | E_R1 | original |
|---|---|---|---|
| ingredient adapter (counts + q-agreement) | P80 | −2.26 | −0.90 |
| ingredient adapter (counts + q-agreement) | P81 | −0.67 | −0.72 |
| 3× training | P80 | −0.68 | −0.90 |
| injm on T1 | T70 | −0.12 | −0.10 (no damage) |

## 4. Outcomes (mandate §17)

| outcome | status | why |
|---|---|---|
| A: insufficient or inaccessible | **partly, as inaccessibility** | Inaccessible to the original one-block learner and to linear probes. Not information-theoretically insufficient. |
| B: sufficient, but downstream learning fails | **partly** | High-capacity readers recover most of the response. The learner's failure is tied to *what is accessible*, not to supervision. |
| C: inherited readout causal | **not supported** | Modulates the error by about a quarter at most. |
| D: specific operation identified | **not reached** | The trained-model route is identified (readout-token query/direct/MLP). It is sufficient to carry the error but not necessary for it to arise, and it is not a unique circuit. |
| E: selective correction succeeds | **supported, as a behavioural rescue only** | By a hand-specified, input-derived conjunction feature that is close to an oracle on the deciding subset (97 % choice agreement there). It is specific against the controls that were run, on independent lineages, with R2/R3 preserved. It is consistent with Theorem 1 and the accessibility results, but it does **not** establish an internal mis-computation (mandate §9.5). |
| F: generic training explains | **not supported for selectivity; undetermined for budget** | Balancing moves R1 but destroys R2/R3. 3× training leaves the sign wrong (n = 1, development). LoRA, logit-recalibration and learned-adapter baselines were not run. |
| G: cause unidentifiable | **partly**, at the operation level | Several internal implementations (heads, route mixtures) are compatible across parents. |

## 5. Scope and what is not claimed

**Scope.**
- One generator (K = 3, Q = 6, three groups of two, ρ = 1).
- One architecture.
- Frozen-feature transfer. The unfreezing control was not run.
- New lineages from the same pipeline.

**Not claimed.**
- That the pretraining *objective* alone causes the difference: data and step counts also differ.
- That networks compute m explicitly.
- That P-count pretraining damages reasoning in general.

**Relation to earlier investigations.**
- **Investigation III.** Count information at different sites matters causally only through *which relation is accessible
  where*. The readout-token count site is the route P-count models use, and it is the separable one.
- **Investigation IV.** No common downstream operation between the curricula. The routes differ (source keys vs readout-token
  query).
- **Investigation V.** No inference from output-error correlations. All claims here rest on edits, factorial replays and
  training interventions.
- **Investigation VI.** The result stands, and replicates on a third cohort (8/8).
- **Original COR.** The variable that matters here (m_g, the cross-group eligibility) is independently defined from the
  generator, and Theorems 1–2 are candidate-response theorems about it. This is not a vindication of the O/D framework and is
  not relabelled as one.
