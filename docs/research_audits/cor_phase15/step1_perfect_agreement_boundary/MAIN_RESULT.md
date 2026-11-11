# Main result — Investigation VI: the perfect-agreement boundary

**Question.** Does T1 vs P-count pretraining change how models use the 4/5 → 5/5 agreement transition when they update a
candidate decision, beyond exact Bayes and approximation error? If so, what causes it?

**Answer in one paragraph.** Yes, prospectively and in observable decisions. In 16 new, independently initialised parents
(8 + 8), trained after a frozen protocol, there is a regime where exact Bayes says to *raise* a group's candidate when the
group reaches perfect agreement: its vote agrees with the existing perfect group. Every T1 parent raises it (mean Δ +0.19 vs
Bayes +0.23). Every P-count parent *lowers* it (mean −0.51). The Bayes-corrected difference is +0.70 log-ratio units
(p = 7.8e-5, CI [0.59, 0.81]). The difference is:
- specific to the boundary relative to 3→4 in that regime;
- present at every transfer stage, and *growing* while P-count quality improves;
- unchanged at matched KL;
- not produced by global gain or temperature;
- invisible to F0 controls.

Causally, the difference is carried by the change that the 4→5 edit makes at the input of the transfer-trained layer, mostly
within a count-decodable subspace. The computation that differs is **not identified**, and the frozen localisation criterion
fails in one arm. **Outcome A is reached; Outcome B is not.**

## 1. Mathematical evidence (proved or exactly computed)

- **Exact Bayes at the boundary** (EXACT_BAYES_BOUNDARY_THEORY.md, THEOREMS_AND_COUNTEREXAMPLES.md T1).
  - A 4→5 edit creates the hypothesis "g is the colluder".
  - In regimes R2/R3 the Bayes update toward g's vote changes sign: it is a credit below the boundary and a discount
    (≈ −2.1 to −2.4) at it.
  - In R1 it stays a credit (+0.23).
  - Below the boundary, updates are weakly positive. They are exactly 0 in R2 (Lemma B1, proved).
- **Proposition 3 corrected** (PROPOSITION_3_CORRECTION.md).
  - The claim "a hard gate is necessary" is withdrawn. The mandate's counterexample is verified: null response 0.0, F45
    response 0.63.
  - Under stated nondegeneracy (verified, null dimension 1) and gate regularity, null invariance holds iff the gate is hard
    with φ_rel affine on 0–4, **or** the gate is finite with φ_rel and φ_col constant on 0–4.
  - s is unrestricted in both branches.
  - Proposition 1's quantifier is corrected to the finite generator's floor δ_min.
- **Calibration limit** (T4). Pure gain predicts the *opposite* sign of the observed D-R1 difference. No single-family excess
  separates gain from a boundary operation.
- **Representation insufficiency and nonidentifiability** (T2, T3). Decodable counts do not fix the boundary update. s is
  confounded with the gate increment.

## 2. Behavioural evidence (frozen, independent confirmation)

INDEPENDENT_PARENT_REPLICATION.md and OBSERVABLE_BOUNDARY_RESPONSES.md give the full results.

| frozen prediction | result |
|---|---|
| P1(a) pooled B45, T1 > P | confirmed: +0.36, p = .018. Development had predicted power ≈ 0.08, so it was expected to fail |
| P1(b) B45-R1, T1 > P | confirmed: +0.70, p = 7.8e-5; the arms do not overlap |
| P2 specificity, D-R1 = B45-R1 − B34-R1 | confirmed: +0.66, p = 7.8e-5. The pooled analogue is n.s. (p = .09) |
| P3 stages 500 / 1000 / 2000 / 4000 | positive and p ≤ 1e-4 at every stage, including 250 |
| P4 quality | KL-matched: +0.68, p = 7.8e-5, with overlapping KL ranges. ANCOVA arm effect p = 1.5e-4 |
| P5 contexts | R1 confirmed, and the probability-space sign agrees (+0.058). R3 point estimate positive, n.s. (expected not positive) |
| F0 control | max \|B0\| = 0.013 |
| ITT vs successful-only | one P-count parent failed (acc .653). Results are identical in direction and size either way |

**Training dynamics** (TRAINING_STAGE_AND_QUALITY.md). Over transfer, P-count KL falls from 0.44 to 0.09. Its 3→4
deficit closes (−0.46 → −0.07), while its 4→5 deficit deepens (−0.34 → −0.74). T1 converges at both. This pattern
contradicts an approximation-error account.

## 3. Causal evidence (frozen P6, independent episodes)

See CAUSAL_EVIDENCE_UPDATE_MECHANISM.md.

- **Confirmed.**
  - P6b: the site-only patch carries the difference (108 % of it).
  - P6c: the count-subspace-only patch carries it (89 %).
  - P6d: F0 specificity, ratio 0.04.
  - P6e: ablating the subspace change leaves a median 42 % of the response.
- **Not confirmed.**
  - P6a localisation: the P-count median site share is 0.77 < 0.8.
  - Two T1 parents route the information through other positions entirely. This is implementation diversity among parents
    with similar decisions.
- **Not tested.**
  - Cross-curriculum transplant: the residual spaces are not aligned.
  - Q/K/V/O, LayerNorm and MLP decomposition.
  - Collateral effects on baseline decisions.
- **Consequence.** Whether the difference lies in how the frozen pretrained representation encodes "5/5", or in how layer 2
  learns to read it, is unresolved.

## 4. Fitted operators (exploratory)

See FITTED_OPERATOR_IDENTIFIABILITY.md.

- The fitted s ordering T1 > P-count replicates (0.76 vs −0.44). s is weakly identified, and the flat sets overlap across arms.
- The best-fitting Bayes mixture reproduces only half of the P-count boundary observable.
- An "s explains it" reading is rejected in favour of the observables.

## 5. What is and is not claimed

**Claimed.** A prospective, curriculum-associated difference in how perfect agreement is *interpreted* in decisions:
- P-count models apply a perfect-agreement discount where the generator calls for a credit;
- this survives stage, quality, gain, output-space and support controls.

**Not claimed.**
- A distinct neural algorithm.
- A causal effect of the pretraining objective as such. Curricula differ in more than the objective: pretraining data and step counts (6000 vs 3000)
  differ too.
- Generalisation beyond the 4→5 boundary of this generator (K=3, Q=6, three groups of two, ρ=1).
- Any statement about readiness, a paper thesis, or the closure of Step 1.

**Outcome classification.** **A**, with parts of B. B is not reached: the computation that differs is localised but not
identified, collateral preservation is untested, and the frozen localisation criterion failed in one arm.

## 6. Earlier verdicts

All earlier verdicts are preserved.

**Corrections to Investigation V's text:**
- Proposition 3: the hard gate is not necessary.
- Proposition 1: quantifier.

**Investigation V's "s is decision-relevant" now has a prospective, fitted-parameter-free counterpart:**
- the observable R1 sign reversal;
- its fitted s-ordering, which replicates but does not explain the size of the effect.
