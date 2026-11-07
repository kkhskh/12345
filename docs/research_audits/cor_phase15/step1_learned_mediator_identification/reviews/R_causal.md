# R_causal: adversarial causal-identification review

Package: `step1_learned_mediator_identification` at a899a7c (protocol frozen at ff180da).
Reviewer scripts and raw outputs are in `reviews/R_causal_work/`:
- `checks.py`: content correlations, readout alignment, manifold normalisers, near-boundary and unmatched interchange, four necessity variants;
- `salt.py`: whether S_alt is decodable;
- `lnfrac.py`: how much of the final state lies in span(U).

The outputs are `checks_*.json` and `*.out`. All checks use the package's own code and seeds, with no change to the package. Every number below comes from these runs or from `logs/`.

## Findings

### F1 (FATAL to the headline as worded). On the matched-pair family, the relation is confounded with the target, so S_rel recovers the model's decision variable, not a relation that is independent of it.

- **Construction of M.** For REL-XNOR, `s = 1[y = +1]`. For REL-MIS, agree gives `y = -sign(a) = -sign(z)` and conflict gives `y = sign(z)`. So **every** matched pair (same z, opposite s) also has opposite targets in both tasks. M is a family of target-contrast pairs at fixed z.
- **Consequence for S_rel.** Its SVD of `h(agree) - h(conflict)` is a target-class difference of means at fixed z. It uses the same information as S_das, the target-trained alignment. The "output-free vs target-using" distinction (MAIN §1.1, §3; DISTINCTIVENESS, "identification advantage") is nominal for these two tasks.
- **Measured: U is the decision direction.**
  - XNOR TF2 at `resid_mid1@R`, seeds s2/s4/s7/s8/s9:
    - corr(U₀, model margin) on natural inputs is 0.984 / 0.978 / 0.996 / 0.897 / 0.995;
    - |cos(U, centred direct readout direction `gf ⊙ (W_U0 − W_U1)`)| is 0.81 / 0.93 / 0.87 / 0.94 / 0.97.
  - MIS, the same pattern:
    - corr with margin is 0.94–0.96 in TF1 and 0.75–0.96 in TF2;
    - |cos| with the readout is 0.89–0.99 (s3 TF2: 0.37);
    - corr with sign(ab) is ≤ 0.05.
  - In MIS, U carries the *target's* interaction and not agreement. That is what one expects if S_rel finds the decision variable, and not what one expects if it finds the relation.
- **What the data support.** One account covers every task: S_rel recovers the decision variable wherever matched pairs differ in target. In XNOR that variable equals sign(ab) only because `y = sign(ab)`. The one family where s and y come apart on M is REL-GATE (y differs on about 50% of pairs). There S_rel never found relation content (0/12).
- **No positive instance.** Nowhere does the package show that a relation which is *not* the target is computed and used. MAIN §4.1 calls XNOR a "learnability control", but §1 still answers "Yes".
- **Correction.**
  - Replace the headline with: "No instance was found of a learned model computing source agreement as an intermediate variable distinct from its target. In XNOR, where the target *is* agreement, the target variable is linearly localised (k = 1–2) before the final MLP and created by the attention normaliser."
  - Remove "identified without any output information" for XNOR and MIS. Write instead: "selected without the model's outputs, but from pairs that are target-contrastive by construction."

### F2 (MAJOR). In TF2 XNOR, the `resid_mid1@R` site is effectively the readout state, which is the trivial site the protocol meant to exclude.

- **The protocol's rule.** PROTOCOL §2 excludes final residual sites because "a readout-aligned direction there transfers decisions trivially".
- **Why this site is the same in practice.** At `resid_mid1@R` in the headline TF2 XNOR models:
  - the MLP1 path carries 0.00 of the effect and the direct path 0.99–1.00 (path split, INTERVENTION_RESULTS);
  - 84–97% of the LN-centred *final* residual norm lies in span(U) (`lnfrac.py`);
  - U is readout-aligned (F1).
- **Consequence.** Interchanging U is then close to interchanging the logit direction. IIA = 1 and the random/S_alt contrasts are therefore expected, and they are not evidence of a specific relation mediator.
- **This also explains S_grad's IIA of 0.000 at these sites** (DISTINCTIVENESS says S_grad "finds what moves the margin locally"). The final LN Jacobian projects out the dominant state direction, and that direction is U. The gradient is therefore close to orthogonal to U (cos_Srel_Sgrad = 0.23 in s2). This is LN geometry, not evidence that S_rel is a better *relation* finder.
- **Correction.**
  - State that the TF2 XNOR mediator is the pre-readout decision coordinate.
  - Report the readout alignment and the direct-path share as part of the result.
  - Drop the "S_rel beats S_grad in 9/22" claim as an identification advantage, or explain it by LN.

### F3 (MAJOR). "Sufficiency" is the same test as reverse-direction interchange, so T-MED's H1 and H2 are not independent.

- **The identity.**
  - `complement_interchange(base=A, src=C)` builds `hC + (hA − hC)UUᵀ` and expects dec(A). That is exactly the state that `interchange(base=C, src=A)` builds, and it expects the same decision.
  - At a complete cut, downstream reads only the site state, so which input the run starts from does not matter.
  - `iia()` already averages over both directions.
- **Hence suff ≡ IIA(S_rel) identically.** The two values are equal in all 150 of the 150 site rows across logs/protocol_E1–E4 (e.g. 0.908/0.908, 0.851/0.851, 0.735/0.735, 0.974/0.974).
- **Effect on the theorem.** The "five causal gates" are really four. T-MED's (H1)+(H2) give no more than (H1).
- **Correction.** Merge the two tests, or define sufficiency non-redundantly: for example, keep U, resample the complement from a *random* natural input (not the partner), and check the decision.

### F4 (MAJOR). S_alt is not "equally decodable", so the decodability-versus-use control has no force at the headline sites.

- **The claim.** PROTOCOL §3, MECHANISMS §4 and DISTINCTIVENESS describe S_alt as "equally decodable".
- **The measurement** (logistic decoding of s on a held-out natural sample, TF2 XNOR `resid_mid1@R`):

  | seed | S_rel | S_alt | full orthogonal complement of U |
  |---|---|---|---|
  | s2 | 0.999 | 0.727 | 0.726 |
  | s7 | 1.000 | 0.517 | 0.517 |
  | s9 | 0.999 | 0.651 | 0.651 |

- **Consequence.** The complement holds little linearly decodable s. S_alt's IIA of 0.000 therefore cannot show that "decodability is not use".
- **Correction.**
  - Report S_alt's decode accuracy next to its IIA.
  - Use the decodability-vs-use claim only where S_alt decodes s about as well as U does, or drop it.

### F5 (MAJOR). Matching on the aggregate does not by itself make S_rel relation-specific.

- **The claim.** MAIN §3 says matching on z "is what makes it identify relation content rather than source content".
- **Why that fails.** At equal `a + b`, the contrast still contains every *nonlinear per-source* term. For example, `φ(a_A) + φ(b_A) − φ(a_C) − φ(b_C) ≠ 0` for saturating φ.
- **Measured.**
  - In TF1 XNOR, U₀ correlates about 0.5 with a and with b individually, 0.09–0.29 with the margin, and 0.10–0.27 with sign(ab).
  - The package's own verdicts agree: S_rel found source-additive content in 5/6 TF1 XNOR models, 11/12 REL-GATE models and K3-GATE.
  - So matched-z S_rel finds "whatever differs at fixed z", and that is often source content (the reviewer's concern (1)).
- **What protects the verdicts.** The content (additive-surrogate) test is what keeps the verdicts honest, and the code does that correctly:
  - `GridCoords.surrogate('additive')` is the two-way ANOVA main-effects projection on a uniform 61×61 grid over the training support, done per p;
  - `surrogate_test` replaces only the U-coordinates and keeps the complement.
- **A limit of the content test.** For U ≈ c·sign(ab) on a symmetric grid, the main effects are about zero. The additive surrogate is then about a constant, so the decision collapses, whatever the downstream computation. The test shows that U carries an interaction. It cannot tell "relation" from "decision", because the margin itself has the same interaction form (F1).
- **Correction.** Credit the aggregate matching together with the content test, not the matching alone.

### F6 (MAJOR). E4 is not an out-of-distribution test of the causal claims.

- **What E4 changes.** In `run_protocol.analyse` with `dist = "gauss"`, only `ev` is drawn from the Gaussian. The fit pairs, evaluation pairs (`matched_pairs` with seed 6102, same uniform z/d law), and bank are all from the training law.
- **Consequence.**
  - Cut, IIA, S_rand, necessity, sufficiency and the pair_flip surrogate in E4 are the E1/E2 tests with a fresh seed (logs/protocol_E4*: identical values to two decimals).
  - Only `acc_ev` and the accuracy arm of the content criterion see the Gaussian.
  - The REL-MIS TF2 drop from 6/6 to 1/6 comes entirely from the surrogate *accuracy* on gauss `ev`. The additive pair_flip is 0.99 on train as well. On matched pairs the additive surrogate already keeps almost every decision, so even the train-distribution "relation-computing" label for TF2 MIS rests on the 0.05-accuracy arm alone.
- **Correction.**
  - Reword MAIN §1 ("held … in 5/6 under a new Gaussian evaluation distribution") to: "the content verdict was re-evaluated with Gaussian accuracy; the causal tests were not re-run under the Gaussian".
  - Mark TF2 MIS as relation-computing only by the accuracy arm.

### F7 (MINOR). The cut check cannot fail.

- **Why.** At `resid_mid0@R` (TF1), `resid_mid1@R` (TF2) and `resid_post0@ALL`, the architecture (`model.forward`) gives the downstream computation no other input. Full-state interchange IIA = 1 is guaranteed up to numerics.
- **So** "complete cuts are checked, not assumed" (PROTOCOL §2) describes a hook-code sanity check, not evidence.
- **Correction.** Say "complete by construction (verified as a code check)".

### F8 (MINOR, partly in the package's favour). The manifold ratio is ill-conditioned and overstates how far off-manifold the patches are, but no on-manifold test exists in the package.

- **Why the ratio is ill-conditioned.**
  - The denominator is the natural nearest-neighbour distance, and natural states collapse onto LN-saturated plateaus. Its median at the TF2 XNOR site is 0.003 against a bank of 6000.
  - Near-boundary pairs (inputs only 2ε|z| apart, ε = 0.02–0.3) still give ratios of 15–230. That only says that U jumps as a step.
- **Better normaliser: patched-NN distance over within-pair distance.**

  | model | value |
  |---|---|
  | TF2 XNOR s2 | 0.25 |
  | TF2 XNOR s4 | 0.07 |
  | TF2 XNOR s7 | 0.12 |
  | TF2 XNOR s8 | 0.007 |
  | TF2 XNOR s9 | 0.07 |
  | MIS TF1 | ≤ 0.03 |
  | MIS TF2 | ≤ 0.10 |

- **On-manifold proxies run by the reviewer:**
  - (a) Near-boundary matched pairs: IIA is 1.000 for every ε in all TF2 XNOR, TF1/TF2 MIS and 5/6 TF1 XNOR models (TF1 s8: 0.89–1.0).
  - (b) Each patched state's nearest *natural* state has the source's relation, and the same model decision, in 100% of cases (TF1 XNOR s8: 98.5–99.5%).
  - (c) Unmatched interchange, U taken from a random natural input with the opposite decision at any z: IIA 0.95–1.00.
- **Caveat.** Proxy (b) is partly implied by U dominating the state norm (F2), so it is weak support.
- **Overall.** Interventions are not grossly off-manifold relative to between-class distances. But the off-manifold combination (complement of x, U of x′) is unrealisable by any input, because U is a deterministic function of the input. Limit 3 in MAIN §4 should be restated with the better normaliser.

### F9 (MINOR). The pair-mean necessity result is not driven by landing on the decision boundary, but it does depend on the ablation choice in 2 models.

- **Margins after pair-mean ablation.** The median |margin| after ablation is 2.5–7.7, against about 10.4 before, for TF2 XNOR s2/s4/s7/s8/s9. The decisions are not knife-edge.
- **Flip-after rate under four ablations** (gate: ≤ 0.2):

  | ablation | s2 | s4 | s7 | s8 | s9 | MIS TF1 s2 |
  |---|---|---|---|---|---|---|
  | pair mean (package) | 0.000 | 0.152 | 0.002 | 0.008 | 0.000 | 0.110 |
  | eval-bank mean | 0.000 | **0.752** | 0.000 | 0.006 | 0.000 | **0.682** |
  | resample U from one same-z natural input (shared) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
  | independent same-z resamples | 0.35 | 0.35 | 0.35 | 0.35 | 0.35 | 0.35 |

  For the independent resamples, the expected rate if U alone decides is 0.35, and the observed rate is 0.35 in every model.
- **Interpretation.** The resample ablation keeps U at natural values. It is the cleanest test, and it passes everywhere. The mean ablations put U at intermediate, unnatural values. There the complement can decide (s4, MIS s2), so the necessity verdict is ablation-dependent in 2 of the 11 models checked.
- **Correction.** Adopt the resample ablation as the primary necessity test, and report the mean ablations as sensitivity analyses.

### F10 (MINOR). Agreement versus spread is a definitional identity, not an experimental limit.

- **Why.** For scalar sources, `ab > 0 ⇔ |a − b| < |a + b|` holds on the whole domain, not just at fixed z. No intervention on any two-scalar family can separate the two. The statement in MAIN §4.2 is correct.
- **Not fatal.** "Uses agreement" and "uses thresholded spread" are the same variable here, extensionally.
- **Mildly misleading.** The content result (R² by sign(ab) ≈ 1, a step at the axes) does exclude *un-thresholded* spread |a − b|: corr(U₀, |a − b|) is about 0.5, and with sign(ab) about 0.99.
- **What would separate them.** Sources whose confidence is not their magnitude, or K ≥ 3 argmax agreement. K3-GATE found no relation mediator.
- **Correction.** Say "inseparable by definition on scalar sources". The real limitation is F1, not this.

### F11 (MINOR). Protocol conformance

- **Code unchanged after the freeze.** `git diff ff180da a899a7c` shows **no** modification to `src/mediator.py`, `src/run_protocol.py`, `src/generators.py` or `src/model.py`. It only adds files (`locus.py`, `explore_followups.py`, tests, scripts, logs, docs).
- **Verdict code matches PROTOCOL §5**, with two notes:
  1. Necessity is coded as `flip_after ≤ 0.2·flip_before`. Since `flip_before ≡ 1` on the model-differing subset, this equals the protocol's ≤ 0.2.
  2. NO_CUT can be reached only when the causal gate fails, which matches the table's precedence.
- **The interactive criterion matches §4.7.** For binary tasks the reference is `exact`. For K = 3 it is `interaction`.
- **Development scripts.** PROTOCOL says they were in `/tmp/...`. Copies exist in `scripts/`, so the reference is merely stale.
- **No post-freeze verdict tampering was found.**

## Verified claims (reproduced or checked by the reviewer)

1. S_rel IIA is 1.000 in TF2 XNOR at `resid_mid1@R`, seeds 2/4/7/8/9, and the pair-mean necessity values match the logs (0.000/0.152/0.002/0.008/0.000).
2. U in TF2 XNOR carries almost purely the sign(ab) interaction:
   - corr(U₀, sign(ab)) is 0.89–0.995;
   - corr(U₀, a − b) is ≤ 0.04, so U is **not** the raw a − b source direction (reviewer's concern (1) is rejected for these models);
   - the additive surrogate destroys decisions (logs).
3. In MIS, U is a non-sign(ab) interaction (corr with sign(ab) ≤ 0.05) and tracks the target's decision.
4. Interventions transfer decisions on near-boundary pairs and between unmatched natural inputs (IIA 0.95–1.0). So U alone sets the decision across z. This is stronger than the package claims, but it is consistent with U being the decision variable (F1).
5. No change to frozen code or verdict logic after ff180da.
6. The aggregate-only controls (0/500 matched-pair decision differences) and the TF1 T-ATT pattern-freeze result were not challenged by these checks.

## Net assessment

- **The engineering is sound.** The code follows the protocol, and the measured numbers reproduce.
- **The causal claim is overstated.** What was identified is the pre-readout *decision* coordinate of models trained to output agreement. "Relation", "output-free" and "independently defined" add no evidence beyond that on a pair family where relation and target are perfectly confounded (F1).
- **Two tests are redundant or weak.** Sufficiency duplicates interchange (F3), and the S_alt control is not equally decodable (F4).
- **Recommended verdict.** RESTRICTED-NEGATIVE for "uses a consistency relation distinct from the target"; POSITIVE for "localised, attention-normaliser-created target variable in XNOR".
- **A decisive experiment that is still missing.** Train on a target where agreement is a non-trivial *intermediate* that is not a deterministic function of the target on M, and is not threshold-representable away. Then test whether matched-z S_rel finds sign(ab) rather than the decision direction.
