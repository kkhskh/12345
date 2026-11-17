# Review A: mathematics of COR Investigation XV (adversarial agent review)

**Scope.** This review covers:
- RELATIONAL_COMPLEXITY_AND_VALUE_MATHEMATICS.md, which this review calls "MATH";
- TASK_VARIANT_EXPERIMENTAL_DESIGN.md, called "DESIGN";
- `src/xvtask.py` and `src/xvdiag.py`;
- `scripts/xv_math*.py` and `scripts/xv_noisy_math.py`;
- the logs `xv_math_dev.json`, `xv_math_verified.json`, `xv_noisy_math.json`, `xv_math_grid2.json` and `refs_*.npz`;
- the claims about mathematics in OPTIMIZATION_AND_MECHANISM_ANALYSIS.md, called "OPT", and LEARNING_TRANSITIONS_AND_FROZEN_CONFIRMATION.md, called "LEARN".

This is an agent review, not human peer review. No package file was modified, and no model was trained.

**Reviewer computations.** The scratch scripts are in `/tmp/revA/c1.py` to `c9.py` and are not part of the package. Each is an exact enumeration over the 729 inputs, or a sampler check.
- c1: an independent re-derivation of the joint from the verbal generator; a χ² test of the sampler; a check of the stream pairing.
- c2: G for random order-k rules.
- c3, c4, c9: A_p and G for constructed rules on the frozen evaluation stream (98_700_000, 1500 episodes).
- c5: generators with separate reliabilities for the pair and the single source.
- c6: the noisy-gate Bayes rule and an order-5 fit.
- c7: gains of restricted pairwise classes.
- c8: a gate-source law with the last gate source drawn uniformly.

## Verified without objection (summary)

- **The generator matches the documentation.**
  - My independent joint matches `T.joint` to 2·10⁻¹⁹ in P(x) and 6·10⁻¹⁷ in P(y | x), for all four conditions.
  - The sampler passes a χ² test on 6·10⁶ questions, both on x (p = 0.21–0.56) and on (x, y) (p = 0.18–0.37). No zero-probability cell was ever hit.
  - The P(g) = ½ construction, the irrelevant x₂ for a = 2, the ν noise-pair law and single-source uniformity are all implemented as stated.
- **The reference caches are correct.** `refs_*.npz` (px, bayes, gatefree) equals a fresh computation exactly.
- **Proposition 1 (minimal gate arity) is correct.**
  - No proper subset of the gate sources carries information about g.
  - The MI values are reproduced. There is no pairwise shortcut to g for a = 3.
  - Gate pairs are uniform in law for a = 3.
- **The upper bound of Proposition 2 is correct.** The normaliser does not depend on y, so the decision order is ≤ a + 1. The order-(a + 1) optimum reaches KL ≈ 10⁻¹⁵.
- **The lower bound of Proposition 2 is correct.**
  - Because the fit is convex and converged, the order-a optimum's KL > 0 (0.138–0.277) already proves the order exceeds a.
  - See F5 for the G = 0 argument.

## Findings

### F1 (major): the P3 "stage present" call in C and D is close to entailed by censoring, so P3 discriminates between hypotheses only in B

**Evidence.**
- At a = 3 the order-a optimum *is* the gate-free mixture: order-3 evidence features are complete on (x₃, x₄, x₅). So its exact reference values are A_p = 1.000 and G = 0 (c3).
- Suppose a model never acquires the gate but converges on the evidence. That is exactly what LEARN reports for the 6 censored models: final KL equals the gate gain.
- Such a model must therefore satisfy A_p ≈ 1 ≥ 0.5 and G ≈ 0 ≤ 0.1 at two consecutive snapshots. And a censored t_G counts as "stage present" under §4.
- So for C and D, "stage present" follows mechanically from "gate not acquired within budget, evidence learned". It is not independent evidence of a staged learning order.
- The three patterns differ only in B (RACE −, H1 +) and in C (H2 −). The C cell is therefore, in effect, a test of gate *timing*.
- The B cell is the threshold-fragile one: A_p peaks at 0.53–0.67, and RACE is recovered at a threshold of 0.6.

**Correction.**
- OPT §3 and LEARN §2 should state that the 6 censored "stage present" calls are implied by censoring.
- Evidence for H1's ordering claim at arity 3 rests on the 2 competent models.
- "Stage 16/16" should be reported as "8/8 at arity 2 (B fragile), plus 2/2 competent at arity 3; the 6 censored models are uninformative for the ordering".

### F2 (moderate): A_p does not identify the *pair-agreement* trust cue [x₃ = x₄]

**Evidence** (exact fits, c3 and c7). Restricted order-2 classes on the frozen stream give:

| class | ν = 1/3: gain over order 1 | ν = 1/3: A_p | ν = 0.05: gain over order 1 | ν = 0.05: A_p |
|---|---|---|---|---|
| singles + (3,4) only | 0.0178 | 0.76 | 0.0730 | 0.99 |
| singles + (3,5), (4,5) only: **no (3,4) term** | **0.0179** | **0.70** | 0.0409 | 0.77 |
| singles + (4,5) only | 0.0087 | 0.00 | 0.0150 | 0.00 |
| all evidence pairs (the order-2 optimum) | 0.0245 | 1.02 | 0.0757 | 1.12 |

- In B and C, a rule that never looks at x₃ = x₄ reaches A_p = 0.70 and has the same value as the pure pair-agreement rule.
- So A_p ≥ 0.5 certifies "a gate-free pairwise-evidence interaction in the direction of the mixture". That could be consensus between x₅ and x₃ or x₄. It does not certify the pair-agreement cue.
- The claims in MATH §4 ("It can be seen through A_p"), OPT §3 ("then the pairwise-agreement trust cue") and LEARN §3 are therefore stronger than the diagnostic supports. This matters most in B.

**Correction.**
- Rename A_p to a "gate-free interaction index".
- Alternatively, add an edit that separates the routes. For example, resample x₄ so that only [x₃ = x₄] changes while x₅ and the multiset of cross-pair coincidences are kept, then regress on the (3,4)-only and cross-pair-only references.
- In the meantime, describe the stage as "gate-free evidence interactions", not specifically "pairwise agreement".

### F3 (moderate): A_p and the magnitude of G depend on calibration

**Evidence** (c9, c3).
- A gate-free rule at logit temperature λ has A_p = λ exactly: 0.30, 0.50, 0.70, 1.30 for λ = 0.3, 0.5, 0.7, 1.3.
- The order-a optimum behaves the same way.
- Bayes × λ has G = λ: 0.5 and 2.0 for λ = 0.5 and 2.0.
- In B, the rule 0.7 × gate-free has A_p = 0.70 with KL 0.305. That is *worse* than the order-1 optimum (0.302), yet this rule would be scored as being at the intermediate stage.
- LEARN §3.14 says "Not calibration". That is right only for the statement "no gate use ⇒ G = 0".
- B's weak A_p peaks (0.53–0.67) cannot be told apart from an under-confident but complete gate-free rule. Mid-transition values of t_G can likewise move with calibration.

**Correction.**
- Qualify LEARN §3.14 accordingly.
- For sensitivity, report A_p after a per-snapshot temperature fit, or a scale-free variant (for example, the A_p coefficient divided by the coefficient of the model's own fitted response), particularly for B.

### F4 (moderate): the rationale for the usefulness confound is wrong, and "gate gain is 2× larger" depends on the reference point

**Evidence.**
- DESIGN §1 says the two gains move together "because their sum is fixed by β". **That is false.** The sum is Δ_gate + Δ_int = L*(order 1) − L(Bayes), which is 0.3017 at ν = 1/3 and 0.2155 at ν = 0.05. The order-1 NLL depends on ν (0.590 against 0.504), because an additive rule partly exploits agreement.
- The identity in MATH §3, labelled "[proof]", holds by definition for any β. It is not a constraint.
- "With β fixed, no knob changes one gain without changing the other" is true only because ν is the generator's only remaining parameter.
- **Counterexample** (c5): use separate reliabilities, β₁ for the pair and β₀ for the single source.
  - (β₁, β₀, ν) = (0.80, 0.60, 0.15) gives Δ_gate 0.128 and Δ_int 0.0281.
  - (0.95, 0.60, 1/3) gives Δ_gate 0.211 and Δ_int 0.0289.
  - That is a gate-gain ratio of 1.65 at an intermediate gain matched within 3%.
  - (0.98, 0.70, 0.15) gives 0.141 / 0.0705, against A's 0.140 / 0.076.
  - These points change the Bayes NLL, which is a different confound, but they show that a design separating the two gains exists.
- **The size of the usefulness contrast depends on the reference:**
  - Δ_gate ratio B/A = 1.98;
  - full gate value from the order-1 state, (L₁ − L_B), ratio 1.40;
  - accuracy-gain ratio 3.9.

**Correction.**
- Delete "because their sum is fixed by β".
- Reword MATH §3's confound statement as an observation (both move), not a consequence of a fixed sum.
- State the dependence on the reference point.
- Mention the two-reliability family as the way to remove the confound.

### F5 (minor): the lower-bound and G = 0 argument as summarised in MATH §2.2 is incomplete, but it does extend

**Evidence.**
- For a feature on exactly the a gate sources, MATH justifies the moment by saying "g is independent of Y". That is insufficient.
- The data moment is P(x_G)/3. The model moment is P(x_G)·E[Q*(y | x_E) | g].
- Equality needs E[Q* | g] = 1/3. That needs the relabelling-equivariance plus uniqueness step of XIV Proposition 3 (as corrected there).
- That step does carry over. The XV evidence law given g, *including the noise-pair law*, is invariant under every common permutation of the values of Y and the evidence. The optimum of the centred-logit problem is unique on the full support.
- The cases "omits a gate source" (including mixed gate-and-evidence subsets, and x₂ for a = 2) are correct. Independence of (g, Y, x_E) holds jointly, because x_S ⫫ g and (Y, x_E) depends on x only through g.

**Correction.** Add the equivariance and uniqueness step explicitly. Note that the order lower bound itself needs only KL(order-a optimum) > 0.

### F6 (minor): "every rule of order ≤ a gives G = 0 exactly" (DESIGN §2) is true in the population, but not as cited, and not exact on the stream

**Evidence.**
- MATH §5 cites Proposition 2, which covers only the *optimum*.
- **Population check** (c2). With p(x)-weighted, on-support flips, random order-≤ a tables give |G| ≤ 7·10⁻¹⁷ in all four conditions, while order-(a + 1) tables give up to 0.22. G is linear in the logits, so a random direction suffices almost surely. The stronger claim is therefore true [num].
- **Frozen stream** (c4). On the 9000-question stream, order-≤ a rules perturbed from the optima give |G| ≤ 0.01. That is small relative to the 0.1 and 0.5 thresholds, but not exactly 0.

**Correction.** Change the wording to "G = 0 in the population for every order-≤ a rule (verified numerically); |G| ≲ 0.01 on the evaluation stream".

### F7 (minor to moderate): the §7 counterexample about the gate-source law is wrong

**Evidence.**
- MATH §7 claims that drawing the last gate source uniformly, so that P(g = 1) = 1/3, would break value matching across arity. It does not.
- P(g) = 1/3 then holds for *both* arities, and the argument of Proposition 3 applies unchanged.
- Exact values (c8):
  - Bayes 0.3235 in both;
  - Δ_gate 0.2555 against 0.2541 (ν = 1/3), and 0.1266 against 0.1249 (ν = 0.05).
- The residual 0.5–1.3% is the same order-a capacity effect as in the main design.

**Correction.** Replace the justification. P(g) = ½ balances the two regimes and keeps comparability with XIV. It is not needed for arity matching.

### F8 (minor): wording on "exact" matching, and the "best rule without the gate" label

**Evidence.**
- DESIGN says B and C are "matched exactly" and also "differ by 0.09%".
- Matching is exact (0.2770182 in both arities) only under the gate-free-mixture definition. Under the frozen order-a definition it is 0.09% (ν = 1/3) and 0.9% (ν = 0.05).
- For a = 2 the order-2 optimum is not "the best rule without the gate" (0.56531 against 0.56505). The comment in `xv_math.py`, "equals the gate-free optimum", is false for a = 2.
- The intermediate class is asymmetric: for a = 3 it includes the order-3 evidence interaction. This is disclosed, and its effect is ≤ 0.0013 nats.

**Correction.** State that matching is exact under the gate-free-mixture definition and approximate under the order-a definition. Fix the code comment.

### F9 (minor): the sampler check is weak and is misreported

**Evidence.**
- MATH §1 says 1.7·10⁻⁴. The log shows 3.2·10⁻⁴ and 3.9·10⁻⁴.
- The pass tolerance of 0.002 is 40% of the largest cell probability (0.00503) and 13× the smallest (0.00015). It could not detect a cell-level error.
- The conclusion is nonetheless correct: see my χ² tests above.

**Correction.** Fix the reported number, and replace the tolerance test with a χ² test or a per-cell z-test.

### F10 (minor to moderate): relation arity and decision order are conflated in the interpretation

**Evidence.**
- At a = 2 the gate relation g = [x₁ ≡ −x₀] is a pairwise, equality-type relation. That is the same relational arity as agreement, [x₃ = x₄].
- What differs is decision order: 3 against 2.
- The arity contrast moves relation arity (2 → 3) and decision order (3 → 4) together, so the design cannot attribute the delay to either one alone.
- OPT §3 says "lower-order relations are learned first". At arity 2 this is true only in the decision-order sense.

**Correction.** Say "lower decision-order rules first", and state that relation arity and decision order are confounded in the P1 contrast.

### F11 (moderate): the frozen predictions need auxiliary assumptions, and OPT §3 overstates H1's status

**Evidence.**
- **H1, P2 "no effect".** This does not follow from a complexity-*first ordering* hypothesis. Value could still affect timing, so "no effect" is an extra assumption.
- **The frozen outcome for P2** was INCONCLUSIVE, not NO_EFFECT. All 4 arity-2 pairs have ratios of 1.4–2.6, above the ln 1.25 no-effect band, which is the opposite of H1's stated P2 pattern at arity 2.
- So OPT §3's "consistent with all frozen tests" is incorrect for H1 as tabulated in DESIGN §3.
- **H2, P3 (+, −, −, +).** This follows only with an unstated threshold on Δ_int/Δ_gate somewhere in (0.089, 0.54):
  - A 0.54, B 0.088, C 0.089, D 0.56;
  - XIV's GATE 0.124.
- The LEARN §3.9 argument survives (B has a lower ratio than XIV yet shows a stage), subject to F1 to F3.
- RACE is derived from development data, not from the mathematics (disclosed).

**Correction.**
- Mark H1's P2 component as "not supported (inconclusive; the arity-2 subset contradicts it, confounded)".
- State the auxiliary assumptions, including H2's threshold, in DESIGN §3.

### F12 (minor): the noisy gate (Workstream F)

**Evidence.**
- The values are reproduced, and an independent Bayes rule matches to 2·10⁻¹⁶.
- The docstring says k = 1..5, but the code fits k = 1..4.
- I fitted order 5: KL ≈ 10⁻¹⁴. The minimal order is therefore exactly 5, which is all the relevant sources.
- The order-1 and order-2 NLLs are unchanged by γ, because the gate-free law is unchanged. That is correct and worth stating.
- OPT §4's "both complexity and value change at once" is right only formally. Order 3 is within 0.0013–0.0021 nats, so the *practical* decision order is unchanged. The manipulation is primarily a value manipulation: the gate gain falls from 0.277 to 0.054.

**Correction.** Fix the docstring. Rephrase as "formally full order; practically order 3; mainly lowers value".

### F13 (minor): seed pairing (LEARN §1)

**Evidence** (c1).
- Seed-paired arity-2 and arity-3 streams share Y, g, x₀ and all evidence exactly. x₁ and x₂ differ: they agree in only about 1/3 of questions.
- The law of the input positions (x₀, x₁, x₂) differs as well. Under a = 2 the non-uniformity is pairwise; under a = 3 it is three-way.
- That is part of the arity manipulation, so "caused by the gate's arity" stands.
- But "paired models behave the same before the gate" is an empirical observation. It is not implied by the shared stream, because the networks receive different tokens.

**Correction.** Label that sentence as empirical.

## Diagnostics: degeneracy check (no defect found beyond F2 and F3)

- **The A_p regression is well conditioned.** Over the masked swaps, corr(dG, dB) = 0.15 / 0.34 and corr(dG, dO) = 0.62 / 0.16, for ν = 1/3 and 0.05. The condition number is 6.4 and 7.2. Of |dG|², 60% (ν = 1/3) and 86% (ν = 0.05) is not explained by (dB, dO). B is the noisier case.
- **Order-1 rules give A_p = 0.** Any label-shift-symmetric order-1 rule, at any per-source weights or scale, gives A_p = 0 exactly: its swap response is one-dimensional and proportional to dO. Random asymmetric order-1 tables give |A_p| ≤ 0.07.
- **The diagnostics are linear along the stage path.**
  - Logit interpolations order-1 → gate-free give A_p = t and G = 0.
  - Gate-free → Bayes gives A_p = 1 − t and G = t.
  - Order-1 → Bayes gives A_p = 0 and G = t.
  - So A_p separates gate-free interaction use from gate use and from order-1 rules, apart from F2 and F3.
- **The reference values are reproduced.** On the frozen stream, order-a A_p = 1.016 (B), 1.117 (A), 1.000 (C, D).
- **Most of the reference table is tautological.** The order-1, gate-free and Bayes values (0, 1, 0) are exact by construction, because those rules are the regressors. Only the order-a row is informative.

## Surviving claims

1. The generator definitions match the code, including the P(g) = ½ construction, the irrelevant x₂, the ν law, and single-source uniformity for every ν.
2. Proposition 1: a minimal gate arity of a, with no pairwise shortcut for a = 3.
3. Proposition 2: the decision order is exactly a + 1 (3 and 4). The order-a optimum ignores the gate (G = 0). With the F5 step added, this extends XIV's argument.
4. In the population every order-≤ a rule has G = 0; on the stream |G| ≲ 0.01 (F6). Bayes has G = 1.
5. Bayes NLL, the gate-free mixture NLL and the full gate value from order 1 are identical across arity. The gate gain is matched to within 0.09% and 0.9% under the frozen definition, and exactly under the gate-free-mixture definition.
6. Value matching makes the arity contrast a valid complexity contrast at equal value, subject to F10 (relation arity and decision order move together).
7. The usefulness contrast is confounded with intermediate-cue strength (disclosed). The stated rationale is wrong (F4).
8. The order-1, order-a and gate-free optima have equal accuracy, so accuracy cannot reveal the intermediate stage.
9. No low-order shortcut exists in any variant. x₂ is independent of everything for a = 2. No rule of order ≤ a gains anything from the gate sources.
10. The noisy-gate values are correct: Bayes is a mixture, the exact minimal order is 5, order 3 comes within 0.002 nats, and the gate gain falls from 0.277 to 0.054.
11. A_p is well conditioned and separates the three reference behaviours along the stage path. It is not specific to the pair-agreement cue (F2) and not calibration-invariant (F3).
12. The H2 stage pattern and the RACE pattern are falsified as frozen. H1's ordering claim survives, but at arity 3 it rests on 2 competent models (F1), and its P2 component is not supported (F11).
