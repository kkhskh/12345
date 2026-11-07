# 22. Final scientific verdict

**Verdict:** `STEP1E_VERIFIED_RELATIONAL_CERTIFICATE_KNOWN_THEORY`

The jointly reachable branch readouts on the invariant line are strictly anti-monotone. Their support maximum is strictly smaller than the sum of the exact marginal maxima, for every positive gain and every forward segment. For `μ = 1/2`, `α = 1`, `s_gain = 1`, `ε = 10^{-5}`, and the zero base, the joint positive branch maximum is exactly `0` and the gap at step `1` is `0.7070997102248`. The baseline is that marginal sum, not a norm ball and not a `1/√ε` Lipschitz constant.

After the scalar signs are verified, the comparison is the elementary support criterion (Theorem S1). The linear model is the joint operator norm (Theorem S4), which is the Euclidean form of a shared-noise affine bound. The certificate is sound on its family and reduces to that known mathematics. It does not decide a token, it does not define an obstruction score, and it does not authorize an experiment.

Code correctness, proof correctness, theorem scope, causal interpretation, novelty, and readiness are separated in the gates and the answers below. The proofs are the case split and the monotonicity argument. The tests check that two block implementations match the scalar functions those proofs name.

## Gates

| Gate | Mark | Evidence |
| --- | --- | --- |
| A. Exact joint set | PASS | `S` is the image of one recomputed intervention. The graph lemma says it is a function of `a` when the value map is `μ I`. |
| B. Product distinguished | PASS | Set B is the marginal box. Set C is the ball ceiling `1.595...`. The win is against Set B. |
| C. Support mathematics | PASS | S1 states attainment. The segment is compact and the readouts are smooth. S4's equality case is nonnegative parallelism. |
| D. Relation not defined as the target | PASS | The relation is the pair of derivative signs. The gap is derived. |
| E. Certificate covers the family | PASS | S3 covers every `t ≥ 0`, both sites. Float samples are checks, not the inclusion. |
| F. Fair baseline | PASS | Same block, segment, and branch objective. The baseline is `max f + max g`. |
| G. Strict advantage | PASS | Proved, not only numerical. The strong-gain magnitude is extra and is labeled numerical. |
| H. Nontriviality | PASS | Endpoint attention `0.707...` and MLP `-0.888...` at unit special gain. The raw response at step `1` is `0.819...`. Saturated bases are reported separately and are small. |
| I. Transformer-like block | PASS | Pre-norm, causal mask, two heads, positive `ε`, tanh MLP, residual, fixed readout, two token sites. The pattern is the labeled zero-score pattern. |
| J. Final normalization | PASS | Note 14. No transfer. At step `0.2`, raw `0.0185...` and normalized `1.375...`. |
| K. Cost stated | PASS | Note 15. Constant-time weight check for this class. No general saving is claimed. No timings. |
| L. Known-theory reduction | PASS | Note 16. S1 and affine dependence; S4 and the joint operator norm. |
| M. Causal reading limited | PASS | Magnitude on a fixed readout. No decision, no obstruction. |
| N. Adversarial review | PASS | Eight same-context passes. No in-hypothesis break. Off-line and negative-gain limits kept. |
| O. Reproducibility | PASS | Recorded only after the extract procedure in the verification log. The mathematical gates do not depend on that log. |

Gate O is about the archive built with this file included. The log records the commands that were actually run.

## Separation of claims

- **Code.** The vectorized block and the loop block agree, and both agree with the scalar reduction, on the tested steps.
- **Proof.** S1–S4 and the graph lemma are proved for the hypotheses in the register. The strong-gain location of the maximum is not proved.
- **Scope.** One line, fixed pattern, rank-one opposing MLP, raw readout. Not a ball, not a moving pattern, not final LayerNorm, not a stack of blocks.
- **Causal interpretation.** The gap is a property of one intervention's image. It is not a preference between decisions.
- **Novelty.** The abstract comparison is known. The block calculation is a sufficient condition for that comparison.
- **Readiness.** Not ready for an empirical test. Phase 2 remains `NO GO`.

## Answers

1. **Exact jointly reachable set.** For one base state and one family `U`, it is the set of pairs `(a(δ), b(δ))` obtained by adding `δ` at one token and recomputing the block. On the structural model with an invertible fixed value map, `b` is a function of `a`, because both increments factor through one LayerNorm row and ignore constant translates.

2. **Difference from the product of marginals.** The product contains pairs whose attention half and MLP half come from different steps. On the special-gain segment the product contains `(f(ρ), 0)`. The image contains `(f(ρ), g(ρ))` with `g(ρ) < 0`, and it does not contain the corner.

3. **Norm-ball outer approximations.** Replacing each marginal by a ball of radius `||a_r||` and `||b_r||` raises the ceiling to `1.595484248969981` at step `1`, above both the joint value `-0.181...` and the marginal sum `0.707...`. That is Set C. It is valid and strictly weaker than Set B. Step 1D's product-ball theorem remains the sharp statement about Set C and is not a description of this block.

4. **When the product is exact for a readout.** Under attainment, the separable support equals the sum of the marginal supports exactly when one feasible pair attains both marginal maxima. On the line this happens when the two readouts are maximized at the same step, as with `s_gain = -1`. It also happens when one path is identically zero.

5. **When the gap is strict.** When the two argmax sets are disjoint. On the line, `f' > 0` and `g' < 0` put them at opposite ends of every nontrivial segment. In the linear model, the gap is strict exactly when the two readout gradients are both nonzero and not nonnegative multiples.

6. **Inference from the frozen computation.** The line signs are inferred from `μ`, `α`, `s_gain`, the axis alignment, and `ε`. They are not inferred for a generic weight. The linear equality case is inferred from the two matrices and `w`.

7. **Certification without enumerating the family.** Theorem S3 certifies the bound `0` for every step by the case split, without a search. The size of the baseline, `f(ρ)`, is one evaluation of `λ`, not a sweep of the joint response. A non-special gain has a proved strict inequality and, if one wants the numerical maximum, a one-dimensional search that does evaluate the joint readout.

8. **Does the composition restrict the joint increments?** Yes, in two nested ways. The graph lemma restricts the pair to a graph over `a` whenever the fixed value map is invertible. Opposite monotonicity restricts that graph further so that it misses the support corner. The first restriction does not imply the second.

9. **Strongest restriction established.** Theorem S3: on the special-gain line, the positive branch sum is at most `0`, while the attention marginal is `f(ρ) > 0` for `ρ > 0`. Theorem S2 is the same strict comparison for every positive gain, without the value `0`.

10. **Is it nontrivial?** The paths are not zero. At the special-gain endpoint the attention readout is `0.7070997102248` and the MLP readout is `-0.888384538745181`. The raw response is `0.818715171479619`. The joint branch maximum being `0` is the cancellation of the positive corner, not a vanishing branch. At a saturated base the gap is real and tiny; that base is identified separately.

11. **Certified finite bound.** Yes. `f + g ≤ 0` on the special-gain ray, and `R(t) ≤ ρ` for the read-token raw response. Both cover the whole segment.

12. **Stronger than a valid separate certificate.** Yes, stronger than the exact marginal sum, which is itself a valid upper bound. Not claimed to be stronger than the exact evaluation of the scalar function, and not compared with the loose Lipschitz bound as the headline.

13. **Same information and budget.** The marginal sum and the joint bound use the same weights, the same segment, and the same objective. The joint bound `0` is a constant after the weight check. The baseline's number needs one `λ(ρ)`. Neither needs the joint response. A grid search for a non-special maximum does use the joint response and is not part of the certificate.

14. **Source of the improvement.** The maximizing step of the attention readout is the minimizing step of the MLP readout. Adding the marginal maxima counts a corner the block does not reach.

15. **Nearly zero branch.** Not required. The strict theorem allows any positive gains. The order-one numbers are above. The saturated-base gap is a small response, not a small weight, and is not the headline.

16. **Direct-readout orthogonality.** Not required. At the read token `w · n = 1`, so the skip is the step itself. The branch gap is proved before the skip is added. An orthogonal readout kills both sides because both increments are parallel to `n`.

17. **Substantial responses.** Yes, on the zero base. The strong-gain joint branch maximum on the grid is about `1.052`, with marginals `2.828` and `-1.777`. That joint value is not a near-zero output. Its identification with the endpoint is numerical; the strict inequality is not.

18. **More than one direction.** The line has two orientations, and the derivative signs cover both. A second spatial direction is not covered. `(1, 1.2)` produces branch sum `0.181627925474685` and is a counterexample to dropping the line hypothesis. Constant shifts give a zero comparison, not a second positive example.

19. **Final normalization.** The raw theorems do not survive. At step `0.2` the raw response is `0.018548133077188` and the normalized response is `1.374816323880916`. At step `0.01` the normalized response is more negative than the raw response.

20. **Multiple heads.** The structural block uses two heads, each carrying half of the value map. Zeroing one output block halves the attention increment. The theorem is not a one-head calculation with an idle second head.

21. **Earlier token.** Included in S2 and S3. At step `1` the raw response equals the branch sum `-0.181279889763888`, because the skip does not land on the read token. The attention marginal is again `0.7070997102248`.

22. **Representation dependence.** The objective is one linear readout. Replacing `w` by the normalized all-ones vector makes both branch contributions zero on the line. The gap is not invariant under a change of readout.

23. **Arbitrary metrics.** The certificate bounds `w · (a + b)`, not `||a|| + ||b||`. Those norms sum to `1.595...` at the special-gain endpoint, while the readout sum is `-0.181...`. An `ℓ1` or Euclidean certificate for the increments is a different claim and was not proved.

24. **Identification.** From the frozen weights, for the sign and the special gain, together with the base coordinates that place the state on the line at `0`. Those coordinates are the undisturbed activation. They are not target edits. The same check does not identify a gap for a generic MLP.

25. **Target outcomes.** Not used by S2's inequality or by S3's bound. Used by a grid that locates the strong-gain maximum, and that use is labeled. The constant-fiber and reflection examples are evaluations used as counterexamples, not as inputs to the bound.

26. **Joint operator norm.** Theorem S4 is exactly that reduction, and it is not a finite bound for the nonlinear block. The finite line result is not an operator norm: the derivative at `0` has the wrong sign for `R(1)`.

27. **Neural-network verification.** The linear gap is the dependence idea in affine arithmetic and in zonotope certification, with a Euclidean rather than an `ℓ_∞` parameter. The finite line result was not obtained from a general verifier. It is exact analysis of a one-dimensional restriction. No new enclosure algorithm is claimed.

28. **Causal abstraction.** No high-level model, no interchange, no faithfulness score. The graph lemma is a statement about two increments of one block, not an abstraction theorem.

29. **What remains distinctive.** Only the sufficient condition that this pre-norm line produces opposite scalar slopes, and the explicit width-3 example in which the same attention increment pairs with two MLP increments. Neither is a general method.

30. **Strongest counterexample.** The edit `(1, 1.2)` on the same special-gain block: the branch sum is positive, so the bound `0` is false as soon as the edit leaves the line. The negative-gain corner is the strongest conceptual counterexample, because it keeps a non-product joint set and loses the gap.

31. **Unexplored classes.** Moving patterns as a theorem rather than one negative draw; multi-direction families; generic MLPs beyond the graph lemma; final LayerNorm as an objective; more than one block; trained weights.

32. **Decision preference or magnitude.** Magnitude only. The same weights produce a negative raw response at step `0.01` and a positive one at step `1`. There is no candidate and no preference.

33. **Original obstruction language.** Not justified. The result does not mention, and does not support, `O_c` or `D_c`. The Phase 1 status stays `COR_PHASE1_CONDITIONAL_THEORY_REPAIR_REQUIRED`, and `OBSTRUCTION_READY` stays `NO`.

34. **Empirical test.** Not ready. There is no pre-registered quantity that an experiment could confirm without first choosing a decision the theory does not define. Phase 2 stays `NO GO`. Step 2 of Phase 1.5 is not opened.

35. **Strongest conclusion.** A fixed-pattern, rank-one, pre-norm block has an invariant line on which attention and MLP readouts of one intervention are strictly anti-monotone, so the product of their marginal supports is a strictly loose finite-response bound. The proof is elementary once the slopes are known, and the linear analogue is the joint operator norm. That is a verified relational certificate inside known convex and verification mathematics. It is not a causal decision theory.
