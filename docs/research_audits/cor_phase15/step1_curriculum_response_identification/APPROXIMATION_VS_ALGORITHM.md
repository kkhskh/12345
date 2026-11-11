# Approximation bias vs a different algorithm (mandate §10)

## 1. A mathematical distinction relative to an operator class

Let g_M(x_q, v) be the model's centred-logit output as a function of the own-question reports and the count vector, measured
at the input level on the V-surface (all same-world v, contents averaged; NATURAL_INTERVENTIONS.md §5). All V-level evidence
about "the computation from V to the decision" is a statement about g_M on this domain (Investigation IV, C4).

**Operator class.** A parametric family 𝔽 = {f_θ} of maps (x_q, v) ↦ ℝ³ with interpretable structure. Two are used:
- **Bayes mixture** 𝔽_mix: w_θ(g,h) = exp(φ_rel(v_g) + φ_col(v_h)), f_θ(y) = τ log Σ_{(g,h)} w_θ(g,h) P(x_q, y | g,h), centred
  (11 parameters; exact Bayes is φ_rel(n) = n log ρ, φ_col(n) = −∞·[n < 5], τ = 1). It factorises the decision into
  "count → per-group configuration evidence → posterior mixture over latent configurations".
- **Weighted voting** 𝔽_vote: f_θ(y) = Σ_s a_θ(v_{g(s)})[x_{q,s} = y], centred (6 parameters). It factorises the decision into
  "count → per-source vote weight → additive vote". 𝔽_vote does not contain exact Bayes.

**Definition (relative to 𝔽, tolerance ε, ceiling c_M = split-half R² of g_M).**
- M is *𝔽-representable* if R²_within(best f_θ, g_M)/c_M ≥ 1 − ε.
- Two 𝔽-representable models share the *same algorithm up to approximation bias* if their best parameters differ
  (θ_M ≠ θ_{M′}) — a **stable approximation bias** when θ is reproducible across independent runs of one curriculum.
- They implement *different algorithms relative to 𝔽 and 𝔽′* if one is 𝔽-representable and not 𝔽′-representable while the other
  is the reverse, at matched approximation quality.
- A model that is representable in neither is *not characterised* by these classes (no algorithmic claim).

This makes "algorithm" a statement about factorization classes, testable on g_M, and keeps "bias" (parameters within a class)
separate from "different factorization".

## 2. What cannot be distinguished (Proposition 2) [P]

For any finite intervention domain 𝒟 (here at most 91 count vectors × the own-question report patterns) and any two classes
whose closures intersect on 𝒟, there are models in the two classes with identical g on 𝒟: no V-level experiment on 𝒟
separates them. *Proof.* Take any g ∈ cl(𝔽)|_𝒟 ∩ cl(𝔽′)|_𝒟 and realise it by a member of each class (or a limit). ∎
Hence a class verdict is only as sharp as the classes are separated on 𝒟; 𝔽_mix and 𝔽_vote are well separated (the best
𝔽_vote fit to exact Bayes has R²_within ≈ 0.1 on the development surface).

## 3. Counterexamples (constructions) [P]

1. **Correlated output errors without a shared internal algorithm.** Model A computes exact Bayes and adds
   β·(n₀+n₁+n₂)·c(e_ŷ) with an MLP at the readout; model B computes exact Bayes through attention to source tokens and obtains the
   same additive term from the attention-weight entropy (a monotone function of total count in this generator). Their null responses
   are identical (r = 1); their internal operations share nothing beyond the Bayes target.
2. **Uncorrelated output errors with a shared algorithm.** Two members of 𝔽_mix with θ_A = θ* + ξ_A, θ_B = θ* + ξ_B, ξ independent,
   zero-mean: same factorization, null responses uncorrelated in expectation.
3. **Similar internal representations, different downstream computation.** Two models with the same site code E(v) (identical
   encoders) and readouts h_A ∈ 𝔽_mix, h_B ∈ 𝔽_vote.
4. **Distinct implementations with identical causal responses on the tested family.** Any two realisations of the same g on 𝒟
   (Proposition 2), e.g. a softmax-attention implementation of the mixture and an MLP implementation of it.

## 4. Empirical results

(Filled from the confirmatory analysis: per-model operator-class fits at budget 2000 and at the quality-matched budget, fitted
parameters, and the account of null responses by the fitted class.)
