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

Input-level V-surface fits (`logs/conf/model_*.json`; relative R² = R²_within / split-half ceiling).

| | T1 cohort, 2000 steps (6) | T1 parents at 500 steps (6) | P-count cohort (6) |
|---|---|---|---|
| ceiling (split-half R²) | 0.68–0.98 | 0.19 (cT67), 0.89–0.96 | 0.73–0.91 |
| 𝔽_mix relative R² | 0.81–0.97 | 0.67–0.92 (one degenerate ceiling) | 0.38–0.85 |
| exact Bayes (scaled) relative R² | 0.35–0.96 | — | −0.02–0.77 |
| 𝔽_vote relative R² | −0.15–0.12 | — | 0.17–0.36 |
| fitted output scale τ | 0.95–1.04 | — | 0.69–0.79 |

- **No model is 𝔽_vote-representable**; every full-budget T1 model and the better P-count models are largely 𝔽_mix-representable.
  At the frozen ε this classifies the curricula as using the *same factorization* (count → configuration evidence → posterior
  mixture), with P-count models further from it (more out-of-class structure) and with a compressed output scale (τ ≈ 0.75: the
  P-count models are under-confident in their count-dependent evidence, a calibration-like bias inside the class).
- The arm difference in 𝔽_mix fit (P3(b)) is significant at full budget (+0.30, p = 0.004) and not at the quality-matched budget
  (+0.37 but p = 0.056, with one degenerate ratio). It is consistent with "same class, P-count further from it", and its
  persistence under quality matching is undecided.
- **How much of the null responses the class explains.** On π-equivalent pairs within the surface, the fitted 𝔽_mix member
  reproduces only 0–16 % of each model's observed null-response variance. The fitted members' predicted null responses are
  mutually coherent (within T1 0.50, within P-count 0.76), but the observed ones are not (0.20 / 0.18): **most of each model's
  null response lies outside the shared factorization** and is model-specific. The parameter differences inside the class
  (stable approximation bias) are not what produces the observed null-response pattern.

**Exploratory parameter-level difference (post hoc, `scripts/phi_sign_posthoc.py`).** In the fitted class, the reliability
evidence at perfect agreement, s = φ_rel(5) − φ_rel(4), is negative in **every** P-count model (13/13, development + cohort) and
positive in 12 of 13 full-budget T1 models (exception: the failed run cT66, −0.03); cohort-only Fisher p = 0.0014. Exact Bayes
has s = log ρ = +1.36. P-count models treat a group that agreed in all five other questions as *less* reliable than one that agreed
in four — a "perfect agreement is suspicious" heuristic that double-counts the collusion signature the class already represents
through φ_col. It is not explained by approximation quality alone: cT67 (KL 0.121, P-count-like) has s = +0.57, while cP62 and cP64
(KL 0.07) have s < 0. It is, however, unstable in T1 parents transferred for 250/500 steps (signs mixed), so it is a property of
fully transferred T1 models, not of T1 pretraining alone. This is a candidate stable curriculum-dependent approximation bias —
the same algorithm with a qualitatively different parameter — and has not been prospectively tested.

## 5. Answer to §10

Relative to the two declared classes, the evidence supports **the same algorithmic factorization with curriculum-dependent
approximation bias**, plus large model-specific out-of-class errors that carry the null responses. Distinct algorithms are not
supported (no model is better described by 𝔽_vote). By Proposition 2 this is a statement relative to the classes, not an
identification of the internal implementation.
