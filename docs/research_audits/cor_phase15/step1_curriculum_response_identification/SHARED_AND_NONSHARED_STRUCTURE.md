# Shared and non-shared structure (mandate §11)

All statements concern the 12 successful cohort models unless marked; [C] frozen-confirmed, [E] exploratory/descriptive, [P] proved.

## Shared by both curricula

1. **The relational statistic.** Both use the generator-defined count vector V (Investigation IV; not re-tested here except through
   the V-surface, which every model follows far above its noise floor).
2. **The response operator's factorization** [E, relative to declared classes]: count → per-configuration evidence → posterior mixture.
   Every full-budget T1 model and the better P-count models are 𝔽_mix-representable (relative R² 0.81–0.97 / up to 0.85); no model
   is 𝔽_vote-representable (≤ 0.36). The shared object is a *restricted candidate-response operator*:
   g_M(x_q, v) ≈ τ_M · c(log Σ_{(g,h)} exp(φ_M,rel(v_g) + φ_M,col(v_h)) P(x_q, · | g, h)), with model-specific (τ, φ).
3. **Decision-relevant responses** [C: P3(a)]: the errors of Bayes-relevant responses are equally (in)coherent in both curricula
   (Δ_R = −0.004, CI [−0.21, 0.18]).
4. **A structured "more agreement, more confidence" bias** in null responses, with the same sign pattern in all models
   (coefficient correlations T–T 0.93, P–P 0.93, T–P 0.87) [E; the frozen ≥ 0.9 criterion for a common component failed narrowly].
5. **The kind of null response** [P]: in both, null responses are exactly increments of approximation error (Corollary 1), and their
   size tracks the model's distance from Bayes (model-level r 0.99 with R-errors).

## Where the curricula diverge

1. **Approximation quality** [C]: successful T1 transfers are an order of magnitude closer to Bayes (median KL 0.013 vs 0.099);
   P-count models are under-scaled (τ ≈ 0.75) and less well described by the shared factorization.
2. **Null-response coherence** [F: P1]: higher within T1 (0.34 vs 0.19) but not replicated under the frozen criterion, absent in
   probability space, and absent when T1 parents are quality-matched (P2(b): 0.09 vs 0.19).
3. **Routes of the null response** [C: P5(a)]: in four of six T1 parents a readout-local route carries 26–56 % of the null response
   but little of the relevant response; blocking it removes 34–65 % of the null response and keeps 87–95 % of the relevant one.
   P-count models use one route for both.
4. **Reliability evidence at perfect agreement** [E, post hoc]: s = φ_rel(5) − φ_rel(4) < 0 in all P-count models, > 0 in 12/13
   full-budget T1 models (cohort Fisher p = 0.0014), not explained by quality among full-budget models, unstable in short T1 transfers.

## Formal summary [P + E]

Write each model's V-dependent output as g_M = τ_M f_{θ_M} + r_M, with f_θ ∈ 𝔽_mix and r_M the out-of-class residual. Then:
- the factorization 𝔽_mix is shared;
- θ_M differs systematically by curriculum in one qualitative feature (sign of s) and in scale τ (exploratory);
- r_M carries most of the Bayesian-null responses (𝔽_mix explains 0–16 % of them) and is model-specific in both curricula;
- T1's apparent null-response coherence is a property of fully transferred, near-Bayes T1 models, consistent with r_M becoming
  small and partly shared as training approaches the posterior, not with a different factorization.

This decomposition is identified only relative to the declared operator classes and intervention domain (APPROXIMATION_VS_ALGORITHM.md,
Proposition 2); it is not an identification of the internal implementation.
