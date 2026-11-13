# Cross-mechanism comparison (Workstream F)

Every family studied so far has an exact Bayes rule of the same *Bayesian* form, a mixture over a latent selector:
p(y | X) ∝ Σ_r w_r(X_ctx) · ℓ_r(y), where r is a latent role or gate and w_r is a posterior weight computed from context.
**That form is ordinary Bayesian model averaging. It is not a new theory and is not claimed as one.** The families differ
in three things that *are* informative:

- **S**, the statistic of the context that w depends on;
- **O**, the interaction order across source groups (Theorem 2);
- **D**, whether w factorizes over per-group features (a pooled, deep-sets form) or needs joint pairwise features.

| family | latent selector | statistic S | O | D | learned mechanism (where known) | source |
|---|---|---|---|---|---|---|
| T4 latent reliability (one reliable group, others noise; C = 0) | which group is reliable | per-group counts n_g | **1** (additive over groups: p ∝ Σ_g W_g L_g) | per-group | causally used last-layer routing in some seeds; not recurrent across implementations | atlas continuation; Theorem 2 with C = 0 |
| P-count / R1 eligibility (T4 with a colluder; C = 1) | (reliable, colluder) | per-group counts **and** the cross-group count S of eligible groups | **2** | per-group (pooled: p ∝ S·A − B) | small readers: attention to eligible partners (Inv. VIII). **Native C1: per-group routing with collapse at n = 5, a separable shortcut (this investigation)**. Native fine-tuning: see `CAUSAL_MECHANISM_RECONSTRUCTION.md` | Theorem 1 |
| C colluders | (reliable, colluder set) | counts, S | **1 + C** | per-group, ρ of degree 1 + C | not trained | Theorems 1-C, 2 |
| partial copying (β < 1) | (reliable, colluder) | per-group colluder likelihood ratios (soft) | 2 | per-group | readers trained at β = 1 keep the hard gate and fail (GP4) | Theorem 3 |
| **adaptive echo** (the colluder copies the reliable group with probability γ) | (reliable, colluder) | **cross-group echo counts** (joint statistic of pairs g, h) | 2 | **pairwise, not per-group** | not trained | `logs/adaptive_sources.json` |
| GATE | open/closed gate | a within-question relation between two non-answer sources (x1 = −x0 mod K) | 2 (gate pair × answer group) | per-question multiplexer | matches Bayes; relation localized to evidence tokens, then readout residual; gate-dependent attention is causally unnecessary | atlas continuation |
| TRANS | two latent transformations σ, τ (two-hop composition) | — | not analysed here | — | not learned in a truncated 4000-step run; **no new evidence in Investigation IX** | atlas continuation |

## Findings

**1. Sufficiency changes with source semantics: adaptive sources (F4) [N].**
- *Setting.* The colluding group's common report copies the reliable group's first source with probability γ, and is
  uniform otherwise. γ = 0 is T4; it matches brute-force T4 Bayes to 1.8e-15.
- *Result.* For γ = 0.5 and 0.9, T4's sufficient statistic (within-group counts plus target reports) is **no longer
  sufficient**: the exact posterior varies by up to 0.95 within its cells, over 300,000 decisions. Adding the six ordered
  cross-group echo counts (questions on which h's report equals g's) restores sufficiency (spread ≤ 3.6e-15).
- *Reading.* A dynamic, history-dependent source makes a *between-group* relation decision-relevant. No per-group
  statistic replaces it. This is a genuinely different relational class from R1 eligibility, whose cross-group operation
  reduces to a scalar count over per-group indicators.
- *Restricted claim.* With per-group features only, a pooled sum decomposition can represent the adaptive posterior only
  through a high-rank re-encoding of each group's full report pattern. This is not proved here; it is stated as the
  hypothesis that separates "pooled" from "pairwise" classes.

**2. Interaction order separates the T4 families exactly [P].**
- Latent reliability without a colluder is order 1, so additive pooling suffices.
- The colluder introduces order 2: the R1 phenomenon exists only because of the colluder.
- Each additional colluder adds one order.

This gives a mathematical reason why the P-count R1 boundary is a *different computational class* from the latent
reliability routing studied earlier, although both live in "T4". It is also why a model whose routing implements
reliability weighting per group (order 1) fails exactly where order 2 is needed. The native C1 block (§3 of
`PCOUNT_FAILURE_DIAGNOSIS.md`) is that model.

**3. Routing as causal mechanism: three cases, no common causal abstraction below behaviour.**
- *Latent reliability:* last-layer routing is causally used in some seeds.
- *GATE:* gate-dependent attention is causally unnecessary.
- *P-count C1 (this investigation):* routing is causally used and carries 83–91 % of the R2/R3 responses under the
  attention freeze. But it implements the *wrong* (order-1) computation for R1.

So "attention routing implements the relation" recurs neither as a universal mechanism nor as a universal failure. What
recurs is narrower: in every case, the relation-derived variable is computed early and carried in token residuals. In P-
count, the counts are perfectly decodable from the final pretrained readout tokens of complete parents.

**4. Outcome classification (mandate F5).**
- **Outcome 2, several distinct mathematical classes, is supported:** order 1 (reliability), order 2 pooled (eligibility,
  partial copying), order 2 pairwise (adaptive echo), and per-question multiplexing (GATE).
- **Outcome 1, a common abstraction**, holds only at the level of Bayesian latent-selector mixtures. That is ordinary
  conditional computation, so outcome 3 is also partly true.
- The informative structure is (S, O, D), which predicts which reader architectures can represent the rule. In the
  experiments here, the learned implementations differ even within one class.
- **TRANS remains a separate, unresolved problem.** No TRANS run was executed in Investigation IX, for compute reasons.

## What was not done

- No TRANS continuation.
- No learned model on the adaptive-echo generator.
- No cross-family causal-abstraction maps.
