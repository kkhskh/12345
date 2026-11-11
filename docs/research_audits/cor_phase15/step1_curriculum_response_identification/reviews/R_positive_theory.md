# Review R_positive_theory — Investigation V (mandate §13, positive-theory reviewer)

Scope: MAIN_RESULT, FROZEN_PREDICTIONS (frozen block + OUTCOMES), SHARED_AND_NONSHARED_STRUCTURE, APPROXIMATION_VS_ALGORITHM,
INDEPENDENT_PARENT_REPLICATION, BAYES_NULL_RESPONSE_STRUCTURE, DOWNSTREAM_CAUSAL_MECHANISMS; archived arrays in `logs/conf/`.
Checks: `reviews/checks/rv_pos_{1..5}_*.py` → `logs/review_checks/rv_pos_{1..5}_*.out` (all <5 min CPU, <1 GB).
**No frozen verdict is changed.** P1 FAILED, P2(b) "explained by quality", P3(c) FAILED, P4 agreement, P5(a) PASSED all stand.
Everything new below is post hoc [E] unless it only restates a frozen number. Severity: H = changes a headline claim's wording,
M = changes a secondary claim or adds a missing result, L = precision/notation.

## A. Claims that are worded too negatively (or too strongly in the negative direction)

**1 [M] "did not replicate" is right for the verdict, but the headline should say what was and was not excluded.**
Evidence: the cohort CI [−0.047, 0.341] excludes Investigation IV's 0.45 (its upper bound is 0.34), so the *original magnitude* is
refuted. The direction comes back (perm p = 0.039), but the effect is fragile: dropping one model moves Δ from 0.11 (cT62 or cT64
out) to 0.25 (cT67 out) (rv_pos_2); N2 resampled-null Δ = 0.06 (p = 0.17); surface-null Δ = 0.02; probability-space Δ = −0.06.
A Gaussian factor simulation fitted to the observed loadings gives the frozen joint rule power ≈ 0.7 at a true Δ of 0.15. That is
an upper bound, because the simulated bootstrap intervals are narrower than the observed ones. So the failure says little about
effects of about 0.15. Proposed wording (MAIN_RESULT ¶1, §17 Q1): *"The pre-registered replication criterion failed. The cohort
interval excludes Investigation IV's effect size (0.45); a same-direction contrast about one third as large (0.15, perm p = 0.039)
was observed but is neither confirmed nor excluded by a 6 + 6 design, and it does not hold up across families or output spaces."*

**2 [M] A pre-registered secondary test that passed is missing from MAIN_RESULT.** P1b (frozen, secondary): EXCESS_TT = +0.135,
perm p = 0.003 (successful runs); +0.102, p = 0.0013 with all 16 runs (`allruns.json`); at b* = 500 it is +0.038, p = 0.74.
Fully transferred T1 models share a component beyond the single factor common to all models. This positive, pre-registered
result is the best support for "something T1-specific at full budget". It belongs in MAIN_RESULT ¶1 and §17 Q7, with its
dependence on training stage.

## B. Claims that are worded too strongly in the "approximation explains it" direction

**3 [H] "explained by approximation quality" (MAIN ¶2, §17 Q3) overstates what P2 shows.** The verdict P2(b) = "explained by
quality" is frozen and stands. But "approximation quality" as an *explanation* is not identified, for three reasons.
(i) KL and training stage are fully confounded within T1: KL < 0.025 is reached only by full-budget T1 models, and no P-count
model or 500-step T1 model gets there (rv_pos_1 (c)). The T1 coherence could be a function of KL or of stage, and the data cannot
tell which. (ii) The b* arm is matched on its median only. Its spread runs from 0.025 to 0.226, and bcT67_b500 is at the
failed-run level. The three b500 models with KL below every P-count model have coherence 0.26, above P-count's 0.19, so "removes
the coherence entirely" depends on which models are in the subset. (iii) P2(a)'s adjusted CI [−0.44, 0.35] carries no
information, because the KL ranges overlap only through cT67. "No curriculum effect after adjustment" (OUTCOMES row) should read
"uninformative".
What *is* supported, as a positive common result, is check rv_pos_1 (d). Take the null-response direction shared by the five
near-Bayes T1 models, with each model's own parent left out of the consensus. P-count models line up with it as closely as the
500-step T1 models do at similar KL: the lineage coefficient after log KL is +0.02, exact perm p = 0.78 over 12 independent
parents. Alignment rises as KL falls across all 22 non-reference models (r = −0.42 with log KL). Also, a 500-step variant
correlates 0.52 with its own 2000-step continuation and 0.18 with other parents: errors are inherited from the parent and become
shared as training converges.
Proposed wording: *"At matched approximation quality there is no curriculum difference: 500-step T1 transfers are not more
coherent than P-count models, and P-count models align with the full-budget T1 error direction as much as T1 models of similar KL.
The T1 coherence is a property of the late, near-Bayes regime (KL ≲ 0.02), which only full-budget T1 models reach. Whether it is
caused by approximation quality or by training stage is not identifiable here."* Rename §17 Q3's answer from "fully removed by
quality matching" to "not curriculum-specific at matched quality; quality vs stage confounded".

## C. The shared abstraction — stronger *and* more precise than stated

**4 [H] Give the shared factorization an exact statement, and say what follows merely from approximating Bayes.**
𝔽_mix contains exact Bayes. A good fit by every near-Bayes model is therefore largely inherited from approximating Bayes, which the
mandate warns against treating as a common algorithm. What 𝔽_mix contributes beyond that can be stated exactly.
*Proposition (null invariance inside 𝔽_mix) [P, sketch; N-checked].* Let f_θ ∈ 𝔽_mix with a hard collusion gate
(φ_col(n) = −∞ for n < 5). On every π-preserving pair (|H| = 1, colluder h fixed, both non-colluder counts shifted by ±1, x_q
fixed), f_θ(x′) = f_θ(x) if φ_rel is affine on {0,…,4}. The converse holds for generic x_q. τ and φ_rel(5) are unconstrained.
*Sketch.* With the gate hard, only configurations (g, h) with v_h = 5 survive. On |H| = 1 the non-colluder counts are ≤ 4. A null
edit multiplies the weight ratio w(g₁,h)/w(g₂,h) by exp(Δφ(v_{g₁}) − Δφ(v_{g₂})), with Δφ(n) = φ_rel(n+1) − φ_rel(n). This
factor equals 1 for all realisable (v_{g₁}, v_{g₂}) iff Δφ is constant on {0,…,3}. J = P(x_q,·|g,h) is unchanged. ∎
Numerical check (rv_pos_4): on all 9000 π-equivalent surface pairs, every model's fitted θ, made affine on {0..4} with a hard
gate, gives a null response of exactly 0 (mean |Δ| < 10⁻⁴). The in-class null response therefore splits *exactly* into
**curvature of φ_rel on {0..4}** and **gate leakage** (finite φ_col). Corollary: s = φ_rel(5) − φ_rel(4) cannot produce
Bayes-null responses. It acts only on |H| ≥ 2 classes, so it is a decision-relevant parameter.
The shared, non-trivial, parameter-level content across all 12 successful models (rv_pos_4): φ_rel strictly increasing on {0..4}
(12/12); a strong but leaky perfect-agreement collusion gate, max φ_col(n<5) between −1.9 and −6.5 (12/12 ≤ −1.88); a common
reliability slope (mean Δφ(0..3) 1.02 in T1 vs 1.03 in P-count, against exact log ρ = 1.36, so a shared ~25 % under-weighting).
The limit: the fitted 𝔽_mix member captures only 11–41 % of each model's deviation from scaled exact Bayes. Most approximation
error, and most null response, lies in the out-of-class residual r_M.
Proposed wording (SHARED §2/§Formal summary): *"Restricted common abstraction: every successful model's V-dependent decision map
is, up to a model-specific out-of-class residual, a posterior mixture over latent (reliable, colluder) configurations with monotone
reliability evidence on 0–4 agreements, a soft perfect-agreement collusion gate, and a common, sub-Bayesian evidence slope. Within
this class, Bayes-null invariance is exactly the conjunction 'affine reliability evidence on {0..4}' + 'hard gate'. Both curricula
violate it, in different ways (below). The class membership itself follows largely from closeness to Bayes; the parameter-level
regularities are the shared content."*

## D. The strongest curriculum-specific finding — stronger than presented

**5 [H] The s-ordering survives the frozen quality-matched budget. "Unstable in 250/500-step transfers" is incorrect at 500.**
rv_pos_5 (post hoc statistic, exact two-sided perm over 6 + 6):
- Full budget: s ∈ [+0.53, +0.86] in T1 vs [−1.05, −0.11] in P-count, complete separation, p = 0.0022. Split-half refits
  reproduce every s within ±0.06, with no sign change (rv_pos_4).
- **At the frozen b* = 500** (median KL 0.064, *better* than P-count's 0.099): s ∈ [−0.07, +0.36] vs [−1.05, −0.11]. Separation
  is still complete (p = 0.0022), with a narrow margin at the boundary. Only at 250 steps do the ranges overlap.
- Not a KL effect: cT67 (KL 0.121) has s = +0.57; bcT67_b500 (KL 0.226, worse than every P-count model) has +0.09.
- Parametrization: with the gate fixed hard, the full-budget separation holds (p = 0.0022, min T +0.55 vs max P +0.02), and the
  b500 separation is near-complete (p = 0.0043). But two P-count models go to s ≈ 0. **The *sign* claim ("negative in every
  P-count model, 13/13") depends on the parametrization. The *ordering* does not.**
Related: P-count φ_rel is concave on {0..4}, dropping sharply at 3→4 (curvature 0.18–0.57). Full-budget T1 is near-affine
(0.04–0.15). By finding 4, this is the in-class source of Bayes-null responses, so P-count's in-class null responses are 2–5×
larger (rv_pos_4, null|fit − null|affine).
Proposed wording (MAIN ¶7, APPROX §4): *"Post hoc, the strongest curriculum-associated regularity is a parameter of the shared
class: the reliability evidence given to perfect agreement relative to 4/5 agreement (s) is higher in every T1-lineage model than
in every P-count model, at full budget and at the frozen quality-matched budget (exact p = 0.0022 each; robust in ordering, not in
sign, to the gate parametrization; split-half stable). It is a decision-relevant bias (by the Proposition it cannot generate
Bayes-null responses), and it is the best candidate for a prospective test."* Recommend freezing it as the primary prediction of a
follow-up cohort: s_T > s_P at b*, rank statistic.

**6 [H] The route separation is weaker as a *curriculum* finding than presented: it depends on training stage, as the coherence
does.** rv_pos_3 uses sel = R remaining − N remaining with Rq blocked. Per-model episode-bootstrap CIs exclude 0 in 5/6
full-budget T1 models (cT61 small, +0.07), and the post hoc arm contrast is p = 0.0065. *But* at 500 steps the mean sel is −0.05,
the same as P-count, and sel flips sign within the same parent (seed 62: −0.30 → −0.56 → +0.59 at 250/500/2000). In bcT62_b500,
blocking the readout *increases* the null response (N remaining 1.54): there the readout route works against the null response
instead of carrying it. The only stage-stable curriculum difference in this analysis is where the *relevant* response enters:
source tokens in successful T1 lineages, readout in P-count. Failed T1 runs cT60 and cT66 also route R through the readout
(0.71, 1.00). This is the site/representation-location confound that Investigation IV withdrew, so it should not be promoted.
Proposed wording (MAIN ¶6, §17 Q7): *"In 4–5 of 6 fully transferred T1 models a readout-local route carries part of the null
response but little of the relevant response, and blocking it is selective. The separation is absent at 500 steps and its sign
varies within a parent across training, so it is a property of the converged T1 regime, not a stable curriculum pathway."*

**7 [M] The two stage-dependent T1 phenomena should be presented together.** Coherence (P6), P1b excess, and route selectivity
all appear only at full budget. The s-ordering is the only curriculum-associated feature that persists at b*. The most accurate
summary is that the curriculum shows in the shared operator's parameters, not in the null-response signature, which belongs to
the converged regime. MAIN ¶7 currently comes last and is labelled "candidate"; it is the most robust curriculum-specific
finding (still post hoc).

## E. Precision / notation

**8 [L]** NATURAL_INTERVENTIONS lists the generator parameter as "ρ = 1", while π and the exact 𝔽_mix use ρ = E.RHO = 3.88
(log ρ = 1.36). One of them is a different quantity; rename it. **9 [L]** `cohort_quality.json` reports T median KL 0.085 / 0.017
(all 8 parents); the documents report 0.064 / 0.013 (successful parents). b* = 500 either way; say which set the frozen rule
used. **10 [L]** "13/13" and "12/13" in APPROX §4 / MAIN ¶7 should become rank-based statements (finding 5).
**11 [L]** "Relative R²" divides an R² on the two-half mean by a half-vs-half R². Ratios above 1 are then expected, not only
"degenerate"; a Spearman–Brown ceiling would be the consistent choice.

## Bottom line

- *Strongest restricted common abstraction:* a configuration-posterior map with monotone reliability evidence, a soft
  perfect-agreement collusion gate and a common sub-Bayesian evidence slope. Inside it, Bayes-null invariance ⇔ affine evidence on
  {0..4} + hard gate (exact). Most error lies outside the class. At matched quality the null-response error direction is
  curriculum-independent.
- *Strongest curriculum-specific finding:* the ordering of perfect-agreement evidence s (T1 > P-count), complete at full budget and
  at frozen b*, not explained by KL, decision-relevant. It is post hoc and should be the next frozen prediction.
- *Weaker than presented:* the route separation (stage-dependent). *Less negative than presented:* P1 (an effect of about ⅓ size
  is not excluded; P1b passed) and P2 (quality vs stage are not identified, rather than "explained by quality").
