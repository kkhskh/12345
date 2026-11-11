# R_formal: independent adversarial formal review (mandate V §13, "Formal reviewer")

Scope: FORMAL_DECISION_EQUIVALENCE.md (FDE), APPROXIMATION_VS_ALGORITHM.md (AVA), SHARED_AND_NONSHARED_STRUCTURE.md (SNS),
NATURAL_INTERVENTIONS.md §1 (NI), MAIN_RESULT.md (MR), src/cri.py; generator gen_family.Fam() T4 (K=3, Q=6, α=0.8, ρ=1, pc=1),
helpers eqv.py. Checks (read-only, about 3 min of CPU in total) are in `reviews/checks/rv_formal_{classes,bounds,klfloor}.py`, with
outputs in `logs/review_checks/rv_formal_{classes,bounds,klfloor}.out`.

## Verified correct

- Theorem 1 (corrected C3) and its temperature remark: the algebra is right. Corollary 1 holds exactly (by the definition of e; F5).
- Lemma 1 (posterior ⇔ log-odds; calibrated ⇔ log-odds up to a positive per-model scale; classification is strictly weaker).
- Lemma 2: no violations in 20 000 random instances; constants right. FDE counterexamples 1, 5 and 6 are correct as stated.
- Class structure (NI §1): the 91 support vectors give **43 distinct π = 27 (|H|=1) + 15 (|H|=2) + 1 (|H|=3)**. Every π-equal pair
  has |H| = 1 and a common shift of the two non-colluder counts. A minimal edit is missing only at (n_g1, n_g2) ∈ {(0,4), (4,0)}.
  The analytic share of decisions that admit a minimal null edit is 0.8285, consistent with the reported 81–82 %.
- 𝔽_mix contains exact Bayes in the limit: mix_exact matches exact Bayes on the stored confirmation surface to 2.7e-15.
  The parameter counts (11 / 6) and gauges (φ_rel(0) = 0 and φ_col(5) = 0 fix the 2-dimensional additive gauge) are right.
  The best 𝔽_vote fit to exact Bayes has R²_within = 0.106 (doc ≈ 0.1).
- The Proposition 1 construction keeps the argmax, and KL ≤ δB(1 − p*_ŷ) holds (checked at B = 1 … 100).

## Findings

**F1 (major). The 𝔽_mix vs 𝔽_vote verdict tests closeness to Bayes; it does not identify an "algorithm".**
Evidence: cl(𝔽_mix) contains exact Bayes and 𝔽_vote does not. So R²_within(best 𝔽_mix) ≥ R²_within(exact Bayes) ≥ 1 −
‖e_within‖²/‖g_within‖², and any near-Bayes map is 𝔽_mix-representable, whatever its internal computation. Two constructions:
- Exact Bayes plus an iid "lookup-table" error (η = 0.3) gives R²_mix 0.87 and R²_vote 0.09.
- Exact Bayes plus an additive *voting* component of equal within-episode norm gives R²_mix 0.79 and R²_vote 0.68. That is about
  the level (cT67: 0.81) at which the package calls a T1 model 𝔽_mix-representable.

The successful T1 models already have exact-Bayes relative R² 0.88–0.96, so their 𝔽_mix fit adds little beyond "close to Bayes".
"The same algorithmic factorization" (AVA §5, SNS "Shared" 2, MR item 5 and §17 Q8 "common computation … Yes") is therefore an
overclaim. By the definition's own asymmetry, a near-Bayes model can never be classified as a "different algorithm".
Correction:
- State the class evidence as the out-of-sample *increment* of 𝔽_mix over the scaled exact-Bayes member, penalised for its
  10 extra parameters.
- Add a competing class that also contains Bayes in its closure (e.g. Bayes plus an additive vote term), so that "different
  factorization at matched quality" becomes reachable.
- Rename the conclusion "same output-map family on the V-surface". Drop "algorithm" and "common computation".

**F2 (major). The "frozen ε" does not exist, and the package's own definition leaves most P-count models uncharacterised.**
Evidence: FROZEN_PREDICTIONS.md freezes only the arm difference in mix_rel (P3(b)). No ε appears anywhere in the package, yet AVA §4
says "At the frozen ε". The cohort values (`logs/conf/model_c*.json`) for the 6 successful P-count models give 𝔽_mix relative R²
0.38, 0.52, 0.52, 0.70, 0.74 and 0.85. For any ε < 0.26 at most 2 of 6 are representable; the rest are "not characterised" by the
AVA definition, and the definition then licenses no algorithmic claim. The excluded cP65 has 0.33 (𝔽_mix) vs 0.31 (𝔽_vote).
Correction: either declare ε as post hoc and report per-model classifications, or restrict the "same factorization" statement to
the full-budget T1 models (subject to F1).

**F3 (major). The definitions of computational and causal equivalence are under-specified and inconsistent with how they are used.**
Evidence (FDE §2): "correspond under a fixed alignment" does not name the class of alignments. With arbitrary maps on a finite
domain, any two models whose mediator values are injective in each other trivially correspond. The definition also involves only
mediator *values* under input interventions; there is no intervention *on* Z (no interchange or patching). It therefore does not
express "equivalent downstream causal operations" (mandate §4.1), and causal equivalence (§13) is never defined. Because it requires
log-odds agreement on I, computational equivalence excludes two temperature-scaled copies of one model. It also excludes AVA's
"same algorithm up to approximation bias" (θ ≠ θ′). The package thus uses three unrelated notions of "same computation":
FDE §2, AVA §1, and MR Q8.
Correction:
- Fix the alignment class (e.g. bijective affine maps fixed across I).
- Require interchange: Z_M ← a(Z_M′(ι(x))) gives the M′ output.
- Add an "up to output reparametrisation" variant.
- State the hierarchy: computational ⇒ log-odds on I(D) ⇒ posterior ⇒ classification. Class-relative "same factorization" implies
  none of these, and should be presented as such.

**F4 (major). The theorem's implications are misstated in MR §17 Q12.**
(i) MR says equal margin responses "*require* exact posterior matching *and* pointwise interchange exactness". Theorem 1 gives only
*sufficiency*. Counterexample: two models with the same non-Bayes error function e_M = e_M′ have identical input-level responses
(Corollary 1).
(ii) "Small KL … impl[ies] nothing about null responses" is false in expectation. T4 has a posterior floor p* ≥ 0.0152, and pointwise
KL(p*‖p_M) ≥ ≈ 0.01·min(B², B) at log-odds error ‖e‖∞ = B (worst over the 400 lowest-floor posteriors; rv_formal_klfloor).
Jensen's inequality then gives E‖e‖∞ ≲ max(10√κ, 100κ), and the density-ratio row bounds E_Π|Δ_M|. Proposition 1 defeats only
pointwise bounds.
Correction: say "sufficient", and add a table row for "average KL + posterior floor ⇒ average bound, no pointwise bound".

**F5 (major as a novelty claim; the maths is correct). The headline identity is definitional.**
Evidence: with e := c(ℓ_M) − c*, Corollary 1 is the expansion Δ = Δ* + e(x′) − e(x), and Theorem 1 is a two-line substitution.
MR §17 Q16 nevertheless lists "the exact error-increment identity" as "beyond standard results". Mandate §4 says not to claim a new
theorem by restating elementary facts.
Correction: grade both [E] and remove the novelty claim. The useful content is the *interpretation* (null-response correlation =
correlation of error increments), which can stay.

**F6 (minor). Proposition 1's quantifiers are false on the finite T4 domain, and the stated magnitude is off.**
- The decision inputs are finite: p₀ := min positive input probability ≳ 1e-17, so P(S) ≥ p₀ and, since pointwise KL grows with
  B (F4), E KL ≥ p₀·0.01·min(B², B). "For every κ > 0 and every B" fails. The construction gives ‖Δ_M‖∞ = (2/3)B, not ≥ B.
- "S has positive probability: 81 % of decisions admit such an edit" confuses P(S) = δ with the share of decisions that have a
  null partner.

Correction: "for every B and every κ ≥ c·p₀·B …" (practically vacuous), with B rescaled by 3/2.

**F7 (minor). π-preserving and Bayes-null are conflated.**
FDE Theorem 1 writes "π-preserving pairs (p*(x′) = p*(x))". Only the forward implication holds: 13.7 % of in-support
single-component count edits leave p* at q unchanged while changing π, and 2.6 % of decisions have p* invariant to all 91 v. So:
- NI's "null edits exist only when exactly one group has count 5" holds for π-null edits, not for Bayes-null edits.
- CE2's "every null edit changes the total count by ±2" holds for minimal π-null edits only; Bayes-null single edits change it by ±1.

Correction: define "null" as π-preserving everywhere and state the converse failure.

**F8 (minor). Errors in the approximation table (FDE §4).**
- TV row: "TV ≥ ε/2 per coordinate" has the wrong direction and constant. The fact needed is ‖p − q‖∞ ≤ TV (equality for K = 3,
  verified), so pointwise TV ≤ ε gives pointwise ‖·‖∞ ≤ ε.
- Probability row: "+ same for the max term" is superfluous. ‖e‖∞ ≤ log((p_min+ε)/(p_min−ε)) holds alone (max ratio 0.60 in
  50 000 trials). In T4, p_min = 0.0152, so this row is non-vacuous only for ε < 0.015.

**F9 (minor). Missing factor in the average-log-odds row.**
Π is conditioned on edit validity (≈ 83 %), so the base x is not distributed as P.
Correction: E_Π|Δ| ≤ (‖dP_Π/dP‖∞ + ‖dQ′/dP‖∞)·η̄, where ‖dP_Π/dP‖∞ ≥ 1/P(valid).

**F10 (minor). Missing assumptions in the counterexamples.**
FDE CE2 needs β > 0 (with β = −0.5 the argmax agrees with Bayes on only 4.6 % of decisions). CE4 needs null pairs to be ≤ 10 %
of the IIA evaluation distribution, which is false for natural N/R mixtures (N valid in 83 %). AVA CE2 ("uncorrelated in
expectation") holds only to first order in ξ. AVA CE1's "attention entropy is monotone in total count in this generator" is
unproven; replace it with "any mechanism that computes total count".

**F11 (minor). Proposition 2 and the "well separated" claim need qualification.**
- "(or a limit)": when the closures intersect only in the limit, no exact members need exist. Exact Bayes itself needs
  φ_col = −∞, which is outside 𝔽_mix. The conclusion is then "arbitrarily close", not "identical".
- cl(𝔽_mix) and cl(𝔽_vote) do intersect, at V-flat maps (φ ≡ 0; a ≡ const), so separation is not uniform.
- "Well separated" is supported only at exact Bayes (0.106). Fitting 𝔽_mix to a random 𝔽_vote member gave 0.39 in one of 3 draws.
- Correction: replace with a two-sided separation statistic over both classes.

**F12 (minor). The relative R² normaliser is mis-specified.**
- The ceiling is a 2-vs-2-realisation split-half R², but the target is the 4-realisation mean, so the ceiling is underestimated
  (Spearman–Brown). This produces ratios > 1 (cT60: 3.97; bcT67_b500: 2.02).
- The fits minimise loss over all entries but are scored on the within-episode part, so R²_within can be negative from that
  trade-off rather than from class misfit (e.g. 𝔽_vote −2.31 on cT60).
- Correction: use a Spearman–Brown-corrected ceiling and fit on the within-episode part.

**F13 (minor). Notation and grading.** The "ρ" in π(n) ∝ ρ^{n_g}[n_h = 5] and in "s = log ρ = +1.36" is E.RHO = 3.882 (the
reliability likelihood ratio), not the generator's ρ = 1; rename it (e.g. λ). SNS "Formal summary [P + E]": "𝔽_mix is shared"
inherits F1/F2 and nothing in it is [P]; grade it [E].

## Verdict table

| item | verdict |
|---|---|
| Theorem 1 (corrected C3), temperature remark | correct; trivial (F5); MR misstates it as necessary (F4) |
| Corollary 1 (error-increment identity) | correct; definitional, novelty overclaimed (F5) |
| Lemma 1 | correct |
| Lemma 2 | correct (numerically confirmed) |
| Approximation table | KL/accuracy "no pointwise bound" correct; TV row wrong direction (F8); average row missing factor (F9); average KL bounds average error under the T4 floor (F4) |
| Proposition 1 | construction valid; "any κ, any B" false on finite domain; ‖Δ‖ = 2B/3 (F6) |
| FDE counterexamples 1, 5, 6 | correct |
| FDE counterexamples 2, 4 | correct with added assumptions (β > 0; null share ≤ 10 %) (F10) |
| FDE counterexample 3 | = Proposition 1 (F6) |
| Class structure 27 + 15 + 1; π-null edits only at |H| = 1 | correct for π-null; not for Bayes-null (F7) |
| Null-edit availability 81–82 % | correct (analytic 82.85 %) |
| Operator-class definitions | well-posed but ε never frozen (F2); test is a Bayes-closeness test (F1); normaliser biased (F12) |
| Proposition 2 | correct for exact intersection; "or a limit" wrong; separation not uniform (F11) |
| AVA counterexamples 1–4 | valid as existence constructions; CE1/CE2 need the qualifications in F10 |
| Equivalence definitions (classification/posterior/log-odds) | correct |
| Computational / causal equivalence | under-specified, non-interventional, inconsistent with "same algorithm" usage (F3) |
| "Same algorithm / common computation" (AVA §5, SNS, MR Q8) | overclaim; supportable as "same output-map family near Bayes" for full-budget T1 only (F1, F2) |
