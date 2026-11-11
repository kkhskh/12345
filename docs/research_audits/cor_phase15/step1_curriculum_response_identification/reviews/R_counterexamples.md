# Counterexample review (mandate §13, "Counterexample reviewer")

Reviewer: independent; I did not produce the package. Scope: MAIN_RESULT, APPROXIMATION_VS_ALGORITHM, SHARED_AND_NONSHARED_STRUCTURE,
BAYES_NULL_RESPONSE_STRUCTURE, DOWNSTREAM_CAUSAL_MECHANISMS, FORMAL_DECISION_EQUIVALENCE, src/cri.py. Every construction is evaluated
on the package's own frozen conf generator data (`logs/conf/generator.npz`: the exact N/R pairs and the 300×36 V-surface) and uses
`cri.py` (`mix_logits`, `fit_family`, `r2`, `J_at`, `mix_exact`) and `E.pi_of_n`/`bayes_from` (exact gen_family T4 generator).
Scripts: `reviews/checks/rv_cex_{1..6}_*.py`; outputs: `logs/review_checks/rv_cex_*.out|json` (each < 3 min CPU, < 1 GB).
Severity: **major** = inference must be withdrawn or restricted; **moderate** = must be reworded or carry a stated condition;
**minor** = the package survives; only wording or a caveat changes.

## Findings

**1. A different pooling algorithm gets the same class verdict; for P-count models the "posterior-mixture" factorization is not
identified. (major)** `rv_cex_1`. Construction: F_log, the *logarithmic* opinion pool. It has the same 11 parameters and the same
count-gated configuration weights as F_mix, but the decision is Σ w(g,h)·log P(y|x_q,g,h) rather than log Σ w(g,h)·P(y|x_q,g,h).
This is a different algorithm: an expected log-predictive, not a posterior mixture. On the exact surface, the exact F_log member
fits F_mix at R²_within 0.912 (τ 0.96), MAP-configuration selection fits it at 0.891, exact Bayes fits F_log at 0.896, and all three
fit F_vote at ≤ 0.12. So a noiseless non-mixture algorithm clears the mix-relative R² of every P-count model (0.38–0.85) and the
bottom of the T1 range (0.81). Refitting the real cohort surfaces with F_log (relative R², same ceilings):

| | T1 cT61–cT67 | P-count cP61–cP67 |
|---|---|---|
| F_mix | 0.95 0.97 0.92 0.92 0.94 0.81 | 0.74 0.85 0.52 0.70 0.38 0.52 |
| F_log | 0.91 0.86 0.83 0.86 0.85 0.77 | 0.72 0.85 0.52 0.69 0.37 0.53 |

T1 models prefer the mixture by 0.04–0.11, so the classes do separate them; P-count models do not discriminate (|Δ| ≤ 0.02; cP63
and cP67 fit F_log at least as well). Consequence: "every full-budget T1 model and the better P-count models use the same
factorization (count → configuration evidence → posterior mixture)" (MAIN 5, SHARED §2, APPROX §4–5, answers 8 and 13) is supported
for T1 only. For P-count models the data support "count-gated pooling over configurations; linear or logarithmic pooling is
undetermined", so the cross-curriculum "same algorithm" verdict rests on the mixture/vote dichotomy alone. Proposition 2 concedes
that a verdict is only as sharp as the declared classes are separated; the problem is that the declared pair omits an alternative
that is not separated from F_mix at P-count fit quality. Correction: add F_log (or another non-mixture pooling class) to the
declared classes, and restrict the "same factorization" claim.

**2. The "13/13 P-count models have s = φ_rel(5) − φ_rel(4) < 0" pattern depends on the parameterization. (moderate)** `rv_cex_1`.
Under the equally fitting F_log, P-count s = +0.16, +0.24, −0.27, +0.23, −0.25, −0.65: three of six are positive. T1 models stay
positive under both classes (+0.60 to +1.14). Output ceilings do not create s < 0: label smoothing (ε 0.03–0.2), logit clipping and
tanh saturation applied to exact Bayes all keep s ≥ +1.36, so the sign is not a saturation artifact. That part survives.

Consequence: the reading "a 'perfect agreement is suspicious' heuristic inside the shared factorization" (APPROX §4, MAIN 7, SHARED
divergence 4) is a property of the F_mix coordinates, not of the models. Correction: state the sign as "under F_mix", and report that
it is not invariant across classes that fit P-count models equally well.

**3. τ ≈ 0.75 does not identify "under-confidence in count-dependent evidence". (moderate)** `rv_cex_1`. Exact Bayes with a
probability floor, p = 0.9·p* + 0.1/3, fits F_mix at R² 0.989 with τ = 0.74 and essentially exact φ_rel (1.53 per count vs
log ρ = 1.36); logit clipping and tanh saturation give τ 0.55–0.92. So an output floor that leaves the evidence untouched produces
the same τ. Correction (APPROX §4, SHARED divergence 1): τ measures compression of the output log-odds; where that compression
arises (count evidence vs readout floor) is not identified.

**4. A shared algorithm can give *opposite* irrelevant-count responses at matched, near-Bayes quality. (minor; it strengthens the
package's Counterexample 2)** `rv_cex_2` (A). Construction: two exact F_mix members, φ_rel(n) = n log ρ ± 0.1(n − 2.5)², with exact
φ_col and τ = 1. Both have KL < 5·10⁻⁴; each correlates 0.996 with the Bayes R response, and their R responses correlate 0.983 with
each other. Their Bayes-null responses correlate **−0.985** on the package's N pairs (κ = 0.25: KL 0.003, R r 0.90, null r −0.92).
Consequences: null-response coherence, high, zero or negative, carries no information about whether the factorization is shared,
even among near-Bayes models. The class itself produces null responses through curvature of φ_rel, so "the fitted F_mix explains
0–16 % of the null variance" (APPROX §4) is an empirical fact about these fits, not a property of the class. The package's reading
(Corollary 1; "not what produces the observed pattern") survives.

**5. Quality-dependent coherence arises with no shared algorithm, but it is not a function of KL alone. (moderate)** `rv_cex_3`.
Construction: random-feature ReLU networks; each seed draws an independent random first layer (a different implementation), with a
least-squares readout to exact Bayes on natural episodes. They share only target, data and input coding. Null coherence on N pairs:

| width | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|
| null coherence | 0.05 | 0.11 | 0.28 | 0.21 |
| KL | 0.30 | 0.15 | 0.07 | 0.05 |

Probability-space values are similar; R-error coherence moves the same way (0.06 → 0.40 → 0.29). This supports Corollary 1 and
"cannot identify an algorithm". But the relation is non-monotone, and an approximator with a different inductive bias contradicts
the quality account: a quadratic polynomial in counts (KL 0.077) has null responses *anti*-correlated with the RF networks at
matched KL (−0.31 to −0.38). Consequence: "explained by approximation quality" (MAIN 2, answer 3) assumes that two arms at equal KL share an error shape. Models
with different inductive biases (curricula) can have equal KL and opposite errors. Neither P2(a) (coherence linear in KL) nor P2(b)
(a training-stage manipulation that changes quality and stage together) excludes this. Correction: write "removed by a manipulation
that lowers T1 quality (and changes training stage); consistent with, not shown to be, a quality effect". Finding 7 shows that the
model-level data do fit the quality reading.

**6. A pure output ceiling cannot produce the coherence, but it can produce the sign of the "common structured bias". (minor)**
`rv_cex_2` (B), `rv_cex_3` (B2). Constructions: (B) unbiased iid noise per model and per input, with standard deviation decreasing in the total agreement count,
through a shared output ceiling C·tanh(·/C), at several C and τ; (B2) the same noise added to the configuration log-weights,
followed by the exact mixture readout. Neither produces null coherence (|coh| ≤ 0.014 at KL 0.01–0.14; cross-group τ = 1 vs
τ = 0.75: 0.008). The observed 0.19–0.34 needs genuinely shared deterministic error components, so the package survives this
alternative. The ceiling does produce a positive "dir" coefficient (more agreement → more confidence; +0.02 to +0.08); without a
ceiling, or with homoscedastic noise, it is ≈ 0 or negative.

Consequence: the common sign pattern in BAYES_NULL §4 / SHARED §4 does not by itself indicate a shared heuristic. Count-dependent
noise read out through any saturating map yields it. Correction: add this alternative to §4.

**7. P1b is invalid as a test; the model-level data fit a single quality-scaled factor. (major for P1b; supportive of the main
reading)** `rv_cex_4`, `rv_cex_6`.
- Construction: fit one common factor with model-specific loadings (S_ij ≈ l_i l_j) to the 66 cohort pairs. This is a world with no
  curriculum-specific component, in which coherence is purely loading/quality.
- Simulating it 200 times and running the package's P1b procedure (leave-pair-out factor, exact 924-label permutation) gives
  EXCESS_TT mean +0.094 and **p < 0.05 in 200/200 runs**. The leave-pair-out factor attenuates the loadings. The permutation null
  exchanges models whose loadings differ, so it cannot represent heterogeneous loadings.
- FROZEN_PREDICTIONS line 111 reads P1b as a "T1-specific shared component at full budget only". That reading is unsupported and
  should be withdrawn. (The observed +0.135 exceeds the simulated values, but the simulation treats the 3 284 response entries as
  independent and is therefore over-confident.)
- The same null predicts Δ = +0.19 (observed 0.149). The fitted loadings track log KL across all 12 models (Spearman −0.91, R² 0.89);
  adding arm does not help (coefficient −0.09; T1 mean residual −0.017). Residuals by pair type: TT +0.018, PP +0.060, TP −0.028.
- Better statistic: D_cross = cross − √(coh_T·coh_P) = −0.075, model-bootstrap CI [−0.164, +0.010]. Any excess is P-count-specific,
  not T1-specific, and is not established.

These model-level facts are the strongest support for MAIN 2. The package does not report them; they should replace P1b in the text.

**8. The within-arm contrast and the KL-adjusted contrast cannot detect curriculum-specific error structure. (moderate)**
`rv_cex_4` (iii). Construction: each arm gets its own shared error component (orthogonal between arms, loading 0.45) at equal
quality. Result: Δ = −0.018, KL-adjusted Δ = −0.018, cross-arm coherence 0.004. So "no curriculum effect after adjustment"
(P2(a); SHARED "the factorization is shared") is equally predicted by completely different error structures. Only cross-arm
coherence discriminates: here 0.174, close to within-P 0.186, which with finding 7 argues against this world for the cohort. Correction: report cross-arm coherence and D_cross whenever "same
structure" is inferred from a null within-arm contrast.

**9. The "selectively blockable nuisance route" does not survive: blocking the readout route makes the null response larger in three
of the four cited models. (major)** `rv_cex_5`. The route fractions are projections, rem = ⟨Δ_block, Δ_full⟩/‖Δ_full‖². They are not
removal fractions.
Construction: a toy with additive routes. The source route carries the Bayes response plus a large level signal +a·ΔΣn·w; the
readout route carries −a·ΔΣn·w, which cancels it; each route adds a small independent error. The toy reproduces the package's
pattern (N rem 0.31–0.42, "removes 58–69 %"; R rem 0.99, "keeps 99 %"), yet blocking *increases* the null-response norm 4–9× and
doubles the R-error norm.

The real arrays behave like the toy:

| | cT62 | cT64 | cT65 | cT67 |
|---|---|---|---|---|
| N rem | 0.35 | 0.53 | 0.66 | 0.59 |
| ‖blocked N‖ / ‖natural N‖ | **3.87** | **2.27** | **2.63** | 0.84 |
| R-error norm, natural → Rq-blocked | 0.18 → 0.42 | 0.27 → 0.45 | 0.26 → 0.43 | 0.33 → 0.31 |

Only cT67 shows a genuine reduction. In P-count models blocking shrinks both the N and the R responses (norm ratios 0.35–0.82 and
0.31–0.48), i.e. it is not selective.

Consequence: DOWNSTREAM §2 ("removes much of the departure from Bayes and keeps most of the useful Bayesian behaviour … a selective
modification") and MAIN 6 / SHARED divergence 3 ("selectively modifiable nuisance pathway") are false for cT62, cT64 and cT65. In
those models the readout route behaves as a large compensating signal that cancels a source-route level signal. P5(a) survives only
as a difference in the *profiles* of route projections between N and R. This independently matches the mechanistic reviewer's
non-additivity result (`logs/review_checks/rv_mech_routes.out`). Correction: report norms of the blocked responses next to the
projection fractions, and withdraw "nuisance" and "selective" except for cT67.

## Do the package's inferences survive?

| inference | verdict |
|---|---|
| Null coherence measures error similarity, not algorithm (Cor. 1, §5) | **survives, strengthened** (findings 4, 5) |
| "Explained by approximation quality" (MAIN 2, P2) | **survives at model level as a correlation** (loadings ~ KL, R² 0.89; finding 7). Not shown causally: the stage/quality confound remains, and equal-KL models can have opposite errors (5, 8) |
| T1-specific shared component (P1b) | **withdraw**: the test has a 100 % false-positive rate under a quality-only null (7) |
| Same factorization relative to classes (mix vs vote) | **restrict to T1**; P-count linear vs log pooling is undetermined (1) |
| s < 0 in P-count = "perfect agreement is suspicious" bias | **restrict** to the F_mix parameterization; not a ceiling artifact (2) |
| τ ≈ 0.75 = under-confident count evidence | **restrict**: an output floor gives the same τ (3) |
| Common "more agreement → more confidence" bias | **add the alternative**: count-dependent noise plus a saturating readout gives the same sign (6) |
| Curriculum-specific nuisance route, selectively blockable | **withdraw** for cT62, cT64, cT65; holds for cT67 only; keep P5(a) only as a profile difference (9) |
| Coherence produced purely by output saturation | **excluded** by construction (6): the package survives this alternative |
