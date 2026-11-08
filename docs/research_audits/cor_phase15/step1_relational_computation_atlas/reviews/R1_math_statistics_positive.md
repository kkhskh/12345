# R1: formal mathematics, statistical inference, positive-theory defense

Reviewer scope: THEOREMS_BOUNDARIES_AND_IMPOSSIBILITIES.md (TB), SOURCE_RELATIONS_AND_GENERATIVE_MODELS.md (SR), PROSPECTIVE_PREDICTIONS_AND_FALSIFIERS.md (PP), `src/gen_family.py`, `src/local_bayes.py`, `scripts/{bayes_boundaries,pp1_sign,pp2_shift,pp2b_local_posthoc}.py` and their logs. The mandate sections are §12 (a clearer statement of what is possible, conditional or unidentifiable) and §17.

My checks are in `/tmp/r1/c1.py` to `c7.py`. They were small CPU runs and are not part of the package. Where I computed something, the number is quoted.

**Verdict.** The core Bayes-level mathematics is sound:
- Theorem S on the stated family;
- Theorem U;
- the drop-condition algebra;
- Theorem D;
- I1 and I2.

Five things need correcting:
- the general form of Theorem S (assumption (U) alone does not suffice);
- the D1 "specialisation" (false as worded);
- the PP2(a)/(b) interpretation (underpowered, and the reference is arbitrary off the support);
- an undisclosed change to the PP1 protocol;
- several overstatements of I2.

No finding is fatal to the package's headline. Two findings are major for the PP2 conclusions.

---

## A. Verified as correct (positive defense)

**A1. The generator and likelihoods are correct, and (U) holds for every role.**
- Each role's `pair_lik` sums to 1 for every y. This includes the κ mixture (1−κ)p_a p_b + κ·1[x_a = x_b]·p_a.
- The sampler matches the likelihood: max |empirical − likelihood| is 0.0008 for rel and 0.0039 for coll, with n = 2e5 (c7).
- The Y-marginal of each role is constant on the diagonal and constant off the diagonal:
  - rel (κ = 0.4, ρ = 0.7): 0.26533 and 0.034;
  - coll: 0.26667 and 0.0333;
  - noise: 1/9.

  So (U) holds for rel(κ), coll(ρ) and noise.
- Analytically:
  - for rel, the off-diagonal value is (1−κ)(2αe + (K−2)e²)/K;
  - for rel, the diagonal value is ((1−κ)a₀ + κ)/K;
  - coll depends only on 1[x_a = x_b].

**A2. Theorem U is correct, including the "monotone extremes" claim and the a = 1 case.**
- **The proof is valid.** 1/P_j − 1 is a positive combination of exponentials that are affine in n, so it is convex. Its sublevel sets are therefore intervals, P_j's superlevel sets are intervals, and P_j is quasi-concave. When every s_{iJ} < 0, P_J is non-decreasing. This holds with ties too, because s = 0 terms are constant.
- **Degenerate case a_J = 1.** Step 5 of the proof is only a remark, but the case holds by either of two arguments:
  - (a) Pointwise limits of quasi-concave functions are quasi-concave. Take ρ → 1; P is continuous wherever it is defined.
  - (b) Directly: g(n) = g_N(n) + c·1[n = M] with c ≥ 0. Adding a non-negative bump at the endpoint keeps the second differences non-negative, so g stays discretely convex.
- **The numerical check B1 covers only P(rel) of group 0** (`bayes_boundaries.py` l.62), but the theorem is stated for every role. I re-ran it for rel, coll and noise over ρ ∈ {1, .9, .7, .5}, p_c ∈ {.05, .5, 1}, α, K, Q ∈ {3, 6} and κ ∈ {0, .3, .6}:
  - 41,391 finite rows;
  - 0 violations of quasi-concavity in any role;
  - 0 violations of monotonicity for the highest-a and lowest-a roles (c1).
- **Fix.** Extend B1 to all roles, or cite this check, and write out step 5.

**A3. The drop-at-full-agreement algebra is correct.**
- P_rel = W_R / (W_R + W_N ρ_N + W_C ρ_C). This gives the stated inequality.
- At ρ = 1: ρ_C(M−1) = 0 and ρ_C(M) = a^{−M}, which gives W_C > a^M W_N(ρ_N(M−1) − ρ_N(M)).
- ρ_N(M−1) − ρ_N(M) = (b/a)^{M−1}[(1−b)/(1−a) − b/a] > 0 when b < a. ✓

**A4. The phase-diagram numbers match `logs/bayes_boundaries.json`, after rounding, for all 25 cells.**
- For example, ρ = 1, p_c = .05 gives .986, and ρ = .5, p_c = 1 gives .361.
- Footnote \* is wrong (minor).
  - At ρ = 1, p_c = 1, 29.4% of the weighted contexts (1 − .042 − .664) have P(rel) **undefined** for n < M. With the other groups both below M, group 0 must be the colluder, so every configuration has zero probability.
  - These contexts are counted in the denominator as "no drop". They are not "pinned to 0 or 1".
  - **Fix.** Exclude undefined rows from the denominator, or report them as a separate column.

**A5. Theorem D is correct.**
- The conditions are complete except for one omission: the margin must be finite before and after the edit. That requires Σ_R π J(c) > 0 for c and c′, and E_{π^c}[r] > 0.
- Absolute continuity π′ ≪ π is the right direction.
- B3 checks the identity only in T4 (ρ = 1, p_c = 1), with 4 edit types × 300 episodes.
- **Positive defense.** The identity is a correct but direct rearrangement of Bayes' rule, closer to [K] than to a new theorem. Its value lies in D2, which is a genuine and correctly stated impossibility for any agreement-only sign law. That result directly refutes the original COR monotonicity claim, and it is not undersold.

**A6. I2 holds, and it is more general than claimed (undersold).**
- The same exchange invariance holds with **independent** honest errors (κ = 0) at ρ* = (a_rel − 1/K)/(1 − 1/K). For α = .8 and K = 3 that is ρ* = 0.49, and max |Δ episode log-lik| = 1.4e−14 (c7).
- So I2 is not about "honest copying". It says: **agreement-rate-matched collusion is undetectable by any statistic of X**, along the whole curve ρ*(α, K, κ).
- Combined with Theorem U, this gives a clean restricted theorem the package does not state:

  > **Corollary (proposed).** If a_coll ≤ a_rel, then for every p_c and every context of the other groups' counts, P(rel | n) is non-decreasing in n. In other words, no reversal is possible anywhere.

  The proof: rel is then the highest-agreement role, so the monotone-extremes part of Theorem U applies.
- The converse needs p_c-dependent conditions. ρ = .8, p_c = .05 shows no drop even though a_coll > a_rel. So a_coll > a_rel is necessary for reversal but not sufficient. The ρ = 0.5 row (0.667 vs 0.66) sits just 0.01 above this exact boundary.
- **Fix.**
  - Retitle I2.
  - Add the corollary as [P].
  - Draw the curve ρ*(κ) on the phase diagram.

---

## B. Findings

**1. Theorem S: assumption (U) is not sufficient as stated (major, statement scope).**
- **The gap.** Step 1 (M_q(R) = K⁻¹ Σ_y Π_g f_{r_g}) factorises only because each configuration has **at most one y-dependent group**. (U) is a per-role marginal condition, and it does not imply this.
- **Counterexample.** Take two reliable pairs (α = .8, K = 3) in one configuration. Both satisfy (U). Yet the per-question marginal is 0.1366 when both agree on the same value, and 0.0043 when they agree on different values (c2). So it depends on cross-group equality of values, not on the agreement bits. The same caveat applies to SR l.123 and to the group-size generalisation (TB §5).
- **Fix.** Add assumption (U′): in every configuration, at most one group's likelihood depends on y. Alternatively, state the general sufficient statistic: the agreement bits plus the cross-group equality pattern among the y-dependent groups.

**2. "Necessity of (U)" is overstated (minor).**
- The biased-noise example shows only that *some* violations of (U) break sufficiency.
- (U) is not necessary. Suppose every role's pair marginal has the form φ(x_a, x_b)·h_r(A) with a common base measure φ:
  - noise: i.i.d. from a non-uniform q;
  - colluder: copies c ~ q²/Σq² with probability ρ, and is otherwise i.i.d. from q.

  Then the role ratios are functions of A alone, so counts stay sufficient while (U) fails.
- The counterexample was not run.
- **Fix.** Say "(U) cannot simply be dropped". State the sufficient condition as "the role likelihood ratios depend only on A". Run the counterexample; it takes a few lines.

**3. The D1 specialisation is false as worded (major).**
- **The claim** (TB l.172): if g votes for c, a gain in g's agreement helps c iff the edit raises P(g reliable) more than P(g colluding).
- **The test.** I applied agreement-creating edits in question k, with g voting v at q, and compared sign(Δm_{v, v+1}) with [P′(rel)/P(rel) > P′(coll)/P(coll)] (c2).

| setting | sign mismatches |
|---|---|
| T4 | 653 / 17,013 (3.8%) |
| ρ .8, p_c .5 | 19,428 / 27,181 |
| ρ .7, p_c .2, κ .3 | 27,098 / 27,100 |

- **Why the last case fails.** There a_rel = .762 < a_coll = .8. Agreement raises P(coll) proportionally more than P(rel), yet it *helps* c, because the mass it moves comes mainly from **noise**.
- Replacing "colluding" with "not reliable" (noise ∪ coll) cuts the mismatches to 2 / 27,100 and 2,069 / 27,181. It does not remove them, because the other groups' votes at q also enter π^c.
- **Fix.** State the specialisation as E_{π^c}[r] vs E_{π^{c′}}[r]. Or, as a sufficient condition: the other groups' X_q is uninformative between c and c′, and r is higher on "g reliable" configurations than on the complement. Mark this [P+N] only after it is checked in ρ < 1 and κ > 0 settings.

**4. SR l.46: reversal "whenever p_c·ρ is large enough" is contradicted by the JSON (minor).**
- ρ = .5, p_c = .5 (product .25) has drop 0.000.
- ρ = .7, p_c = .2 (product .14) has drop .962.
- The governing quantities are a_coll − a_rel together with W_C/W_N (A3, A6).
- **Fix.** Replace the sentence with the A6 corollary and the A3 condition.

**5. The phase-diagram metric records the sign of arbitrarily small drops (minor to major for interpretation).**
- My magnitudes (c1; weighted mean of P(rel)(M−1) − P(rel)(M) over the contexts where it drops):

| cell | fraction with any drop | fraction with drop > .05 | mean size of a drop |
|---|---|---|---|
| ρ .5, p_c 1 | .361 | .000 | .003 |
| ρ .7, p_c .2 | .962 | .420 | .042 |
| ρ 1, p_c .05 | .986 | .907 | .114 |

- So "drop" fractions are discontinuous across the boundary (A6), while magnitudes go to zero continuously. The statement "as ρ → ρ\*, the non-monotone response must vanish" is true of the magnitude, not of the reported fraction.
- **Fix.** Add a column with the mean magnitude, or the fraction of drops above a threshold.

**6. I2 consequences are overstated (minor).**
- TB l.224 says "no estimator… can perform reliability-sensitive candidate selection". SR l.47 says the "decision value collapses to zero", and SR l.111 says "no estimator can succeed".
- The package's own numbers contradict this. Bayes reaches 0.566 against own-question 0.557, and 0.586 against 0.577, on 24,000 questions. Agreement still separates noise from {rel, coll}. Only rel vs coll is unidentifiable.
- The proof also needs more than a uniform agreed value. It needs:
  - an equal **disagreement** law (uniform over ordered distinct pairs, which holds because errors are symmetric);
  - a uniform colluder common value;
  - at most one y-dependent group (Finding 1).
- Y-uniformity does matter. Under a non-uniform Y prior, the honest pair's agreed-value law is non-uniform and differs from the colluder's, so the roles become distinguishable.
- **Fix.** Restate the consequence as "rel vs coll is unidentifiable; the attainable accuracy is the symmetric-mixture Bayes accuracy". List the three conditions.

**7. I4: no uncertainty is given, and the T4 claim is accuracy-specific (minor).**
- **Paired bootstrap over 1500 episodes (c2).**
  - T1 known − unknown = .026, 95% CI [.019, .034].
  - T4 = .0004, CI [−.005, .006]. The SE of the accuracy itself is about .005.
- **Seed sensitivity.** With seed 9, the T1 gap is .043. The Rao-Blackwellised expected accuracy, mean max posterior, gives T1 unknown ≈ .706 under both seeds, against the reported .714. Both posteriors are calibrated (c6), so this is noise.
- **T4 log-loss.** In T4, unknown pairing costs about .046–.050 nats per question (−.581 vs −.535 at seed 8; −.603 vs −.553 at seed 9). So "costs almost nothing" holds for argmax accuracy only.
- **Fix.** Report expected accuracy and log-score with CIs, and use ≥ 5000 episodes.

**8. PP1 was run on 500 base episodes against a frozen 1500 (major, protocol).**
- PP §1 says "stream base 6,100,000, 1500 episodes". `pp1_sign.py` l.10 sets `NB = 500`, and the logged n_edits of 45,000 equals 500 × 90.
- This deviation is not disclosed in PP §2.
- **Fix.** Disclose it, or rerun at 1500.

**9. PP1 scoring choices raise the headline agreement; the pass survives, but the numbers should be reported unfiltered too (minor).**
- **Effect of the threshold.** The filter was frozen, so this is legitimate. Re-running on 150 fresh base episodes (c5):

| model | boundary agreement, all defined edits | \|Δm\| > 0.1 | \|Δm\| > 0.5 |
|---|---|---|---|
| tf s0 | .799 | .837 | .982 |
| tf s12 | .939 | — | .999 |
| ds s0 | .954 | — | .997 |

  - Across all defined boundary edits, tf s0 sits right at the 0.8 bar.
  - The T1 plug-in's boundary agreement rises from .136 to .362 without the filter.
  - The filter therefore selects the large-effect, near-deductive edits.
- **Composition.** 83% of the scored boundary edits are 5→4 edits (730 / 878). These break one of two fully agreeing groups, so the Bayes update is close to logical. The scored boundary edits come from only 65 of 150 base episodes.
- **Clustering.** Edits are clustered at roughly 90 per base episode, so the nominal n overstates the evidence. Errors are concentrated in 1–4 episodes.
- **Off-support exclusion** (30%) is correct, because Bayes is undefined there. It does not inflate agreement, but it restricts the claim to the support, as the package already says.
- **Positive defense.** The PP1 pass is real, and the T1-plug-in control discriminates. However, a sign match on large on-distribution Bayes effects is largely implied by the models' near-Bayes accuracy. Calling it "implementation-independent" says that function-level convergence holds; it is not an additional finding.
- **Fix.**
  - Report sign agreement against a |Δm| threshold, including 0.
  - Use cluster-bootstrap CIs over base episodes.
  - Report 4→5 and 5→4 separately.

**10. PP2(a) and PP2(b) are underpowered: the disagreement sets are near-ties (major).**
- On the PP2 ρ = 0.9 stream, where true Bayes and the plug-in disagree (1150 questions):
  - true Bayes's log-margin between its choice and the plug-in's has a median of **0.023**, and 78% are below 0.2;
  - the plug-in's own margin has a median of 0.012, and **100%** are below 0.2 (c4).
- At ρ = 0.7 the medians are 0.15 and 0.20.
- On such questions a model that approximates *either* reference with on-distribution-sized error would split about 50/50. The models' .41–.53 at ρ = .9 is therefore evidence for neither H1-strong nor H2.
- **The control is in the package's own data.** On distribution, models side with the near-exact global reference only .67–.80 of the time against a weaker alternative (pp2b, T4 row). So the frozen 0.7 bar is at the ceiling of what a correct model attains on contested questions.
- Only the p_c = 0 shift has decisive margins (median 2.41). There, all 7 Transformers are below 0.7 (.19–.64), so **H1-strong is falsified for Transformers on that shift alone**. That is the defensible statement.
- **Fix.**
  - Restate the falsification as resting on the p_c = 0 shift.
  - Add a calibration control: the plug-in plus model-sized logit noise, or the agreement rate on on-distribution near-ties.
  - Restrict the scoring to disagreements with |margin| > τ.

**11. PP2: the plug-in with ρ = 0.999 is an arbitrary extension off the support (major, interpretation).**
- **Training does not determine the reference.** Training at ρ = 1 places no constraint on Bayes behaviour on inputs where no group fully agrees. Every ε = 1 − ρ is equally "training-distribution Bayes" there. In the shifts, 46% (ρ = .9), 76% (ρ = .7) and 87% (p_c = 0) of the disagreements occur in episodes with no fully agreeing group (c4).
- **The reference set itself moves with ε.** Changing ε from 1e−3 to 1e−4 changes 378 plug-in argmaxes at ρ = .9, against a disagreement set of 1150. At 1e−5 it changes 515.
- **Small ε approaches the true reference.** The plug-in family at ε = 0.1 *is* the ρ = .9 true Bayes. So "side with true" can mean "the network's implicit ε is larger", which is a finite-logit smoothness effect, not adaptivity.
- **A principled reference exists.** The ε → 0 limit is well defined (lexicographic: the fewest disagreements in the colluder) and should replace 0.999.
- **Fix.** Fit a per-model implied ρ̂ on off-support data from one shift, and test whether that single ρ̂ predicts the other shifts. That version of H1 can actually be falsified.

**12. The PP2b local/global axis carries a forking-paths risk (major for SR l.132; minor for PP, which labels it post hoc).**
- **Choices made after seeing PP2.** The local reference (`local_bayes.py`) was designed after PP2. It has free choices:
  - a role prior of 1/3 each;
  - the training parameters a_r, including ε = .001;
  - the leave-one-out count;
  - the product form.
- **Same data.** It is evaluated on the same streams as PP2.
- **Mixed results across seeds.** The off-support "local" side-rates on T1 data are .59, .83, .49, .87, .70, .30 and .53 across the 7 Transformers, which is a continuum rather than a dichotomy. The PP text highlights 3 seeds.
- **The on-distribution "global" preference is not evidence of a global mechanism.** Global ≈ true Bayes there, so siding with global only restates PP1-level accuracy.
- **The reported range is slightly wrong.** PP §2 gives the on-distribution range as 0.69–0.80. The JSON gives 0.67–0.80 (DeepSets s0 .669, s2 .685).
- **SR l.132 overclaims.** It says implementations "diverge … in organised ways (PP2: global versus local)", and this is stated as a fact.
- **Fix.**
  - Downgrade SR l.132 to "exploratory".
  - Preregister the local reference, with its ε and prior fixed or swept.
  - Test on new streams and new seeds.
  - Report cluster CIs. Disagreements are clustered in about 5000 of the 6000 episodes on T1 data.

**13. A minor code/protocol mismatch in PP2 (minor).**
- The frozen falsifier is "< 70% … for a majority of models", i.e. #siding < n/2.
- The code (`pp2_shift.py` l.54) uses `<= n/2`.
- This makes no difference to the outcome (0, 2 and 3 of 10), but the code should be aligned with the frozen text.

**14. Theorem U's interpretation conflates the role posterior with the decision response (minor).**
- TB l.108–111 describes "the response to more agreement" as unimodal. Theorem U is about P(rel | n). The candidate margin is a different object, governed by Theorem D, and it mixes all the roles plus X_q.
- Its quasi-concavity in n is not proved.
- **Fix.** Say "the reliability posterior". Either prove or test the margin's shape, or state that it is not claimed.

**15. Editorial issues in the Theorem S proof (minor).**
- Step 1 contains a false start ("M_q(R) = Π_g Σ_y … In fact …").
- **Fix.** Rewrite it as one line, with assumption (U′) from Finding 1.

**16. Known ingredients should be cited, not dismissed (positive defense, minor).**
- **Theorem U** is a variation-diminishing consequence of the log-linearity of L_j in n, i.e. total positivity (Karlin 1968). It should be cited as such.
- **The restricted statement remains a legitimate contribution:** the A3 exact drop condition, plus the A6 boundary a_coll = a_rel. It should not be downgraded merely for being conditional or elementary.
- **Theorem D** is close to [K]. **D2 and I2** are the substantive impossibility results.

---

## Summary table

| # | target | severity |
|---|---|---|
| 1 | Theorem S assumption (U) not sufficient (needs ≤ 1 y-dependent group) | major |
| 3 | D1 specialisation false as worded | major |
| 8 | PP1 run at 500 vs frozen 1500 episodes, undisclosed | major |
| 10 | PP2(a)/(b) disagreement sets are near-ties: uninformative | major |
| 11 | ρ = 0.999 plug-in is arbitrary off the support; falsification should rest on p_c = 0 and an implied-ρ̂ test | major |
| 12 | Local/global axis post hoc, same data; overclaimed in SR | major (SR) / minor (PP) |
| 2, 4, 5, 6, 7, 9, 13, 14, 15, 16 | necessity overstated; p_c·ρ rule; drop magnitude; I2 consequences; I4 CIs; PP1 filter reporting; falsifier code; U interpretation; proof text; citations | minor |
| A1–A6 | generator, Theorem U (all roles, a = 1), drop algebra, phase numbers, Theorem D, I2 (more general than stated) | verified |
