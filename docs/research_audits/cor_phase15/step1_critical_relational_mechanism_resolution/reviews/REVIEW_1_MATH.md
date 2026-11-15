# XIII-C Review 1: mathematics (independent, adversarial; AI agent reviewer)

Scope: `logs/NOTES_MATH_B1.md`, `DECISION_LAW_AND_MATHEMATICAL_IDENTIFIABILITY.md`, `logs/NOTES_MATH_C.md`, the C1 table of
`JOINT_RELATIONAL_FEATURE_AND_REPAIR.md`, `scripts/math_identifiability.py`, `scripts/c_math.py` and their JSON outputs, `src/genlib.py`,
`src/xii_bayes.py`, `src/relpairs.py`, `scripts/fair_fits.py` (with `logs/fair_fits_dev.json`), `scripts/d_das.py` (family construction and
rules only), `scripts/b5_residual.py` (with `logs/b5_residual_dev.json`). No existing file was modified. Check scripts are in
`reviews/review1_scripts/`. All were run at `nice -n 19` with small samples:
- `r1_localq_exact.py`: replays the c_math RNG stream. It gives the exact LOCALQ posterior on the same 3000 episodes as V4, the SNIS
  effective sample sizes, and a convergence check with 1500 draws.
- `r1_localq_large.py`: exact LOCALQ against FULL on 100,000 fresh episodes.
- `r1_edit_families.py`: per-pair check of the E5/E3/QF5/QF4 construction, with two extra rules (A4, SOFT).
- `r1_b5_cells.py`: (a) whether Bayes is constant within b5 cells; (b) geometric against arithmetic cell means.
- `r1_jpair_support.py`: what "on-support" means at β = 0.6, and how typical J-pairs are.

## Summary verdict

The central identities are correct as stated, within the stated generator:
- T1 and T2: power mean ≡ mixture with a misstated reliability at q, followed by the output exponent τ/ρ;
- Proposition S, and the coefficient in Corollary P;
- β = 1 local sufficiency;
- Bayes-invariance of VPERM, MOVE and XVAL;
- sufficiency of the b5 cell key;
- the eligibility-indicator properties of the four edit families. These hold on every pair I re-generated, and every pair is on-support.

One numerical claim is wrong, and its explanation is wrong. The β = 0.6 LOCALQ gap of "4.4 points" (c_math V4, NOTES_MATH_C, JOINT C1
table) is an artefact of a 48-draw self-normalised importance sampler. On the *same* 3000 episodes the exact LOCALQ posterior gives a gap of
2.8 points. On 100,000 episodes it gives 3.6 ± 0.07 points, which is XIII-B's value. The two estimators target the same quantity;
XIII-B's is exact.

Several statements overreach their maths or the package's own tables:
- the T3 framing;
- the "needs per-group format" reading of Corollary P;
- "E is the only rule", which holds only against the three rules tested;
- the B5 prose ("less than calibration", "nearly all … non-invariance");
- the J-pair "on-support" check, which tests nothing at β = 0.6.

None of these overturns a central mathematical claim.

## Must-fix errors

**M1. The LOCALQ gap is mis-estimated, and the discrepancy is mis-explained.** (`scripts/c_math.py` V4 l.55–76; `logs/NOTES_MATH_C.md`
"β = 0.6 (V4)"; `JOINT_RELATIONAL_FEATURE_AND_REPAIR.md` C1 row "4.4 points")

The estimator is correct in principle:
- The orbit (independent S3 relabelling per group and per question q′ ≠ q*) equals the whole cell {x′ : same self-agreement bits, same
  values at q*}. S3 acts transitively on the 3 agreeing pairs and simply transitively on the 6 ordered disagreeing pairs, so uniform
  permutations give a uniform proposal over the cell.
- Self-normalised weights ∝ p(x′) therefore target p(Y | T) consistently.

With 48 draws, however, the estimator is far from converged:
- The median ESS is 4.7 of 48, and 17% of episodes have ESS < 2.
- The mean max-|error| against the exact posterior is 0.057, and 4.8% of decisions differ.
- With 1500 draws on 200 episodes the error falls to 0.0076. The estimator converges to my exact formula, which confirms that both compute
  the same quantity.

The bias also has a known direction. The MC argmax is a randomised T-measurable rule, so its expected accuracy cannot exceed that of the
exact p(Y | T). The MC accuracy is therefore biased **downward**, and the gap **upward**.

Exact computation is cheap, because the cell sum factorises over questions. For each role g, the likelihood at q′ ≠ q* is:
- P(self-agree bit of g | reliable) × [β·1(both of H agree) + (1 − β)·Π_h P(agree | uniform)];
- the exact joint likelihood of the six values at q*.

This is XIII-B's `censored_bayes.py` LOCALQ. Results:

| episodes | FULL acc | LOCALQ acc | gap |
|---|---|---|---|
| c_math V4 stream (3000), MC with 48 draws (reproduced exactly) | 0.792 | 0.747 | 4.4 |
| same 3000 episodes, exact | 0.792 | 0.763 | **2.8** (paired s.e. ≈ 0.4) |
| 100,000 fresh, exact | 0.794 | 0.758 | **3.6 ± 0.07** |

*Fix:* replace V4 with the exact formula, or use the XIII-B number. Report 3.6 points. Delete the sentence "The two estimators differ … both
show a positive gap". The correct statement is that the MC version was an unconverged, downward-biased approximation of the same exact
quantity.

**M2. B5 prose contradicts the package's own tables.** (`DECISION_LAW…md` B5)

- (a) "The combination exponent explains 1–4% … less than … calibration (2–4%)". For uP161 s22, COMB = 0.038 > CAL = 0.024
  (`fair_fits_dev.json` shapley). Fix: drop "or calibration", or say "comparable to calibration".
- (b) "In mixture models nearly all of the (much smaller) error is non-invariance". The table gives within-cell 0.79 and 0.72, and
  structured 0.25 and 0.30. That is about three quarters, not "nearly all". The within-cell term also includes cell-mean noise and the
  geometric-mean effect (S6), so it is not purely non-invariance.
- (c) "Of the structured part, the parametric families capture about half" holds for the early units (0.19/0.37, 0.23/0.45). It fails
  for the mixture units (0.02/0.25, 0.04/0.30) and the hybrid unit (0.06/0.29). Scope the sentence to the early class.

## Should-fix issues

**S1. The T3 framing suggests a regime that does not occur on-support.** (`NOTES_MATH_B1` T3; `DECISION…` T3)

The exact cancellation of ρ needs all of the weight on one group. Under C = 1, β = 1, every on-support episode has at least one eligible
group h. Both groups g ≠ h then have m_g ≥ 1, so **every on-support episode has at least two groups with non-zero exact weight**.

The script's own exact-weight criterion (> 0.999) finds 0.0% single-group episodes (`math_identifiability.json`
`frac_effectively_single: 0.0`). The "6.3% one weighted group" uses a 1% threshold, and there ρ cancels only approximately.

*Fix:* state that the exact case is a limit never reached on support, and that the 6.3% is thresholded. Also say that ρ is uninformative
whenever the weighted groups have proportional L (identical k-profiles at q), which can happen with two or more weighted groups.

**S2. T2 wording and scope.** The identity is correct, and κ = (b²/noi)^ρ / (b′²/noi) does not depend on group or candidate. Three things
should be stated:
- (i) "the exact mixture computed with α′" means the *same-weight* mixture. The identity holds for any fixed w, including fitted
  (a, k, d), not only the Bayes weights. Say "MIX(w) with α′ at q".
- (ii) It needs every group at q to have the same number of sources, and needs per-source likelihoods with two values (the correct answer
  versus K − 1 uniform wrong answers). With unequal source counts, κ_g depends on g. The identity then survives only with reweighting
  w_g → w_g κ_g, which may leave the a·log W + k·log f + d parametrisation. With non-uniform wrong answers it fails. b = (1 − α)/2 is
  specific to K = 3; in general b = (1 − α)/(K − 1).
- (iii) It applies to PM and PML (lapse acts after the mixture). It does not apply to HYB or FULL with γ ≠ 0, because POOLED keeps
  α = 0.8. The `fair_fits.py` docstring line "(T2: == misstated reliability α′(ρ) at q)" should say "for γ = 0".

Consequence for B2 and B5: by T2, the parameter group labelled "COMB ρ" is equally "q-evidence tempering". The sentences "the combination
exponent ρ least" and "ρ_KL separates the classes" should use a label-neutral wording.

**S3. Corollary P's "needs per-group format" is an implementation hypothesis, not a consequence.** (`NOTES_MATH_C` "A downstream mixture
… therefore needs, at group g's tokens, the scalar m_{H(−g)}")

The corollary is a statement about the posterior. With G = 3, C = 2, the map g ↦ H(−g) is a fixed bijection. The posterior can equally be
written as a mixture indexed by pairs, with pair weights exp(c·m_H)·(per-group terms of the complementary group). The "group-dependent
selection" is then a fixed, input-independent permutation, so in principle a position-aware layer can implement it.

What the supG/supL2 contrast tests is whether layer 2 *learns* to use m supplied group-agnostically under one optimiser budget. The JOINT
doc correctly says this is not an impossibility claim. Two further points:
- supG supplies all three pair statistics, permuted per group (`adapt3.py group_feats`), and supplies them only at source tokens. supL2
  supplies them also at the readout. So "tests exactly this" is too strong: the two arms also differ in which tokens receive the features.
- *Fix:* replace "needs" with "a group-indexed mixture can read its gate directly from…", and label the format claim as empirical.

**S4. "E is the only rule correct on all four" holds only against the tested set.** (`NOTES_MATH_B1` T6; `DECISION…` B6)

I re-generated the d_das construction (600 pairs per family):
- 100% of base and edited episodes are on-support;
- the indicator changes are exactly as claimed for every pair (E5: E, A1; E3: A2; QF5: E, A2; QF4: A2);
- the E prediction equals exact Bayes (max diff 0).

Three caveats should be stated:
- (i) E and A1 differ only on QF5. Their predicted-response distances on E5, E3 and QF4 are 0.001–0.003, so the rejection of A1 rests on
  one family.
- (ii) Rules that coincide with E on every on-support input cannot be told apart from it by any experiment. One example is "n_h = 5 and
  (agree at q, or no other group eligible)", which equals E on support because some group always agrees at all six questions. "E" should
  therefore read "E up to on-support equivalence".
- (iii) Graded rules were not fitted, for example a SOFT rule giving weight 0.1 to groups with n = 4 that agree at q. The families can
  separate SOFT from E (mean predicted-response distance ≈ 0.8 on E5, E3 and QF4). Yet behavioural R²_E on E3 is only 0.71–0.79 in the
  early and hybrid units. A graded-eligibility alternative is therefore not excluded by what was run.

Say "E fits better than A1 and A2", not "every class follows E", unless graded alternatives are fitted.

The doc should also say that QF edits change the edited group's L_h at q. All rules' predictions move through L as well as through
eligibility, which is why A1's QF5 prediction is non-zero.

**S5. Fit uncertainty is understated, and the lapse parametrisation is fragile.** (`fair_fits.py`)

- The "bootstrap 5–95%" comes from **B = 8** resamples of the training episodes only. With 8 values, the 5th and 95th percentiles are
  essentially the range, and they ignore test-set and seed variability. The claim that the intervals do not overlap is a within-unit
  precision statement, not class-level inference: the classes rest on 1–2 dev units each. Report B, and use ≥ 100 resamples or report the
  range.
- λ and ρ are clipped inside `logq`. This creates flat regions: the stored raw λ for FULL is −15.2, −5.1 and −3.5 (effective 0). This is
  the likely cause of the uP161 s23 non-convergence (FULL KL 0.00385 > PML 0.00376). Use a sigmoid or softplus parametrisation.

**S6. b5: the geometric cell mean is not the KL-optimal function of the cell, and the additivity caveat is incompletely explained.**

The cell key is sufficient. Canonical exact Bayes is constant within cells (spread 2e-16; 355 cells on 30k episodes), and Bayes is
invariant under group permutation, candidate relabelling and swapping sources within a pair.

But `b5_residual.py` l.65–66 averages *log-probabilities*. The minimiser of the average KL(P_i ‖ c) over a cell is the **arithmetic** mean
of P_i, and with it the in-sample decomposition is an exact identity:

E KL(P‖B) = E KL(P‖c̄) + KL(c̄‖B).

Measured on synthetic deviations:

| synthetic deviation | geometric (b5) within / structured / sum | arithmetic within / structured / sum |
|---|---|---|
| KL ≈ 0.003 (model scale) | 0.551 / 0.430 / 0.981 | 0.543 / 0.457 / 1.000 |
| KL ≈ 0.06 | 0.534 / 0.354 / 0.888 | 0.469 / 0.531 / 1.000 |

So the non-additivity mainly comes from the geometric mean, together with the train/test split. The doc's "not exactly additive" caveat is
present but does not say why. At model scale the effect is a few points, so the 40:60 reading is roughly robust.

*Fix:* use arithmetic means, and report the in-sample identity plus the held-out inflation.

Two more b5 points:
- "Within-cell" also counts dependence on group order, candidate labels and source order within a pair. Bayes ignores these, but no edit
  family probes them; VPERM, MOVE and XVAL cover only q′ ≠ q content.
- "MOVE costs the most" compares edits of different sizes: MOVE changes two reports at two questions, XVAL one report. Normalise per
  changed report, or say so.

**S7. The J-pair "on-support" check is vacuous at β = 0.6.** (`c_math.py` V5; NOTES_MATH_C; JOINT C1)

At β < 1, every report array has positive probability: (1 − β)/K⁴ > 0, and reliable reports have full support. 100% of uniformly random
arrays have a finite posterior. "100% finite posterior" therefore carries no information.

The informative quantity is typicality. J-pairs lower the generator log-probability by a median of 3.6 nats, with 10% at ≤ −10.4 nats; 54%
are more than e³ less likely than their base. Report this. Neural responses on such atypical inputs may differ from those on natural
episodes.

**S8. The misspecification claim is justified but incompletely stated.**

"ρ is a property of (model, family, loss)" is supported by an existence example (0.56 under MSE against 0.71 under KL). Under
misspecification, the pseudo-true parameter also depends on the **input distribution** used for fitting (here, natural episodes). Write
"(model, family, loss, fitting distribution)".

The 0.56 comes from XIII-B on a different sample and objective. I did not re-verify it.

## Confirmed claims

- **T1:** definitional, and numerically 1e-15.
- **T2:** L_g(y) = (b²/noi)·r^{k_g(y)} with k ∈ {0, 1, 2}, and L_g^ρ = κ·L′_g with r′ = r^ρ. κ does not depend on g or y because every group
  has two sources at q and the same noi. α′ = r^ρ/(r^ρ + 2) gives 0.483, 0.635 and 0.765 at ρ = 0.3, 0.6 and 0.9. The weights W (from
  q′ ≠ q) and the eligibility m (from colluder agreement) do not involve the target-question α, so "miscalibration at q only" is right.
  The statement that no input–output experiment can distinguish the two descriptions is correct.
- **T4 and T5:** numerically consistent across four starts with negligible spread. The lapse family is not in PM, because no group has a
  flat L_g. Pure temperature lies in PM at ρ = 1 and is recovered. "Not confounded" should stay restricted to *pure* temperature, since T4
  shows τ and ρ co-move under lapse.
- **Generator against posterior.** `genlib.sample` with C = 2 shares both the copy event (`cop`, per (episode, question)) and the copied
  value (`col`) across the two colluding groups. Without copying, colluder sources keep the independent uniform default. `posterior_shared`
  implements exactly β·1[all four equal]/K + (1 − β)/K⁴. c_math V1's enumeration is independent and correct.
- **Proposition S:** the proof is complete. relm depends only on g's self-agreement bit (candidate symmetry with uniform wrong answers);
  Π colH = A^m B^{5−m}; and p(y | g, X) depends only on X_{q*}. The statement is in fact slightly stronger than written: per-group
  agreement *counts* suffice, not whole patterns.
- **Corollary P:** A/B = 1 + βK³/(1 − β), so log 41.5 = 3.73 at β = 0.6.
- **β = 1 local sufficiency (V3):** correct on support. If two roles are admissible, all three groups are identical everywhere, so all six
  values at q* coincide and p(y | g) is the same for every admissible g. Otherwise the admissible role is fixed by local information.
- **J_D0 control:** an exact change of zero follows from S.
- **b5:** VPERM, MOVE and XVAL keep n_g and the values at q, are on-support (none touches an all-six-agreeing group), and leave Bayes
  invariant. The b5 cell key is sufficient for C = 1, β = 1 Bayes.
- **T6 and d_das families:** the indicator properties hold on 100% of pairs, all pairs are on-support, E equals exact Bayes, and no single
  family separates all three rules.
- **Arithmetic in B2 and B5:** the FULL explained fractions (19–23% early) and the 40:60 split and 2.5–6× ratio for the early class match
  the JSON.

## Scope and overclaim notes

- Every [proof] item holds only for the stated generator: G = 3, two sources per group, uniform wrong answers, a single shared α,
  uniform role prior, and K = 3 in the displayed constants. T2 and S use symmetry assumptions that break under heterogeneous source counts,
  non-uniform errors or per-source reliabilities.
- **T6 against B6.** T6 [num] proves only that the families separate E, A1 and A2 *as rules*. "Every class follows E", "sharpened to E at
  the behavioural level" and the B4 row "early ungated pooling … rejected" are fitted-network findings [dev]. They rest on 4 dev units,
  with one hybrid unit and 1–2 per class. The labels are present, but "class" generalisations from this many units should be flagged as
  such.
- **B4 table, "ρ vs lapse: distinguishable in principle".** This is true as a population statement about PML. Given T2, however, no
  result in this package licenses reading ρ < 1 as a "combination difference" at all; it is equally q-evidence tempering. The B1
  consequence paragraph says so, but B2 reading 2, B5 and B3 ("Why does the early class have ρ < 1?") still use mechanistic language.
- **NPOE.** With ρ = 1e-3 and normalised weights, NPOE is a *weighted geometric mean* (weights summing to 1), not an unnormalised
  product of experts. "Product pooling rejected" applies only to that form.
- **HYB.** HYB's pooled term is one specific ungated pool (Σ log c_s with α = 0.8). The γ ≈ 0 result rejects that form only. The SWAP
  internal test is the more general argument, and it belongs to the mechanism review.
- **NOTES_MATH_C V2.** This is a sufficiency spot-check of single (group, question) relabellings. The proof carries the claim; the check
  is not a broad test.
