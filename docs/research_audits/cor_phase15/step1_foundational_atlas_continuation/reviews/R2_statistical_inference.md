# R2: Adversarial review, statistical inference perspective

Scope: RELATIONAL_GENERATIVE_LANDSCAPE.md, PP1_CALIBRATION_AND_INTERVENTION_THEOREMS.md, REPRESENTATION_AND_CAUSAL_IDENTIFICATION.md §4 with scripts/pp2_dev.py and src/pp2_sig.py, COR_TEN_FAILURE_MATRIX.md §1 with scripts/decision_ledger.py, and PROSPECTIVE_PREDICTIONS.md (P-A).

Method: I read the code and the logs. I re-ran small numpy checks: no training and no model inference. The check scripts are kept outside the package, in /tmp/r2c/c1..c9.py. All numbers marked "rerun" come from those scripts.

## Verified (no finding)

- **V1. The `src/roles_engine.py` posterior is correct.**
  - It agrees with `gen_family.posterior` to 2e-15 for post_y and post_cfg, at p_c = 0.5 and ρ = 0.7.
  - It agrees with a brute-force leave-one-out enumeration for one episode and question: [0.8632, 0.0245, 0.1123] from both.
  - The informative-colluder likelihood ρ·1[xa = xb]·P(c = xa | y) + (1 − ρ)/K² matches `sample_role`. So does the biased-error reliable likelihood.
  - The prior over permuted role assignments is uniform within each role list, which is correct. Duplicate roles collapse through `set`.
- **V2. The reliable↔colluder swap invariance at ρ\* is correct.**
  - With bias = 0, each pair's y-marginal depends only on its agreement rate.
  - The colluder pair is independent of y, so the per-question X-law factorises into pair marginals. Those marginals are equal when the agreement rates match.
- **V3. The one-vote threshold a_c\* = 64/66 is confirmed on a finer grid (rerun).** P(follow colluder) is 0.152 at a_c = 0.965, 0.184 at 0.968 and 0.922 at 0.9705. The residual 0.11–0.18 below the threshold, and the rise just below it, are not explained (presumably role uncertainty). "Sharp" is approximately right.
- **V4. The Theorem-U scope correction survives an exact check (rerun, closed-form binomial enumeration).**
  - Union posterior ("reliable of either α"): non-quasi-concave in n0 at **45/49** fixed (n1, n2) cells.
  - Each single role: 0/49 violations.
  - See F11 for the caveats.

## Findings

### F1 (critical): the "52 % of answers flip at ρ → 1" result is a NaN artifact (§1.5)

`roles_engine.posterior` has no guard for the case where every configuration has zero likelihood. The atlas `norm` has one. At an assumed ρ = 1 on data generated at ρ = 0.6, **78.1 % of (episode, q) cells have NaN post_y** (rerun). `argmax` of a NaN row returns 0. The step 0.95 → 1.0 therefore compares real answers with a constant 0:
- overall flip rate 0.521 (this reproduces the log's 0.517);
- **among cells defined at ρ = 1, the flip rate is 0.062**, which is in the interior range of 1–5 %.

Consequences:
- The claims "52 % at ρ → 1", "maximally sensitive" and "ρ = 1 is a singular point … one structural reason why behaviour at T4 constrains behaviour at ρ < 1 so little" are not supported by this computation.
- The support change at ρ = 1 is real. It shows up as *undefined* posteriors, not as flips.
- Fix: mask the undefined cells, report the undefined fraction separately, and add the atlas guard to the engine.

### F2 (critical): the C4 bound is vacuous on the actual PP1 population, so it "explains" nothing

I evaluated the C4 right-hand side on the scored edits (NB = 600 subsample of the frozen stream, exact Bayes π(x) and π(x′); rerun c5.py). I took min over τ of P(|Δm| ≤ b_τ(x) + b_τ(x′)) + Wε/τ², with W = 1:
- ε = 8e-4 → 1.009;
- ε = 1.2e-3 → 1.013;
- ε = 1.9e-3 → 1.021.

**The bound exceeds 1 for every τ.** The observed failure rates are 0.0002–0.0042.

The claims "W ≈ 1 here, so C4 reduces to natural KL ≈ 10⁻³ plus a margin tail; that explains the 0.996+ agreement mechanistically" (PP1 §4) and the first row of the ledger (COR_TEN §1, "a consequence of natural KL ≈ 10⁻³ with W ≈ 1 (C4)") are therefore unsupported.

The cause is the Markov/Pinsker step: ε/τ² is O(1) at any τ small enough to make b_τ < 0.25. C4 is a valid theorem, but it is not the explanation. The explanation needs the empirical distribution of t(x) on the endpoints, not its mean.

### F3 (major): "biased honest errors … change accuracy but not this boundary" is untested and false (§1.2)

The phase.json rows behind this sentence only use a_c ∈ {0.2, 0.5, 0.8, 0.9}, all far below 0.97. The boundary was never probed under bias.

**Theory.** With bias b = 0.8, the reliable pair's likelihood ratio depends on whether v = c + 1 or v = c + 2:
- for v = c + 1: (0.8/0.18)² ≈ 19.8, so a_c\* ≈ 0.908;
- for v = c + 2: (0.8/0.02)² = 1600, so a_c\* ≈ 0.9988.

**Rerun (bias 0.8):**

| conflict type | a_c = 0.90 | a_c = 0.92 | a_c = 0.975 |
|---|---|---|---|
| v = c + 1 | 0.207 | 0.995 | — |
| v = c + 2 | — | — | 0.005 |

The pooled rate moves smoothly from 0.14 to 0.82. The boundary moves and splits by error direction.

### F4 (major): W ≈ 1.2 is a coarse-partition lower bound; the pointwise ratio is larger (PP1 §4)

The 1.2 comes from sorted count patterns (343 cells). Coarsening can only lower a sup-ratio.

**Exact pointwise dE/dN** (rerun c3.py). I enumerated all toggle preimages: 2 per agreeing cell, 1 per b = a + 1, 0 per b = a + 2. Over the 18,714 S1 endpoints of the frozen stream, the ratio (E unnormalised) has:
- median 0.75;
- 99.9 % quantile 1.27;
- **max 1.36**.

E puts only 0.693 of its mass on S1, so the ratio relevant to the scored, S1-conditioned distribution is up to **1.36 / 0.693 ≈ 1.96**.

The |Δm| > 0.5 filter keeps 51.7 % of S1 (rerun) and depends on (x, q). It can inflate the scored-endpoint ratio by up to another factor of about 1.9. The maximum over a sample is itself a lower bound on the sup.

"W = O(1)" survives. "W ≈ 1" and "W ≈ 1.2" are understated. Given F2, W enters a vacuous bound anyway.

### F5 (major): the targeted CX2 test checks the wrong region; it is not evidence that "C7 doesn't bite"

CX2 corrupts points with **low log P(x)** (below the natural 2nd percentile). C4 and C7 concern a **high density ratio dE/dN**, which is a different property. Low P(x) is not a high ratio.

The "make the noise group perfect" family concentrates all of its mass on the event "≥ 2 groups agree in every k ≠ q". That event has natural probability **0.131** (rerun), so the family's density ratio is at least **7.7** at the event level.

Its endpoints are not low-P(x) because extra agreement *raises* P(x) under T4: a perfect pair is colluder-like. So "0 % of endpoints below the 2nd percentile" is expected for any agreement-creating family.

PP1 §4 says "C7 would bite only for an edit family concentrated off the bulk. None was found." That is contradicted: such a family was constructed, but the corruption region was not placed on it. The test should corrupt the region where dE/dN is high, for example the event above.

### F6 (major): the |Δm| > 0.5 selection removes about half of S1, where failures concentrate

- Scored set: 47,005 of 93,570 S1 edits. The excluded 46,565 have |Δm| ≤ 0.5.
- Every model's failures sit in the lowest scored bin, 0.5–1 (agreement 0.993–0.9997). The 1–2 and 2–5 bins are 1.000.
- So "sign agreement 0.996+" is conditional on a Bayes-side filter that excludes the margin tail. The margin tail is exactly the C4 term the document says governs failures.
- The decision ledger repeats the filter for the support rule.

Report agreement as a function of |Δm| (including (0, 0.5]), or with a sign-defined threshold. Do not use a single filtered rate as the headline.

### F7 (major): pseudo-replication and an unsupported stratum contrast (S1 \ S2, transformers vs DeepSets)

**The population is clustered.**
- The 135,000 edits are 27,000 distinct inputs × 5 query positions, from 1500 base episodes.
- The S1 \ S2 stratum is 3,570 edits, which is **714 distinct endpoints from 119 base episodes** (rerun). The 2,605 scored edits come from the same 119 episodes.
- No clustered uncertainty is given. A 0.968 rate (about 83 failures) can come from a handful of episodes.

**The text misreports the log.** TF s12 scores **1.000** on S1 \ S2. So the transformer range is 0.968–1.000, not 0.968–0.998, and "DeepSets 1.000" is not distinctive.

**The contrast is post hoc.** The transformer/DeepSets contrast is one of at least 10 strata × 10 models, with no multiplicity control. The "key-tag routing" hypothesis is post hoc. The text calls it untested, which is fair, but the stratum is presented as a concentration of failures.

### F8 (minor): S2 and S3 are deterministic artifacts of the edit design

- S2 = "the edited group is not the true colluder": 90,000 = (2/3)·135,000 exactly. Under T4 a colluder edit always breaks agreement; every other edit stays in S2.
- S3 = S2 holds *within this single-cell edit family*. As stated, the set identity is wrong in general: S3 ⊂ S2, because multi-cell x′ in S2 need not be one-mechanism reachable.
- S1 \ S2 is an episode-level selection. It contains colluder-break edits in the 7.9 % of bases where another group happens to be perfect, so the posterior must re-assign the colluder. "Impossible for the true world" is confounded with "forced colluder reassignment".

### F9 (major): the phase-diagram minimum claim contradicts the package's own log (§1.1)

The text says "the accuracy minimum sits at the grid point nearest ρ\* for α = 0.8 and 0.9". It bolds 0.619 at ρ = 0.5 and 0.670 at ρ = 0.7. phase.json shows:
- α = 0.8: minimum **0.614 at ρ = 0.55**. ρ = 0.5 gives 0.619.
- α = 0.9: minimum **0.657 at ρ = 0.75**. ρ = 0.7 gives 0.670.
- The table omits the 0.55 and 0.75 columns.

Rerun (4 × 10k episodes, mean ± SE):
- α = 0.8: ρ\* = 0.49 gives 0.6198 ± 0.0014; ρ = 0.55 gives **0.6145 ± 0.0004**. The minimum is displaced above ρ\* by about 4 SE.
- α = 0.9: ρ\* gives 0.6558, ρ = 0.75 gives 0.6584. The minimum is near ρ\*.

The role-mass minima are also displaced: 0.4 for α = 0.7, 0.75 for α = 0.9. Only the *swap equality* is pinned to ρ\*.

The α = 0.7 "three-way failure" explanation is ad hoc, and the α = 0.8 case contradicts the stated pattern. The swap check also uses only three ρ values, so "exact at ρ\* *only*" rests on theory, not on the grid. The theory is fine (V2).

### F10 (minor): §1.5 has a second off-by-one in "the steps crossing ρ\* ≈ 0.49"

13 % is the step 0.45 → 0.50 and 11 % is the step 0.50 → 0.55. Only the first step crosses 0.49.

### F11 (minor): Theorem-U scope (§1.4): the test object differs from the theorem, and "exact" is Monte Carlo

The script averages P(rel | n0) over the other groups' counts. Theorem U fixes the others' counts. The conclusion is right (V4: 45/49 fixed-others violations), but the logged evidence does not isolate it.

"Two local maxima" rests on 0.313 vs 0.304 (Monte Carlo). The exact values are 0.308 vs 0.307, so the second maximum is fragile. The quantity is computable exactly by binomial enumeration and should be.

### F12 (major): P-A is nearly uninformative at n = 4

I bootstrapped the development leave-one-out residuals, drawing 2 TF and 2 DS models (20k draws, rerun c6.py):

| claim | probability under the development model | what the result can show |
|---|---|---|
| P-A1: A1 beats all of B1–B5 | **0.53**; B1 (architecture) is the only real competitor (development LOO MAE 0.136 vs 0.147) | a coin flip; contradicts "a failure is informative" |
| P-A2: LPI_deg MAE ≤ 0.10 | **≈ 1.00** | a pass is near-certain, so it is no risk |
| P-A3 | low risk: development maximum 0.087 vs threshold 0.15 | — |
| P-A4 | low risk: development DeepSets 0.05–0.16 vs threshold 0.30 | — |

Further problems:
- **LPI_deg is nearly the target construct** (Spearman 0.988 with the local index). It measures "sides with the local plug-in when global ≠ local" on degraded T4 episodes instead of p_c = 0 episodes.
- **Those degraded episodes are off T4 support:** colluder disagreements are impossible under T4. So "computed on T4-derived inputs only" does not mean "predicted from on-support structure". CCP_j uses the same off-support inputs.
- **P-A3 measures in-support fit.** It tests that models fit their training support, not extrapolation.
- **The baselines are uneven.** B6 is a weak strawman (a random feature, averaged). B7 ridge fits 4 features on 10 points. Requiring A1 < min over 7 baselines on 4 points penalises A1 for baseline noise.
- **Recommended:** report MAE differences with a paired bootstrap interval, and state the a-priori pass probability.

### F13 (minor): the local-index target depends on an arbitrary plug-in ρ = 0.999

Rerun on the frozen p_c = 0 stream, 36k cells:
- the global plug-in argmax changes on 0.9 % of cells at ρ = 0.99, and on 0.9 % again at ρ = 0.9999. It does not converge as ρ → 1, because n·log(1 − ρ) dominates;
- the local argmax changes on 0.8 % of cells between ρ = 0.99 and 0.999.

The effect on the target is small, but the target is not a well-defined limit. Freeze it explicitly as a ρ = 0.999 quantity.

### F14 (minor): REPRESENTATION §4 correlations

- ρ_s = −0.89 over 10 models is driven by architecture: DeepSets is all 1.00 and all low on the local index.
- Within transformers, ρ_s = −0.71 at n = 7 has exact two-sided **p = 0.088**.
- pp2_dev screens 8 predictors on n = 10 without multiplicity control.
- The document labels this descriptive, which is appropriate. It should not be cited as structural support.

### F15 (major): decision-ledger inferences (COR_TEN §1)

- **"Same for all classes" and "not a class signature" rest on one model per class** (s10, s12, ds s0), with no intervals. A class effect cannot be tested at n = 1 per class.
- **Reversal "follow 0.82–0.83" has no reference rate.** The comparison should be model–Bayes argmax agreement on non-reversal S1 endpoints matched for margin. Without it, 0.82 cannot be read as good or bad. The log also shows that the models flip their own argmax in only 0.59 of reversals; this is not reported.
- **The "near-tie effects of the KL error" reading of the 0.13 confidence-only leakage is asserted, not stratified.**
- **The misleading-relation figure 0.63–0.65 has no T4-Bayes reference.** pp2_dev shows agree_true ≈ 0.33 in T4 support, i.e. the T4 plug-in itself disagrees with the truth about 0.67 of the time. The models are behaving as T4-Bayes, which is expected and not a decision failure peculiar to the networks.
- **The interior 1.000 for the support rule is a theorem-level consequence.** At ρ = 1 a group with leave-q count < 5 cannot be the colluder. It is still restricted by |Δm| > 0.5 (F6). Stating it as a theorem would be stronger than the empirical 1.000.
- **The first row inherits F2.**

### F16 (minor): reproducibility

`perms3` in landscape_phase.py builds configurations from `set(itertools.permutations(...))`. The tuples contain strings, so the iteration order depends on PYTHONHASHSEED. Verified: three seeds give three orders. The config indices passed to `rng.choice` therefore change between runs, so the sampled episodes and every Monte Carlo number in phase.json are not bit-reproducible. checks.json uses `sorted` and is unaffected. Use `sorted`.

### F17 (minor): wording in §2 (unknown groups)

- **§2.3.** The docstring reports mass on {0,1,2,3,4}, but that block is outside the hypothesis class (sizes 2–4). "The posterior converges to the two non-overlapping blocks" describes the *mean* mass 0.5/0.5. Per episode it is likely concentrated on one block. The non-block colluders are also modelled as reliable at α = 0.8, a second misspecification.
- **§2.2.** "Larger blocks are easier at ρ = 1" is confounded with the number of remaining reliable sources (6 − s) and with the size prior implied by the block counts (15/20/15). At ρ = 0.5 the order is non-monotone (size 3 < size 2 < size 4).
- **§2.1.** The T1-perm 1/3 bound is fine. MAP-correct uses first-index tie-breaking, which is harmless under a uniform permutation.

## Summary of severities

- **Critical:** F1 (ρ = 1 NaN artifact), F2 (C4 vacuous; the "mechanistic explanation" is unsupported).
- **Major:**
  - F3 (bias boundary claim false);
  - F4 (W understated);
  - F5 (targeted CX2 checks the wrong region);
  - F6 (|Δm| selection);
  - F7 (clustering, misreported range, post hoc stratum);
  - F9 (minimum location contradicts the log);
  - F12 (P-A power: a coin flip and a near-tautology);
  - F15 (n = 1 per class, missing references).
- **Minor:** F8, F10, F11, F13, F14, F16, F17.
