# Reviewer 1: formal mathematics and Bayesian inference

Pre-review commit 76a9d60. Proofs checked by hand. Numerical checks are in /tmp/r1rev/chk.py (read-only use of src/ and
logs/eval_dev.npz).

## Verified as correct

- **Proposition 1.** Per-question marginals:

  | pair role | agrees | specific disagreement |
  |---|---|---|
  | reliable | 0.22 | 0.0567 |
  | noise | 1/9 | 1/9 |
  | colluder | 1/3 | 0 |

  This gives π(g,h) ∝ RHO^{n_g}[n_h=5] with RHO = 66/17. The remaining factors are common to all six configurations and the
  prior is uniform. The own-q factor (1/K)L_g(y)(1/K)[a_h]K^{−2} then gives Σ_g RHO^{n_g} m_g L_g. Natural episodes are
  always in support.
- **Lemma 1.** The oriented-margin coefficients sum to zero, so the normaliser cancels. The 64 bound holds for every pair
  pattern (α², αε, ε² = 0.64, 0.08, 0.01). By the mediant inequality, R(c)/R(y) ≤ 64 for any non-negative mixture R.
- **Weight bookkeeping (Theorems 1–2).** With |H|=1: m_{h*} goes 0→1 and m_f goes 1→2. In R1, g and h* have identical
  likelihoods, so the mixture really has two components.
- **Numbers.** All values in the R1_INFO §2/§3/§6 tables and in FORMAL match `r1_information.json` and `formal_checks.json`.

## Findings

**1. Major. The Corollary is false as stated** (FORMAL Corollary 1; R1_INFO §3 Corollary and §5 item 2; MAIN bullet 1 and §1).
- *Problem.* 𝕄 allows any W_g(X). Theorem 1 excludes only weights that depend on the group's own count. Counterexample in 𝕄
  that ignores other groups' counts:
  W_g = RHO^{n_g}·0.1^{[n_g=5 ∧ g's q-vote shared by no other group]}.
  On the dev |H|=1 pairs it gives:
  - R1 +0.041 (88 % positive, 0 % negative);
  - R3 −0.716 (100 % negative);
  - R2 +0.697 (wrong).

  So it reproduces the Bayes R1/R3 signs.
- *Correction.* The provable necessity statement is **R1 vs R2**:
  - If W_g depends only on (n_g, q-votes), the edit changes only W_g.
  - An R1 pair and an R2 pair can share a vote pattern and differ only in whether the matching group is perfect.
  - So Lemma 1 forces them to have the same sign.
  - Hence R1>0 together with R2<0 requires information about other groups' X_{−q}.

  Restate the Corollary, §5 and MAIN accordingly.

**2. Major. The Theorem-2 "Reading" contradicts the package's own check** (FORMAL lines 63–66, "R2/R3 stay negative whenever
v(0)≈0"; `formal_checks.py` docstring).
- *Problem.* `formal_checks.json` has `T2_all_hold.R3 = false`. With a fresh seed, 105/300 random (w,v) give positive R3
  pairs. This happens when f votes with h*.
- *Correction.*
  - Drop R3.
  - Add the R2 result, which is provable. Before the edit, all positive weight in an |H|=1 R2 pair is on pairs unanimous on c,
    so m = log 64, the maximum. After the edit, h* (unanimous on c′) gets weight w(5)v(1) > 0, so Δm < 0 strictly.
  - State that the R1/R2/R3 theorem requested in mandate §11 is only partly delivered.

**3. Major. The empirical Theorem-2 table uses the wrong boundary** (FORMAL "Empirical fit").
- *Problem.* The table compares log v(2) − log 2 against Bayes log RHO (1.36). The boundary is each model's own fitted
  log(w5/w4). From `separability_fit_dev.json`:

  | model | log(w5/w4) − (log v2 − log 2) | fitted log(w5/w4) |
  |---|---|---|
  | T70 | +0.91 | |
  | T73 | +0.56 | |
  | P80 | −1.17 | 0.27 |
  | P81 | −0.30 | |

  The conclusion survives. But P80 fails mainly through a collapsed reliability gain, contrary to the Reading's emphasis on
  third-group inflation.
- *Correction.* Tabulate both terms and their difference. Note that τ≠1 does not affect signs.

**4. Minor. The explaining-away table** (R1_INFO §2).
- *Problem (a).* The script takes h* as the first other group with n=5. In 6/1354 dev pairs that group disagrees at q. This
  produces the "before h*" entries 0.003/0.002, which contradict the table's own RHO⁵·0 row. With correct selection they are
  0.000.
- *Problem (b).* The values are normalised W (conditioned on X_{−q} and colluder q-agreement), not the posterior given X.
  - R1-before, g: full posterior 0.953 vs normalised W 0.920.
  - R1-before, f: 0.047 vs 0.080.
  - Maximum per-pair gap: 0.295.
- *Correction.* Select h* with V≥0, and relabel the values "mixture weights" or report the full posterior. The qualitative
  argument stands.

**5. Minor. §5's "minimum information" claims are not proved minimal.**
- *Problem.*
  - "Needed" rests on Theorem 1, and finding 1 shows that inference is misdirected.
  - The minimal sufficient statistic is p*(·|X) itself (two numbers), so the triples are not "smallest". Groups with W_g=0
    need no q-reports.
  - "The Theorem-2 balance is a smaller sufficient summary" is a category error: it is a property of the rule, not a
    statistic of the data.
- *Correction.* Call the triples "a natural sufficient factorisation". State necessity only relative to named rule classes,
  as in finding 1.

**6. Minor. Theorem 1 is stated more narrowly than its proof allows.**
- The proof never uses |H|=1 or the regime: 0 violations in 100 rules × all 1360 oriented F45 pairs, including |H|≥2.
- The results also hold for the mandate's pairwise margin log p(c) − log p(c′), since ds = s(c) − s(c′) ≥ 0.
- State the assumptions explicitly: exact (g-extremal) likelihoods with ε>0; w>0; v(1), w(n_f) > 0.

**7. Minor. "Vote-blind / no vote×count interaction"** (R1_INFO §1) describes a parametrisation, not the function. Bayes Δm
is regime-dependent. Do not use this phrasing to constrain what networks must compute.

**8. Minor. Status and implications are overstated.**
- Theorems 1–2 are one-line consequences of Lemma 1 plus the mediant inequality. Label them [P, elementary].
- MAIN §4E ("rescue is explained by Theorem 1") and bullet 2 ("therefore settles on a near-separable discount") overreach.
  The theorem concerns exact-likelihood mixtures, and the separable fits to P-count models have R² 0.79–0.84.
- Theorem 2 itself shows that supplying m does not guarantee R1>0.
- FORMAL "Limits" is appropriately careful. MAIN should match it.

**9. Minor. Provenance.**
- R1_INFO §4 Nested-S (+0.455/+0.281) and F54o (−0.239) are dev-stream values (`eval_dev_summary.json`). They sit beside
  stream-17_000_000 values (+0.441/+0.268).
- `formal_checks.py` says 17_000_000 but loads `eval_dev.npz`.
- "sign(F54o) = −sign(R1)" holds pairwise only because Theorems 1–2 fix the R1 sign. Say so.
- Bayes R3 is 0.39 % positive overall (|H|=2 pairs only), so "Bayes R3<0" needs |H|=1.

## Results to defend against overly broad rejection

- Proposition 1 is exact and correctly derived: T4 Bayes is a three-component mixture with a cross-group eligibility factor.
- Lemma 1 and Theorem 1 are correct and more general than stated. Every own-count-only rule couples the R1 and R3 signs.
- Theorem 2 (R1), together with the R2 result, is a valid characterisation within the eligibility family. Its P-count
  conclusion survives correction.
- The explaining-away mechanism is correct with the actual latent weights: h* moves from certain colluder to exchangeable
  with g, and c gains 2RHO against the third group's 2.
- No information-theoretic or Transformer-level theorem is claimed.

## Verdict

**Accept with major revisions.** The core mathematics is correct: Proposition 1, Lemma 1, Theorem 1 and Theorem 2 for R1. Three
statements built on it must be fixed:

- the R1/R3 Corollary and §5 necessity claim: refuted, and should be restated as R1/R2;
- the R2/R3 Reading of Theorem 2: R3 is false; R2 is provable and should be added;
- the empirical boundary in the Theorem-2 table.

No numerical claim conflicts with the logs.
