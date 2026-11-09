# Theorems, counterexamples and limits (continuation; mandate §2, §6, §8, §10, §12, §14)

**Labels.**
- **[P]** proof given.
- **[N]** numerical check, exact Bayes.
- **[E]** empirical, learned models.
- **[X]** counterexample.

The atlas theorems (S, M, U, U2, D, D1–D2, I1–I7) are not restated. They are corrected in scope in §1. New results are in §2–§6.

## 1. Scope corrections to the atlas theorems (§8)

| theorem | atlas scope | corrected scope | evidence |
|---|---|---|---|
| **S** (count sufficiency) | under (U) and (O) | (O) is *needed*, not just used. With two answer-dependent groups (α = 0.9 and α = 0.6), episodes equal in leave-q counts and own reports differ in their Bayes posterior by up to 0.458 (median 0.0009). In T4 the spread is 2·10⁻¹⁵. S is a statement at the **role-posterior** level. At the decision level the statistic is (X_q, n^{−q}) (J1) | [N] `logs/landscape/phase.json` |
| **U** (quasi-concavity) | per role, under (U) | holds **per role**. It fails for a *union* of roles with different agreement rates: an episode-level mixture over the reliable accuracy, with an intermediate-agreement competitor. Example: two local maxima, P(rel \| n) = 0.296, **0.313**, 0.304, 0.244, **0.118**, 0.154, 0.737. U says nothing about margins: margins go through D. The "never turns more than once" reading of the atlas applies only when "reliable" is one role | [X, N] RELATIONAL_GENERATIVE_LANDSCAPE §1.4 |
| **U2** (no reversal without a higher-agreement role) | reliability posteriors | correct at the posterior level. At the **decision** level the operative comparison is evidence per independent vote (V1 below). A more accurate colluder (a_c = 0.9 > α = 0.8) still loses conflicts | [P, N] |
| **D** (margin identity) | absolute continuity π′ ≪ π | unchanged. Applicability on the PP1 population: S1 edits only; 41,430 of 135,000 edits are outside S1. On S1 \ S2 the identity holds, because the configurations are possible marginally | [P] |
| **D2** (no universal sign) | — | strengthened by C5 and C6 below: no sign rule from accuracy alone, and none from calibration alone | [P, X] |
| **I2** (role swap at matched agreement) | stated with checks at two settings | verified as an exact symmetry along the whole line ρ*(α) = (K·a_rel − 1)/(K − 1) for α = 0.7, 0.8, 0.9: E[P(true rel)] = E[P(colluder called rel)] to Monte Carlo error at ρ*, and only there | [N] |
| **I5 / Shapley** | head-level and summed shares not invariant (withdrawn R2-c) | replaced by R1 (§3): quotient-game Shapley over exact pattern classes **is** invariant under exact splits and gauge | [P, N] |

## 2. Calibration, support and the sign of interventions (§2)

Full statements and proofs are in PP1_CALIBRATION_AND_INTERVENTION_THEOREMS.md.

| id | result | type |
|---|---|---|
| C1 | \|Δm\| > \|e(x)\| + \|e(x′)\| ⇒ the sign agrees | [P]; never violated in 10 models [E] |
| C2 | the log-probability sup-norm version | [P] |
| C3 | the TV version, with π_min | [P] |
| C4 | distribution-level: P(fail) ≤ margin-tail term + W·ε/τ² (density ratio W, natural KL ε) | [P] |
| C5 | monotone-transform invariance: temperature or ECE does not affect signs | [P, N] (TEMP models: ECE 0.22, sign 1.0) |
| C6 | no accuracy-only guarantee | [X] CX1: Bayes accuracy, sign agreement 0.22 |
| C7 | no average-KL guarantee without a density-ratio bound | [P]; non-binding on PP1's edit family (W ≈ 1.2) [N] |

## 3. Split- and gauge-invariant attribution (§6)

**R1 [P, N].**
- **Statement.** Shapley values of the quotient game whose players are exact attention-pattern classes are invariant under exact head splits and within-head gauge transforms.
- **Proof.** REPRESENTATION_AND_CAUSAL_IDENTIFICATION §1.
- **Limit L1 [X].** Function-preserving reparametrisations that change patterns, by moving mass between equal-value positions, are not covered. No attribution defined over attention heads can be invariant to those *and* still distinguish heads with different patterns. The two changes produce the same function with different patterns, so a pattern-sensitive attribution must differ between them.

## 4. Generative-level results (§9, §10)

**V1, "a copying pair is one vote" [P, N].**
- **Setting.** Known roles, K-ary symmetric errors. A reliable pair (accuracy α) agrees on v. A pair copies c ≠ v with probability 1; the copied value has accuracy a_c. No other evidence.
- **Statement.** The posterior prefers c iff (K − 1)·a_c/(1 − a_c) > ((K − 1)·α/(1 − α))².
- **Proof.**
  1. The copied pair's likelihood ratio between c and v is that of a single report, because the second report is a deterministic copy.
  2. The reliable pair's ratio is the product of two independent reports.
- **Check.** Threshold a_c* = 0.970 for α = 0.8. P(follow) jumps from 0.133 at a_c = 0.96 to 0.922 at 0.97.

**I1′, corollary of I1 [P, N].** Under T1 with a hidden pairing, the posterior mass on the full true pairing is ≤ 1/3 for every Q, because the noise sources re-pair in 3 equivalent ways. Observed: 0.10, 0.15, 0.22 at Q = 3, 6, 12.

**Singularity at ρ = 1 [P, N].** For fixed X, consider configurations that make a pair with n ≥ 1 disagreements the colluder.
- **Statement.** Their likelihood carries the factor ((1 − ρ)/K²)ⁿ. The configuration posterior is *continuous* in ρ on [0, 1], but its log-odds diverge like n·log(1 − ρ), so its sensitivity |∂ log-odds/∂ρ| = n/(1 − ρ) is unbounded as ρ → 1.
- At ρ = 1 those configurations leave the support. Absolute continuity, the assumption of Theorem D, fails exactly there.
- **Check.** On episodes sampled at ρ = 0.6:
  - 52% of argmax decisions flip between assumed ρ = 0.95 and ρ = 1.0;
  - 1–5% flip per 0.05 step in the interior;
  - 11–13% flip at the role-swap line.
- **Consequence.** T4 sits at a boundary point of the latent-reliability family where the posterior is maximally sensitive to the copy rate. *(Corrected before commit: an earlier draft called this a discontinuity. It is continuous with unbounded sensitivity, plus a support change at the endpoint.)*

**Misspecified overlap [X, N].** With a one-block hypothesis class and an overlapping truth (one source copying two blocks alternately), the posterior concentrates on the two non-overlapping blocks (0.50 / 0.50 at Q = 24) and gives the true structure probability 0. That is, misidentification is guaranteed and persists as data grows.

**Relative-count invariance under T1 [P].** Under T1, adding one agreement to every group in a question k ≠ q leaves P(Y_q | X) unchanged.
- **Proof.** P(g reliable | counts) ∝ Π (a_rel/a_n)^{n_g} · ((1 − a_rel)/(1 − a_n))^{M − n_g}. A uniform shift multiplies every configuration by the same factor (a_rel/a_n) · ((1 − a_n)/(1 − a_rel)).
- **Note.** Under T1 exactly one group is reliable, so each configuration's likelihood contains exactly one rel factor.
- **Models [E].** T1 Transformers have invariance ratio 0.02–0.04 (seed 12: 0.14).

## 5. O/D: the restricted bridge (§14)

**OD1 (exact bridge) [P, N].**
- **Statement.** Suppose all sources are conditionally independent given Y, share one accuracy α > 1/K, and have symmetric errors. Let D_c(x_q) = #{sources s : x_{q,s} ≠ c}. Then log P(c | x_q) = const − D_c · log(α(K − 1)/(1 − α)), so the discrepancy ranking of candidates equals the Bayes ranking exactly.
- **Proof.** The likelihood of x_q given Y = c is α^{S−D_c} · e^{D_c}, with e = (1 − α)/(K − 1). ∎
- With identical sources, cross-question information adds nothing, because the roles are all equal.

**Check** (`logs/od_bridge.json`): the Bayes choice lies in the arg-min-D set.

| generator | consistency | Bayes acc | min-D acc |
|---|---|---|---|
| OD1 assumptions (all reliable, α = 0.8) | **1.000** | 0.975 | 0.975 |
| unequal accuracies (0.9, 0.7, 0.55) | 0.989 | 0.958 | 0.939 |
| one correlated pair (κ = 0.7) | 1.000 | 0.962 | 0.954 |
| biased errors (bias 0.8) | 0.980 | 0.993 | 0.963 |
| T1 (reliable, noise, noise) | 0.900 | 0.744 | 0.637 |
| T4 (reliable, colluder, noise) | 0.723 | 0.741 | 0.552 |

**Counterexamples.** Each assumption's failure breaks the bridge.
- **Unequal α [P].** Weighted discrepancy Σ_s w_s·1[x_s ≠ c] with w_s = log(α_s(K − 1)/(1 − α_s)) is needed. This is the Dawid–Skene log-odds weighting.
- **Roles that are uninformative or copying.** Noise and colluders make D count non-votes and duplicate votes. T4 consistency falls to 0.72, and min-D accuracy (0.552) is below own-question Bayes (0.576).

**Determination.** Discrepancy *is* the Bayes statistic in exactly the regime where there are no relations to infer: exchangeable, equally reliable, independent sources. Every relation the latent-reliability family adds (who is reliable, who copies) is precisely what O/D lacks. This sharpens the atlas Obstacles 1–2 into a characterisation: the bridge holds iff the source semantics are trivial.
- The "only if" direction is shown here only for the families tested. It is not proved in general.

## 6. The strongest abstract causal class found (§12)

**Class 𝒦_ε (KL-close implementations).** All models with natural KL ≤ ε on the training generator N.

**Theorem K1 [P].**
- **Statement.** Members of 𝒦_ε agree on intervention signs for any edit family with density ratio W, except on (margin-tail) + O(Wε/τ²) of edits. This is C4 applied pairwise through Bayes.
- **What else the class implies.** Nothing about internal structure, off-support behaviour (C7, PP2) or abstraction invariances that N does not exercise.

**Witnesses that K1 is the most that holds across observed classes [E, X].**
- Every observed class (A, A′, B, C) is in 𝒦_0.002.
- Off support they differ: the local index ranges 0.05–0.87. They also differ in the uniform-shift invariance of their T1 parents, and in tag sufficiency (0.47 vs 0.77).

**Class 𝒦_rel (relative-count abstraction).** Models whose output at q depends on X only through (X_q, relative leave-q counts). Under T1, Bayes is in 𝒦_rel (§4). Under T4 it is not, because the colluder's count is absolute.

**Theorem K2 [P].**
- **Statement.** A model in 𝒦_rel that is KL-exact on T1 has a uniquely determined response to all uniform shifts: none.
- **Empirical status.** T1 Transformers are approximately in 𝒦_rel (ratio ≤ 0.043, seed 12: 0.14) [E].
- **Counterexample [X].** A model that is KL-exact on T1 need not be in 𝒦_rel. Take Bayes plus a perturbation supported on uniformly shifted inputs of measure zero under T1. Membership in 𝒦_rel is therefore an *abstraction property*, not implied by behaviour on N.

## 7. Limits (what is not covered)

- All theorems are for three known (or six hidden) sources, K = 3, and the role families named.
- No theorem connects any of these classes to a training dynamic: *why* a seed lands in class C is open.
- The PP2 internal predictor is empirical, and is being tested prospectively (P-A).
- Multi-dimensional versions of U (groups of size 3, vector counts) were not proved. Only the scalar-count version is covered.
