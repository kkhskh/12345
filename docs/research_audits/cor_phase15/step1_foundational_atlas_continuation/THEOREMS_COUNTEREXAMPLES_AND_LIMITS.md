# Theorems, counterexamples and limits (continuation; mandate §2, §6, §8, §10, §12, §14)

**Labels.**
- **[P]** proof given.
- **[N]** numerical check, exact Bayes.
- **[E]** empirical, learned models.
- **[X]** counterexample.
- **[R]** claim corrected or retracted after review round R2 (`reviews/ADJUDICATION_R2.md`).

The atlas theorems (S, M, U, U2, D, D1–D2, I1–I7) are not restated. §1 corrects their scope; §2–§6 hold the new results. Pre-review text is in git history.

## 1. Scope corrections to the atlas theorems (§8)

| theorem | atlas scope | corrected scope | evidence |
|---|---|---|---|
| **S** (count sufficiency) | under (U) and (O) | (O) is *needed*, not only used. With two answer-dependent groups (α = 0.9 and 0.6), episodes with equal leave-q counts and equal own reports differ in Bayes posterior by up to 0.438 (median 0.0009). T4 gives 2·10⁻¹⁵. S is a role-posterior statement; the decision-level statistic is (X_q, n^{−q}) (J1). Prior art: cross-annotator identifiability of latent-class models | [N] |
| **U** (quasi-concavity) | per role, under (U) | holds **per role**. It **fails** for a *union* of roles with different agreement rates (episode-level mixture over α): not quasi-concave in **45 of 49** fixed contexts. Each single-α generator passes all 98 [R: now exact enumeration at fixed contexts]. Whether an intermediate competitor is necessary is unproven. U says nothing about margins; those go through D | [P, N, X] |
| **U2** | reliability posteriors | correct at the posterior level. At the decision level, compare evidence per independent vote (V1) | [P, N] |
| **D** (margin identity) | absolute continuity π′ ≪ π | unchanged. **[R]** Absolute continuity is a separate condition from S1 membership: it fails on 3,264 edits inside S1 ∩ S2 (reviewer FM-F12, not re-run). Applicability must be checked per edit | [P] |
| **D2** (no universal sign) | — | strengthened by C5 and C6 below | [P, X] |
| **I2** (role swap at matched agreement) | proved; checked at two settings | **[R] labelled [P]**. Exact invariance of the episode law along ρ*(α) = (K·a_rel − 1)/(K − 1), checked for α = 0.7, 0.8, 0.9 (and by reviewer FM for K = 5) | [P, N] |
| **I5 / Shapley** | head-level and summed shares not invariant (withdrawn, R2-c) | replaced by R1 (§3) | [P, N] |

## 2. Calibration, support and the sign of interventions (§2)

Full statements are in PP1_CALIBRATION_AND_INTERVENTION_THEOREMS.md.

| id | result | type | review status |
|---|---|---|---|
| C1 | \|Δm\| > \|e(x)\| + \|e(x′)\| ⇒ the sign agrees | [P] | an identity; "never violated" is a code check. The *empirical* content is the coverage: 98.6–99.7% of scored edits |
| C2 | log-probability sup-norm version | [P] | holds |
| C3 | TV version with π_min | [P] | holds |
| C4 | P(fail) ≤ P_E(\|Δm\| ≤ b_τ + b_τ′) + W·ε/τ² (density ratio W, natural KL ε, bounded-KL implicit) | [P] | **[R] valid but vacuous on PP1**: at ε ≈ 10⁻³ the right-hand side is ≥ 1 for every τ. It does **not** explain the observed 0.996+ agreement |
| C5 | if m̂ = φ(m) at both endpoints, φ strictly increasing, with c1 and c2 fixed, then signs are exact. Temperature and ECE are irrelevant | [P, N] | holds as stated. Per-probability recalibrations (for example uniform mixing) are outside it |
| C6 | **[R] existence form.** For the PP1 generator and edit family, some model has Bayes accuracy and sign agreement 0.22 (CX1). It is not "for every generator": a ranking-preserving distortion gets order-swapping edits right automatically | [X] | corrected quantifier |
| C7 | no average-KL guarantee without (DR) | [P] | **[R]** needs a bounded-KL assumption. A multi-cell edit family realises it (reviewer FM: 46% of endpoints corrupted, agreement 0.875). The authors' high-ratio family (W ≥ 7.7) leaves the *learned* models at 0.963–0.980 unfiltered (PP1 §4) |

## 3. Split-invariant attribution (§6)

**R1 [P, N].**
- **Statement.** Shapley values of the quotient game whose players are exact attention-pattern classes are invariant under exact head splits.
- **Prior art.** This is the quotient-game step of the Owen value with exact clones merged (Owen 1977; Knudsen–Østerdal 2012 on split-proofness).
- **[R] Gauge part.** At the site used (head outputs after W_O), invariance under the within-head gauge W_V G, G⁻¹ W_O is vacuous: head-level Shapley is gauge-invariant too.

**L1 [R].**
- **Original.** "No head attribution can be invariant to value-equal mass moves and still separate heads with different patterns".
- **Status.** Tautological under the reading used.
- **What it leaves open.** A non-trivial invariant may exist: classes defined by attention mass over value-equivalence classes of positions. This is OF-5.

## 4. Generative-level results (§9, §10)

**V1 (one copy = one vote) [P, N].**
- **Setting.** Known roles; a reliable pair agrees on v; a pair copies c ≠ v deterministically, with value accuracy a_c.
- **Statement.** Bayes prefers c iff (K − 1)·a_c/(1 − a_c) > (α/e_{c→v})². Here e_{c→v} is one honest source's probability of reporting v when Y = c.
- **Symmetric errors.** e = (1 − α)/(K − 1), so a_c* = 0.970 for α = 0.8.
- **[R] Biased errors.** With bias 0.8 the threshold splits by direction, to 0.908 and 0.9988. Both were verified (LANDSCAPE §1.2).
- **Prior art.** Nitzan–Paroush 1982 and Ben-Yashar–Paroush 2001 (weighted voting); Dong et al. 2009 (copy discounting).

**I1′ [P, N].** Under T1 with a hidden pairing, the posterior mass on the full true pairing is ≤ 1/3. The noise-source re-pairings have equal likelihood (atlas I1). Reviewer FM-F17 notes the sketch shows equal likelihood but not the full bound under every prior. The bound holds numerically: 0.10, 0.15, 0.22 at Q = 3, 6, 12.

**ρ = 1 support change [R: retracted as "singularity"].** At ρ = 1, configurations that make a disagreeing pair the colluder leave the support. That is the only true statement. The pre-review claims ("unbounded sensitivity", "52% of decisions flip") rested on a NaN artefact and are **withdrawn**. Measured on defined cells, flip rates near ρ = 1 are 2.4–6.2%, like interior steps.

**Misspecified overlap [X, N].** With a one-block hypothesis class and an overlapping truth, the mean posterior mass concentrates on the two non-overlapping blocks (0.50 / 0.50 at Q = 24) and gives zero mass to the true structure (KL projection; White 1982).

**Relative-count invariance under T1 [P].**
- **Statement.** Adding one agreement to every group in a question k ≠ q leaves P(Y_q | X) unchanged under T1.
- **Proof.** Every configuration's likelihood is multiplied by the same factor.
- **Model evidence [E].** T1 models show invariance ratio 0.02–0.04 (seed 12: 0.14). Their uniform-shift KL is about 2–4× a null-edit noise floor of 1–5·10⁻⁴ (reviewer CA-F12).

## 5. O/D: the restricted bridge (§14) [R: corrected]

**OD1 ("if" direction) [P].**
- **Assumptions.** Sources are conditionally independent given Y, Y is uniform, all share one accuracy α > 1/K, errors are symmetric, and D_c(x_q) = #{s : x_{q,s} ≠ c}.
- **Statement.** log P(c | x_q) = const − D_c · log(α(K − 1)/(1 − α)). So the discrepancy ranking equals the Bayes ranking.

**OD1′ (the exact condition for independent symmetric sources) [P].** Let w_s = log(α_s(K − 1)/(1 − α_s)). The D-ranking equals the Bayes ranking on every report pattern iff w(P) < w(R) for all disjoint source sets P and R with |P| < |R| and |P| + |R| ≤ S.
- This is the weighted-versus-simple-majority coincidence condition: Nitzan–Paroush 1982; Ben-Yashar–Paroush 2001; Taylor–Zwicker 1999.
- A sufficient condition is w_max/w_min < (m + 1)/m, with m = ⌊(S − 1)/2⌋. For S = 6 that is a ratio below 3/2.
- Proposed by reviewer NV-A1. The authors checked the derivation, but not the enumeration.

**[R] Withdrawn.** "The bridge holds iff the source semantics are trivial". This is false.
- Unequal accuracies inside the band give an exact bridge (reviewer enumeration: 0/729 violations at accuracies 0.78–0.90).
- The package's own log shows consistency 1.000 for one correlated pair.

**Check** (`logs/od_bridge.json`): the Bayes choice lies in the arg-min-D set.

| generator | consistency | Bayes acc | min-D acc |
|---|---|---|---|
| OD1 assumptions | **1.000** | 0.975 | 0.975 |
| unequal accuracies (0.9, 0.7, 0.55); weight ratio 3.23, outside the band | 0.989 | 0.958 | 0.939 |
| one correlated pair (κ = 0.7) | 1.000 | 0.962 | 0.954 |
| biased errors (bias 0.8) | 0.980 | 0.993 | 0.963 |
| T1 (reliable, noise, noise) | 0.900 | 0.744 | 0.637 |
| T4 (reliable, colluder, noise) | 0.723 | 0.741 | 0.552 |

**Determination (corrected).**
- Unweighted discrepancy is the Bayes statistic exactly when the log-odds weights satisfy OD1′. Equal reliability is one such case.
- The bridge is broken by roles with weight zero (noise) or with dependence that violates the product form (copying, collusion). These are precisely the relations the latent-reliability family adds.
- The repaired O/D is weighted voting with semantics-derived weights, which is established Bayesian truth discovery.

## 6. The strongest abstract causal class found (§12)

**Class 𝒦_ε (KL-close implementations).** All models with natural KL ≤ ε on the training generator N.

**K1 [P; R: quantitative force withdrawn].**
- **Statement.** Members of 𝒦_ε agree on intervention signs for any edit family with density ratio W, except on the C4 bound.
- **Status.** On PP1 that bound is vacuous at the observed ε.
- **Empirical counterpart [E].** What *is* observed is that pointwise margin errors are below \|Δm\| for 98.6–99.7% of edits. That is a property of these models' pointwise accuracy, not a consequence of ε.

**Witnesses that only Bayes-closeness is shared [E, X].**
- Every observed class (A, A′, B, C) has natural KL ≤ 0.002.
- They differ off support (local index 0.05–0.87), in tag sufficiency (0.47 vs 0.77) and in pattern-mediation of the tag (frozen-pattern necessity 0.02 for s10 vs 0.40 for s12).

**K2 [R: replaced].**
- **Withdrawn.** The pre-review K2 and its counterexample are withdrawn. T1 has full support, so a model that is KL-exact on T1 equals Bayes on every input and inherits all its invariances. The counterexample was impossible.
- **Replacement (review NV-B4) [P].** On a full-support generator N, let W_s be the density ratio of shifted to natural inputs. Then E_shift[KL(Bayes ‖ model)] ≤ W_s · ε.
- So T1 models' uniform-shift invariance is a **Bayes-closeness consequence, not an abstraction property**.
- Abstraction questions arise only for generators with *partial* support (T4), which is where PP2 locates them.

## 7. Limits (what is not covered)

- All results are for three known (or six hidden) sources, K = 3, and the role families named.
- No theorem connects any class to a training dynamic.
- The PP2 internal predictor is empirical. It is tested prospectively with low power (P-A annotation).
- Multi-dimensional versions of U (groups of size ≥ 3) were not proved.
- **Proposition SR (interior support rule; review NV-B2) [sketch; not independently verified].**
  - **Statement.** Under T4 (ρ = 1), take an edit that keeps the edited group below full agreement at both endpoints. Then "more agreement helps the group's own candidate" has the Bayes sign.
  - **Sketch.** The group has zero colluder probability at both endpoints, and its reliable-versus-noise likelihood ratio rises with agreement (a_rel > a_noise). Theorem D then gives the sign.
  - **Empirical check.** 1.000 on 16,832 interior edits (COR_TEN §1).
  - **Failure at the boundary.** It fails at the full-agreement boundary (0.111). This is the paradox of unanimity (Gunn et al. 2016).
