# Relational generative landscape (mandate §9, §10)

**Status.** Every result here is an exact-Bayes computation (exact posteriors, with Monte Carlo over episodes unless marked exact enumeration). No network is involved, so nothing here is a neural claim. Any failure of identification reported here is **statistical**: no estimator can do better.

**Revision.** Revised after review round R2 (`reviews/ADJUDICATION_R2.md`). Pre-review text is in git history.

**Code and logs.**
- Engine: `src/roles_engine.py`. It handles general per-group role generators: biased errors, informative colluders, mixtures over the reliable accuracy, and any configuration prior.
- Scripts: `scripts/landscape_phase.py`, `scripts/landscape_checks.py`, `scripts/review_checks_landscape.py`, `scripts/unknown_groups.py`.
- Logs: `logs/landscape/{phase,checks,review_checks,unknown_groups}.json`.

**Bug notes.**
1. The first engine version computed leave-one-out as total − own. At ρ = 1 that gives inf − inf. It was replaced by a direct leave-one-out sum, and that run's output was overwritten.
2. Review R2 found that §1.5 of the pre-review text took argmax of NaN rows: posteriors that are undefined at an assumed ρ = 1. Its "52% flip" result was an artefact and is **retracted**. The script is now NaN-guarded.
3. Configuration order depended on the Python hash seed (`set` of tuples). It is now `sorted`.

`phase.json` was regenerated after fixes 2 and 3. Its Monte Carlo values differ slightly from the pre-review text.

## 1. Error structures and phase diagram (§10)

### 1.1 The reliable↔colluder role-swap line (atlas I2, proved; checked here along the line)

**Setting.**
- Groups: reliable (accuracy α), colluder (copy rate ρ, uninformative common value), noise.
- Uniform prior over the 6 role assignments; uniform Y; symmetric errors; K = 3, Q = 6.

**Theorem (atlas I2) [P].**
- Let ρ*(α) = (K·a_rel − 1)/(K − 1), with a_rel = α² + (1 − α)²/(K − 1).
- At ρ*, the per-question law of a reliable pair and of a colluding pair is the same: "agree with probability a; agreed value uniform; otherwise a uniform ordered distinct pair".
- So the episode law is invariant under swapping the two roles.

**Check** (`checks.json`): E[P(true reliable group called reliable)] versus E[P(true colluder called reliable)].

| α | ρ* | at ρ* − 0.15 | **at ρ*** | at ρ* + 0.15 |
|---|---|---|---|---|
| 0.7 | 0.302 | 0.443 vs 0.327 | **0.382 vs 0.383** | 0.389 vs 0.360 |
| 0.8 | 0.490 | 0.497 vs 0.370 | **0.431 vs 0.433** | 0.462 vs 0.394 |
| 0.9 | 0.723 | 0.557 vs 0.400 | **0.479 vs 0.479** | 0.569 vs 0.388 |

Equality holds at ρ* to Monte Carlo error. Reviewer FM checked |Δ log L| ≤ 1.4·10⁻¹⁴ for K = 3 and 5.

**Bayes accuracy along ρ** (regenerated `phase.json`). Columns are ρ; ↓ marks the observed minimum.

| α | 0 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 0.620 | 0.561 | 0.559 | 0.563 | 0.577 | 0.598 | 0.609 | 0.632 | 0.650 |
| 0.8 | 0.749 | 0.673 | 0.659 | 0.619 | **0.615 ↓** | 0.639 | 0.656 | 0.706 | 0.747 |
| 0.9 | 0.880 | 0.829 | 0.808 | 0.763 | 0.721 | 0.671 | **0.661 ↓** | 0.722 | 0.787 |

**Correction (review ST-F9).**
- The accuracy minimum is *near* ρ* but not at it. For α = 0.8 the minimum is at 0.55–0.6, whereas ρ* = 0.49.
- The minimum of the true-reliable mass is closer to ρ*: 0.5 for α = 0.8, 0.75 for α = 0.9, and 0.35–0.4 for α = 0.7.
- The symmetry point is where the two roles are exchangeable. It is not claimed to be the accuracy minimum, which also depends on confusion with noise.

**Consequence.** A model's failure to separate reliable from colluding sources at ρ* is not a neural failure.

### 1.2 "A copying pair is one vote" (V1; symmetric errors)

**Setting.**
- Groups: reliable (α = 0.8, symmetric errors), an informative colluder whose copied value has accuracy a_c, and noise.
- Conflict: the reliable pair agrees on v, and the colluder pair copies c ≠ v.

**Statement [P].** Bayes follows the colluder iff (K − 1)·a_c/(1 − a_c) > (α/e_{c→v})². Here e_{c→v} is the probability that one reliable source reports v when Y = c. With symmetric errors, e = (1 − α)/(K − 1), so the threshold is a_c* = 64/66 = **0.970**.

**Check** (`checks.json`):

| a_c | 0.90 | 0.95 | 0.96 | **0.97** | 0.98 | 0.99 |
|---|---|---|---|---|---|---|
| P(Bayes follows colluder in conflict) | 0.117 | 0.126 | 0.133 | **0.922** | 0.920 | 0.924 |

**Correction (review ST-F3 / FM-F7).** The pre-review claim that biased honest errors "change accuracy but not this boundary" was false; that range was never probed. With bias 0.8 (80% of errors go to (y + 1) mod K), the threshold **splits by conflict direction**, exactly as the formula predicts (`review_checks.json`):

| conflict direction | predicted a_c* | observed jump |
|---|---|---|
| v = c + 1 | **0.908** | from 0.226 at 0.905 to 0.991 at 0.91 |
| v = c + 2 | **0.9988** | from 0.055 at 0.995 to 0.846 at 0.999 |

**Prior art.** This is optimal log-odds weighted voting with a deterministic copy counted once: Nitzan–Paroush 1982, Ben-Yashar–Paroush 2001 for K-ary, Dong et al. 2009 for copy discounting.

### 1.3 Theorem S without (O)

**Setting.** Two answer-dependent groups (reliable α = 0.9, reliable α = 0.6) plus noise.

**Result.** Episodes with equal leave-q counts *and* equal own-question reports differ in their Bayes posterior at q:
- maximum spread 0.438 over 3,116 classes;
- median spread 0.0009.

**Control.** T4 gives 2·10⁻¹⁵. So (O) is needed for S.

**Prior art.** This is the standard cross-annotator identifiability of latent-class models (spectral or triplet Dawid–Skene). It is a scope correction, not a new result.

### 1.4 Theorem U: scope correction (a mixture over the reliable accuracy)

**Setting.** P(group 0 has *some* reliable role | n₀, n₁, n₂). The generator mixes α ∈ {0.45, 0.99} with a colluder (ρ = 0.6) and noise.

**Exact computation** (count sufficiency plus binomial likelihoods; `review_checks.json`):
- The union posterior is **not quasi-concave in n₀ in 45 of the 49 fixed contexts (n₁, n₂)**.
- Each single-α generator is quasi-concave in **all** 98 contexts.

Example at n₁ = n₂ = 0: P = 0.333, 0.144, 0.037, 0.008, **0.0017**, 0.0019, 0.027 for n₀ = 0 … 6. The posterior falls, reaches its minimum at n₀ = 4, and rises again.

**Correction.**
- Theorem U holds **per role**.
- A union of roles with different agreement rates (an episode-level mixture over α) gives a sum of quasi-concave functions. That sum need not be quasi-concave.
- Whether an intermediate-agreement competitor is *necessary* for the failure is **unproven**. Reviewer FM's grid search found no counterexample without one.

The pre-review table averaged over contexts (Monte Carlo). Its "two local maxima" rested on a 0.009 difference and is replaced by the exact fixed-context result.

### 1.5 Argmax sensitivity along ρ (corrected)

**Setting.** Episodes are sampled at ρ = 0.6. The Bayes argmax is recomputed as the assumed ρ moves in steps of 0.05, counting only cells where the posterior is defined (`phase.json`, `argmax_sensitivity_rho_guarded`).

**Results.**
- Interior steps flip 1–5% of answers.
- The steps ending at ρ = 0.5 and 0.55 flip **12.8% and 11.4%**. Both lie just above ρ* ≈ 0.49.
- At ρ = 1 only 21.9% of cells are defined: every episode with any colluder-pair disagreement leaves the support. On the defined cells the flip rate is 6.2%.
- Refined steps near 1 (0.95 → 0.99 → 0.999) flip 2.4–3.4% (`review_checks.json`).

**Retraction.** The pre-review "52% flip at ρ → 1" and the conclusion that "T4 sits at a point of maximal sensitivity" are withdrawn. What remains true is a **support change** at ρ = 1, nothing more.

## 2. Unknown groups (§9)

### 2.1 Hidden pairing of six sources (exact over 15 pairings)

| generator | Q = 3 | Q = 6 | Q = 12 |
|---|---|---|---|
| T1-perm: mass on true pairing (MAP correct) | 0.10 (0.17) | 0.15 (0.21) | 0.22 (0.27) |
| T4-perm | 0.46 (0.57) | 0.64 (0.73) | 0.81 (0.87) |

- **Under T1-perm the posterior mass on the full true pairing is ≤ 1/3 for every Q.** The four noise sources re-pair in 3 ways with identical likelihood (atlas I1).
- Only the reliable pair is identifiable.
- A learned model's failure to recover the full pairing under T1-perm is therefore, in this part, not a neural failure.
- MAP-correct uses first-index tie-breaking, which is harmless under uniform permutations.

### 2.2 Unequal and larger groups

**Setting.**
- One colluding block of unknown membership and unknown size: all blocks of size 2–4 among 6 sources, uniform over blocks. Blocks of each size are 15, 20 and 15 in number, so the implied size prior is not uniform.
- All other sources are reliable (α = 0.8).

**Posterior mass on the true block:**

| block size | ρ | Q = 3 | Q = 6 | Q = 12 |
|---|---|---|---|---|
| 2 | 1.0 | 0.70 | 0.97 | 1.00 |
| 3 | 1.0 | 0.75 | 0.98 | 1.00 |
| 4 | 1.0 | 0.82 | 0.99 | 1.00 |
| 2 | 0.5 | 0.20 | 0.42 | 0.72 |
| 3 | 0.5 | 0.21 | 0.38 | 0.61 |
| 4 | 0.5 | 0.28 | 0.50 | 0.76 |

**Wording corrected (review ST-F17).**
- At ρ = 1 larger blocks have higher true-block mass.
- This is confounded with the number of remaining reliable sources (6 − s) and with the implied size prior.
- At ρ = 0.5 the order is non-monotone (size 3 < size 2 < size 4).
- At ρ = 0.5 the posterior spreads to sub-blocks and super-blocks. For size 3 at Q = 6: 0.08 on strict subsets and 0.06 on strict supersets.

### 2.3 Overlapping groups under a misspecified hypothesis class

**Setting.**
- Truth: source 2 copies block {0, 1}'s value or block {3, 4}'s value, with probability ½ each per question.
- Hypothesis class: exactly one colluding block of size 2–4. Every non-block source is modelled as reliable (α = 0.8), which is a second misspecification.

**Mean posterior mass over episodes:**

| Q | {0,1} | {3,4} | {0,1,2} | {2,3,4} |
|---|---|---|---|---|
| 6 | 0.45 | 0.43 | 0.06 | 0.05 |
| 12 | 0.52 | 0.47 | 0.007 | 0.004 |
| 24 | 0.50 | 0.50 | 0 | 0 |

The **mean** mass concentrates on the two non-overlapping blocks. Per episode it is likely concentrated on one of them. The true structure is outside the class (KL projection under misspecification; White 1982). This is the abstract form of COR failure F1: a fixed relation grammar cannot express some relations.

## 3. Summary map

| axis | what changes the Bayes computation | statistically identifiable? |
|---|---|---|
| ρ near ρ*(α) | the reliable and colluder roles become exchangeable | **no**, exactly at ρ* [P] |
| a_c vs α (symmetric errors) | the decision boundary at the one-vote threshold | yes |
| biased errors | the one-vote threshold splits by conflict direction | yes |
| two answer-dependent groups | counts stop being sufficient | yes, with more than counts |
| mixture over α | the union role posterior can be non-quasi-concave | yes |
| ρ → 1 | a support change at ρ = 1 (no special sensitivity before it) | — |
| hidden pairing, T1 | noise re-pairing | **no** (≤ 1/3 on the full pairing) |
| block size and membership | identifiable at a rate set by ρ | yes |
| overlap under a one-block class | mis-explained | **no** (outside the class) |
