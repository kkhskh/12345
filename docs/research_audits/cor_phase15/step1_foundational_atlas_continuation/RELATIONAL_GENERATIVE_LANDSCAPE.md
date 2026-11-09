# Relational generative landscape (mandate §9, §10)

**Status.** All results are exact-Bayes computations, from Monte Carlo over episodes with exact posteriors. No network is involved, so nothing here is a neural claim. Any failure of identification reported below is **statistical**: no estimator can do better.

**Code.**
- Engine: `src/roles_engine.py`. It handles general per-group role generators: biased errors, informative colluders, mixtures over reliable accuracy, any configuration prior.
- Scripts: `scripts/landscape_phase.py`, `scripts/landscape_checks.py`, `scripts/unknown_groups.py`.
- Logs: `logs/landscape/{phase,checks,unknown_groups}.json`.

**Bug note.** The first engine version computed leave-one-out as `total − own`. At ρ = 1 that gives inf − inf. It was replaced by a direct leave-one-out sum, which is the same fix the atlas made. Only the first run was affected, and its output was overwritten.

## 1. Error structures and phase diagram (§10)

### 1.1 The reliable↔colluder nonidentifiability line (exact; atlas I2 extended)

**Setting.** Groups (reliable α, colluder ρ, noise). Uniform prior over the 6 role assignments. K = 3, Q = 6.

**Prediction (I2).** At ρ*(α) = (K·a_rel − 1)/(K − 1), where a_rel = α² + (1 − α)²/(K − 1), the episode law is invariant under swapping the reliable and colluding roles.

**Check** (`checks.json`). E[P(true reliable group is reliable)] versus E[P(true colluder is called reliable)]:

| α | ρ* | at ρ* − 0.15 | **at ρ*** | at ρ* + 0.15 |
|---|---|---|---|---|
| 0.7 | 0.302 | 0.443 vs 0.327 | **0.382 vs 0.383** | 0.389 vs 0.360 |
| 0.8 | 0.490 | 0.497 vs 0.370 | **0.431 vs 0.433** | 0.462 vs 0.394 |
| 0.9 | 0.723 | 0.557 vs 0.400 | **0.479 vs 0.479** | 0.569 vs 0.388 |

The symmetry is exact at ρ* only, to Monte Carlo error.

**Bayes accuracy along ρ** (`phase.json`; own-question Bayes ≈ 0.58 at α = 0.8):

| α | ρ = 0 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| 0.8 | 0.740 | 0.684 | **0.619** | 0.637 | 0.706 | 0.736 |
| 0.9 | 0.873 | 0.835 | 0.767 | **0.670** | 0.724 | 0.785 |

The accuracy minimum sits at the grid point nearest ρ* for α = 0.8 and 0.9. For α = 0.7 it is shifted to ρ = 0.4. There the reliable pair (a = 0.535) is itself close to noise (a = 0.333), so the identification failure is three-way.

**Consequence.** A model's failure to separate reliable from colluding sources near ρ* is not a neural failure. Any training distribution that straddles ρ* forces an averaging solution.

### 1.2 "A copying pair is one vote" (exact; derived, then checked)

**Setting.** Groups (reliable α = 0.8, informative colluder with value accuracy a_c, noise). In a conflict, the reliable pair agrees on v and the colluder pair copies c ≠ v.

**Prediction.** A copying pair contributes one likelihood ratio, 2a_c/(1 − a_c). The agreeing reliable pair contributes the square of one source's ratio, (2α/(1 − α))². So Bayes follows the colluder iff a_c > a_c* = 64/66 = **0.970**.

| a_c | 0.90 | 0.95 | 0.96 | **0.97** | 0.98 | 0.99 |
|---|---|---|---|---|---|---|
| P(Bayes follows colluder in conflict) | 0.117 | 0.126 | 0.133 | **0.922** | 0.920 | 0.924 |

The phase boundary is sharp and sits exactly at the predicted value.

**Corollary.** The atlas statement U2 ("no reversal without a higher-agreement role") concerns *reliability posteriors*. At the decision level the relevant comparison is between evidence *per independent vote*. A colluder that is more accurate than the honest sources (a_c = 0.9 > α = 0.8) still loses conflicts. Biased honest errors (80% of errors on (y + 1) mod K) change accuracy (for example 0.764 → 0.839 at a_c = 0.2) but not this boundary (`phase.json`, rows 4).

### 1.3 Theorem S without (O): an explicit numerical counterexample

**Setting.** Two answer-dependent groups: reliable α = 0.9, reliable α = 0.6, and noise. Episodes with identical leave-q counts *and* identical own-question reports still differ in their Bayes posterior at q:
- maximum spread 0.458 over 3,082 classes;
- median spread 0.0009.

**Control.** T4 gives spread 2·10⁻¹⁵. So (O) is needed, not merely sufficient for the proof: the two reliable groups' *cross-agreement* carries information.

### 1.4 Theorem U: a scope correction (mixture over the reliable accuracy)

P(group 0 is reliable | its count n), averaged over the other groups' counts:

| generator | n = 0 … 6 | quasi-concave? |
|---|---|---|
| single α = 0.8, others noise | 0.009, 0.033, 0.116, 0.339, 0.667, 0.888, 0.970 | yes (monotone) |
| mixture α ∈ {0.4, 0.97}, others noise | 0.189, 0.196, 0.200, 0.206, 0.280, 0.811, 0.992 | yes |
| mixture α ∈ {0.45, 0.99} with a colluder (ρ = 0.6) | 0.296, **0.313**, 0.304, 0.244, **0.118**, 0.154, 0.737 | **no** (two local maxima) |

**Correction.**
- Theorem U holds *per role*.
- When "reliable" is a union of roles with different agreement rates (an episode-level mixture over α), the posterior of the union is a sum of quasi-concave functions. That sum need not be quasi-concave.
- The failure needs an intermediate-agreement competitor, here the colluder at a = 0.73, between the two reliable types (0.35 and 0.98).
- The atlas proof is correct for its stated assumptions. Its *interpretation* ("the response to more agreement never turns more than once") holds only when every reliable sub-type is one role.

### 1.5 Argmax sensitivity and the ρ → 1 boundary

**Setting.** Fix 3,000 episodes sampled at ρ = 0.6 and recompute the Bayes argmax as the *assumed* ρ moves across a 0.05 grid. The fraction of answers that flip per step:
- 1–5% in the interior;
- **13% and 11%** at the steps crossing ρ* ≈ 0.49 (the role-swap line);
- **52% at ρ → 1.**

**Interpretation.** ρ = 1 is a singular point of the family.
- The posterior is continuous in ρ, but its log-odds move like n·log(1 − ρ). The answers are therefore maximally sensitive to the copy rate as ρ → 1.
- At ρ = 1 every colluder disagreement becomes impossible, which is a support change.
- T4, the training distribution, sits exactly at this point. That is one structural reason why behaviour at T4 constrains behaviour at ρ < 1 or p_c = 0 so little. The atlas PP2 results are consistent with this, but this does not show it is their cause.

## 2. Unknown groups (§9)

### 2.1 Hidden pairing of six sources (exact over the 15 pairings)

| generator | Q = 3 | Q = 6 | Q = 12 |
|---|---|---|---|
| T1-perm: mass on true pairing (MAP correct) | 0.10 (0.17) | 0.15 (0.21) | 0.22 (0.27) |
| T4-perm | 0.46 (0.57) | 0.64 (0.73) | 0.81 (0.87) |

**T1-perm is bounded at 1/3 for every Q (atlas I1).** The four noise sources can be re-paired in 3 ways with identical likelihood. The *full* pairing is therefore statistically nonidentifiable under T1. Only the reliable pair is identifiable. Any learned model's "failure to recover the pairing" under T1-perm is, in this part, not a neural failure.

### 2.2 Unequal and larger groups

**Setting.** One colluding block of unknown membership and unknown size (all blocks of size 2–4 among 6 sources, uniform prior). All other sources are independent reliable sources (α = 0.8).

| block size | ρ | Q = 3 | Q = 6 | Q = 12 |
|---|---|---|---|---|
| 2 | 1.0 | 0.70 | 0.97 | 1.00 |
| 3 | 1.0 | 0.75 | 0.98 | 1.00 |
| 4 | 1.0 | 0.82 | 0.99 | 1.00 |
| 2 | 0.5 | 0.20 | 0.42 | 0.72 |
| 3 | 0.5 | 0.21 | 0.38 | 0.61 |
| 4 | 0.5 | 0.28 | 0.50 | 0.76 |

(Entries are the posterior mass on the true block.)

- Larger blocks are *easier* at ρ = 1: more sources must coincide.
- At ρ = 0.5 the posterior spreads to sub-blocks and super-blocks. For size 3 at Q = 6: 0.08 on strict subsets and 0.06 on strict supersets.
- Size is identifiable only as fast as membership.

### 2.3 Overlapping groups (misspecified hypothesis class)

**Setting.** Source 2 copies block {0, 1}'s value or block {3, 4}'s value, with probability ½ each per question. The hypothesis class contains one block of any membership, but no overlapping structures.

| Q | {0,1} | {3,4} | {0,1,2} | {2,3,4} |
|---|---|---|---|---|
| 6 | 0.45 | 0.43 | 0.06 | 0.05 |
| 12 | 0.52 | 0.47 | 0.007 | 0.004 |
| 24 | 0.50 | 0.50 | 0 | 0 |

- The posterior converges to the two *non-overlapping* blocks and treats source 2 as an independent source.
- With a one-block hypothesis class, the overlap is not merely unidentified. It is **actively mis-explained**.
- The correct structure is outside the class, so no amount of data recovers it.
- This is the abstract form of COR failure F1: a fixed relation grammar cannot express some relations.

## 3. Summary map

| axis | what changes the Bayes computation | statistically identifiable? |
|---|---|---|
| ρ near ρ*(α) | the reliable/colluder roles become exchangeable | **no**, exactly at ρ* |
| a_c vs α | decision boundary at the one-vote threshold a_c* (0.970 at α = 0.8) | yes |
| two answer-dependent groups | counts stop being sufficient (cross-agreement matters) | yes, but needs more than counts |
| mixture over α with an intermediate competitor | the reliability response can have two peaks | yes |
| ρ → 1 | unbounded posterior sensitivity, then a support change at ρ = 1 | — |
| hidden pairing, T1 | noise re-pairing | **no** (≤ 1/3 on the full pairing) |
| block size and membership | identifiable at a rate set by ρ | yes |
| overlapping groups under a one-block class | mis-explained | **no**: outside the hypothesis class |
