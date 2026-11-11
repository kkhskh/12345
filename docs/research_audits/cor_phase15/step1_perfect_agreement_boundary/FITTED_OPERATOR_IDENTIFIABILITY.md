# Fitted-operator identifiability

**Status.** EXPLORATORY. This analysis was not preregistered, and no confirmatory claim rests on it.

**Code and outputs.** Script `scripts/fitted_operators.py`. Outputs:
- `logs/fitted_operators_dev.json`
- `logs/fitted_operators_conf.json` and `logs/fitted_operators_conf_B.json` (16 cohort parents at step 4000)
- `logs/fitted_operators_summary.json`

**Targets.** Each model's centred logits on 2500 base episodes and their F45, F34 and F23 edits.

**Families.** Each is fitted by least squares on centred logits, given the exact leave-q counts and the exact J:

| family | parameters | definition |
|---|---|---|
| gain | 1 | τ · exact Bayes |
| mix | 11 | Bayes mixture, log w = φ_rel(n_g) + φ_col(n_h) (Investigation V parameterisation) |
| logpool | 11 | normalised-weight logarithmic pool of the per-configuration predictives, same weights |
| vote | 6 | count-weighted source voting |

## 1. Fit and predicted boundary observable (cohort, step 4000, mean of 8 parents)

| | observed B45-R1 | gain: R² / predicted B45-R1 | mix | logpool | vote |
|---|---|---|---|---|---|
| T1 | −0.047 | 0.92 / −0.018 | 0.95 / −0.049 | 0.91 / +0.301 | 0.74 / −2.78 |
| P-count | **−0.742** | 0.80 / −0.053 | 0.85 / **−0.382** | 0.84 / −0.188 | 0.73 / −2.38 |

**Readings**
- For T1, the best-fit Bayes mixture reproduces the boundary observable. Gain-only Bayes nearly does as well, and log pooling
  does not.
- For P-count models, **no fitted family reproduces the observable**. The best-fitting mixture recovers about half of the R1
  excess (−0.38 vs −0.74). Log pooling and gain recover less, and voting overshoots grossly.
- The whole-surface least-squares fit is dominated by base and below-boundary episodes. It trades away the small R1 set.
  This is the mechanism of corrected Proposition 1 seen in a fit: good average fit, wrong boundary behaviour.

## 2. The s parameter

s = φ_rel(5) − φ_rel(4). The exact-Bayes value is log RHO = 1.36.

| | ŝ mean (range) | 2 %-flat profile set (mean bounds) | profile s-values rejected by the observable (\|pred − obs\| > 0.1 at every s) |
|---|---|---|---|
| T1 (cohort) | 0.76 (0.34 – 1.02) | [0.38, 1.19] | 0/8 parents |
| P-count (cohort) | −0.44 (−1.00 – 0.09) | [−1.38, 0.56] | 2/8 parents: no s in [−2, 4] reproduces B45-R1 within 0.1 |
| T1 (dev) | 0.43 | [−0.02, 0.91] | 0/11 |
| P-count (dev) | −0.84 | [−1.80, 0.20] | 2/11 |

**Readings**
- The s ordering T1 > P-count of Investigation V **replicates** in the new cohort, as a fitted quantity.
- s is **weakly identified**. The flat sets are 0.8–1.9 units wide. They overlap across arms for several parents: for example,
  P81 [−0.75, 0.75] and P82 [−0.75, 1.0] overlap the T1 sets.
- An s-interpretation of the P-count boundary behaviour is **rejected** on two grounds:
  1. at the best-fitting s, the predicted boundary observable is half the observed one;
  2. for 2/8 parents no s reproduces it at all.
- The fitted s ordering points in the right direction, but it is not an adequate description of the decision difference. The
  observables (OBSERVABLE_BOUNDARY_RESPONSES.md) are the result; s is not.

## 3. Relation to theory

- **T3(b).** The F45 response depends jointly on s and the gate increment φ_col(5) − φ_col(4). Profiling s with the gate free
  is what produces the wide flat sets.
- **Proposition 3′.** Null-invariance cannot pin s down.
- **T4(b).** Gain-only fits cannot produce the D-R1 pattern. Consistent with that, the gain family predicts B45-R1 ≈ −0.05 for
  P-count models.

No operator in these four families is claimed to be "the algorithm" of either curriculum.

## 4. Post-review additions [R5-5]

These are exploratory reviewer checks: `reviews/checks/R5_pa_discount_fit.py` and `R5_response_weighted_mix.py`.

- **Mixture plus a regime-insensitive "5/5 discount" term.** The fitted discount has the wrong sign in 8/8 P-count models,
  and the fit predicts a *positive* R1 excess (+0.11 to +0.41) against −0.51 to −1.00 observed. A uniform perfect-agreement
  penalty therefore does not describe the P-count behaviour either.
- **Bayes mixture fitted with weight on the edit responses.** It still reproduces only about half the R1 effect (−0.39 vs
  −0.74). The "half" in §1 is therefore not just an artefact of whole-surface weighting.
- **What the families cannot do.** None of them combines a deep R1 discount with near-Bayes R3 behaviour, and the P-count
  models do both. An operator that fits would have to condition on the regime itself, i.e. on whether g's vote agrees with
  the existing perfect group. That is a relabelling of the observable, not an explanation.

The observable is therefore not implied by Bayesian approximation (exact or gain-scaled) or by any of the fitted pooling
operators examined.
