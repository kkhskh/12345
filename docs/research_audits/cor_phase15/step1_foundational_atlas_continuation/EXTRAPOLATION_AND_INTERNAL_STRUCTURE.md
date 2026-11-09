# Extrapolation and internal structure: PP2 (mandate §3)

**Status.** All development analyses here are post hoc relative to the frozen atlas PP2. The prospective test is P-A, frozen in PROSPECTIVE_PREDICTIONS.md (commit `dc1dd3b`) before any held-out model existed. Its outcome is in §5.

**Revision.** §3 was revised after review round R2.

**Scripts and logs.**
- Scripts: `scripts/pp2_dev.py`, `src/pp2_sig.py`, `scripts/pp2_mechanism.py`, `scripts/pp2_probe_control.py`, `scripts/pa_evaluate.py`.
- Logs: `logs/pp2_dev.json`, `pp2_frozen_fits.json`, `pp2_mechanism.json`, `pp2_probe_control.json`, `pa_outcome.json`.

## 1. Question, hypotheses, observable

**Question.** Ten models are KL-close to the T4 Bayes posterior on T4 (natural KL ≤ 0.002). Why do they extrapolate differently to the p_c = 0 generator (no colluder), and can the direction be predicted from structure measured *without* p_c = 0 data?

**Primary observable.** The **local index**, defined on the pp2b quantity:
- the 6000 p_c = 0 episodes of stream 6,200,002;
- the cases where the global plug-in (T4 Bayes at ρ = 0.999) and the local factorised plug-in disagree;
- the fraction of those cases in which the model sides with the local plug-in.

[R] The global plug-in's ρ = 0.999 is part of the target's definition. Its argmax moves on about 0.9% of cells between ρ = 0.99 and 0.9999 (reviewer ST-F13).

**Competing hypotheses.**
- **H-arch.** The direction is fixed by architecture: pooling goes global, attention goes local. This is the Xu et al. 2021 type of explanation.
- **H-struct.** An internal structural quantity, measurable on T4-derived inputs, predicts the direction *within* an architecture.
- **H-none.** The direction is seed noise, unpredictable from anything measured before the shift.

## 2. Support versus distribution change (§3.1)

**Setup.** 8.7% of p_c = 0 episodes lie inside T4 support, meaning some group agrees in all six questions. On them T4 Bayes is defined and assigns that group the colluder role.

| model | local index, inside T4 support | agreement with global plug-in, inside | local index, outside T4 support |
|---|---|---|---|
| TF s0, s1, s2, s10, s11, s12, s13 | 0.014–0.087 | 0.88–0.91 | 0.31–0.89 |
| DeepSets s0, s1, s2 | 0.070–0.084 | 0.88–0.89 | 0.05–0.17 |

(n = 358 inside and 13,103 outside, global/local-disagreement cases per model.)

**Finding.**
- Inside T4 support, *every* model follows the training-distribution Bayes answer, even though it is wrong for p_c = 0. Agreement with the true Bayes answer there is only 0.33.
- All of the between-model divergence is **off support**.
- So the distribution change matters only through the support change. On the support, a changed prior produces no implementation differences. This is expected from KL-closeness, but it is now measured.

## 3. The internal reason (§3.2) [R: revised after review]

### 3.1 Colluder-code persistence under degradation

Developed on T4 episodes whose colluder agreement was broken in j questions; these episodes are off support. CCP_j is the probability that a probe fitted on natural T4 assigns to the true colluder.

| model | CCP_1 | CCP_3 | local index |
|---|---|---|---|
| DeepSets (3 models) | 0.79 | 0.18 | 0.05–0.16 |
| TF s12 (class C) | 0.72 | 0.19 | 0.31 |
| TF s10 | 0.39 | 0.29 | 0.87 |

Global models keep the probe's colluder assignment after one break and then *reassign* it, with CCP_3 below chance. Local models lose it immediately. ("Colluder code" here means the probe's output, not a validated variable; see §3.3.)

### 3.2 Which group the off-support code selects

| model group | probe picks the top-agreement group | local index |
|---|---|---|
| DeepSets | 1.00 | 0.05–0.16 |
| TF s2, s12, s13 | 0.71–0.83 | 0.31–0.53 |
| TF s0, s1, s10, s11 | 0.40–0.61 | 0.59–0.87 |

Spearman correlation with the local index: −0.89 over all 10 models, and −0.71 within Transformers (n = 7, p ≈ 0.07).

### 3.3 Probe control (review CA-F11)

The same probe pipeline on raw input statistics:
- graded counts: 1.00;
- the saturated "agrees in all six questions" indicator: 0.39.

**Revised reading.** The indices measure *which agreement statistic the representation carries linearly*. They are not a constraint-enforcing colluder code. The descriptive hypothesis that survives:

> Global extrapolation goes with representations that carry **graded** agreement counts. Local extrapolation goes with a **saturated** full-agreement detector, which is silent off support, so no group is discounted.

This hypothesis is post hoc and statistically weak within Transformers. It has a probe-free test (OPEN_FOUNDATIONAL_MAP OF-4).

### 3.4 Negative results

- Every model "commits" to some colluder off support (max-probability 0.86–0.99), so commitment does not discriminate (ρ_s = 0.25).
- Natural accuracy, KL and ECE do not predict the local index:

| predictor | development leave-one-out MAE |
|---|---|
| natural accuracy | 0.34 |
| natural KL | 0.23 |
| ECE | 0.30 |
| constant | 0.27 |
| random feature | 0.29 |

## 4. The prospective predictor (§3.3)

**Freeze.** Frozen at commit `dc1dd3b`, before any held-out model existed. The definitions are in PROSPECTIVE_PREDICTIONS P-A:
- the internal predictor A1 (CCP_3);
- the behavioural minimal probe A2 (LPI_deg);
- seven baselines, including architecture, accuracy, KL, ECE, constant, random and ridge.

**Development leave-one-out MAE:**

| predictor | MAE |
|---|---|
| A1 | 0.136 |
| A2 | 0.041 |
| architecture | 0.147 |

**Post-freeze caveats (review; the frozen text is unchanged).**
- A-priori pass probabilities: P-A1 ≈ 0.53 and P-A2 ≈ 1.
- A2 is a near-proxy of the target.
- The predictor inputs lie off support.
- With n = 4 held-out models, a P-A1 pass or fail is weak evidence in either direction.

## 5. Outcome

**Provenance.** Held-out models TF s14, s15 and DeepSets s3, s4. Frozen code and fits, verified by hash. Log: `logs/pa_outcome.json`; full table in PROSPECTIVE_PREDICTIONS (P-A: OUTCOME).

| model | observed local index | A1 (internal) prediction | A2 (behavioural) prediction | architecture prediction |
|---|---|---|---|---|
| TF s14 | 0.356 | 0.495 | 0.385 | 0.617 |
| TF s15 | 0.569 | 0.385 | 0.510 | 0.617 |
| DS s3 | 0.071 | 0.168 | 0.072 | 0.107 |
| DS s4 | 0.087 | 0.173 | 0.105 | 0.107 |

**Verdicts.**
- **P-A1 FAILED.** The internal predictor's MAE is 0.126 against architecture's 0.091, and it orders the two held-out Transformers backwards.
- **P-A2 passed** (MAE 0.026), but it is a near-proxy.
- **P-A3 passed**: the support result replicates (inside T4 support the local index is 0.042–0.089).
- **P-A4 passed**: DeepSets are global (0.071, 0.087).

**Answer to the §1 hypotheses on present evidence.**
- **H-arch** is supported for the DeepSets-vs-Transformer direction: 5/5 DeepSets are global, against Transformers at 0.31–0.87.
- **H-struct** has no prospective support. The one internal predictor frozen failed.
- **Within Transformers.** On two held-out Transformers and seven development ones, the spread is tracked by a behavioural off-support probe (A2: MAE 0.04 in development, 0.03 held-out). A2 is a near-proxy of the target. No measured internal quantity tracks it.
- **H-none** is weakened at the behavioural level, on n = 2 held-out Transformers [R3-F8]. It is not rejected at the internal level.

**What the PP2 shift reveals about internal computation (mandate question 3).**
- On the support it reveals nothing: all models follow the training Bayes answer there.
- Off the support it reveals *which agreement statistic a representation carries*: graded counts versus a saturated full-agreement indicator (§3.3). This hypothesis has not been confirmed prospectively.

