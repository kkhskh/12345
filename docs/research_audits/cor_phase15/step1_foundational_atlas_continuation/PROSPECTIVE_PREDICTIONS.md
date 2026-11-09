# Prospective predictions (Stage V)

Rules:
- Each prediction is frozen in a git commit before the evaluation it concerns is run.
- Later sections append; frozen text is never edited.
- Outcomes go in the "Outcome" subsection added after evaluation, together with the commit hash of the freeze.
- Development material is labelled post hoc.

---

## P-A: structural prediction of off-support extrapolation (mandate §3), held-out models

**Frozen:** 2026-11-09, before any held-out model existed. At freeze time `checkpoints/heldout/` was empty and training was in progress.

**Development (post hoc; not confirmation).**
- Data: the 10 existing T4 models, in `logs/pp2_dev.json`.
- Code: `scripts/pp2_dev.py` and `src/pp2_sig.py` (sha256 2b91afcd…).
- Fitted coefficients: `logs/pp2_frozen_fits.json` (sha256 498c0e60…).

**Target.** The p_c = 0 local index: in global-vs-local plug-in disagreements, the fraction where the model sides with the local factorised plug-in. Stream 6,200,002, 6000 episodes. This is the atlas pp2b quantity.

**Held-out models.**
- Transformers: T1 → T4 curriculum with the original protocol (T1 6000 steps, then T4 3000), seeds 14 and 15.
- DeepSets: T4 trained from scratch, 6000 steps, seeds 3 and 4.
- Total: 4 models.
- Seeds 16 and 5 were dropped for compute before training started (`logs/COMPUTE_LOG.txt`).

**Predictors.** Every signature is computed on the held-out model using T4-derived inputs only. Each is mapped to a predicted local index with the frozen linear fit.

| id | predictor | kind | development LOO MAE |
|---|---|---|---|
| A1 | CCP_3: probe probability of the true colluder after its agreement is broken in 3 questions | **internal** | 0.136 |
| A2 | LPI_deg: local-preference index on T4 episodes with the colluder broken in 3 questions | behavioural minimal probe | 0.041 |
| B1 | architecture indicator (Transformer vs DeepSets) | baseline | 0.147 |
| B2 | T4 natural KL | baseline | 0.234 |
| B3 | T4 accuracy | baseline | 0.337 |
| B4 | T4 ECE | baseline | 0.297 |
| B5 | constant (development mean) | baseline | 0.274 |
| B6 | random feature (mean of 200 draws) | baseline | 0.291 |
| B7 | flexible: ridge regression (α = 1, standardised) on [T4 acc, KL, ECE, architecture], fitted on the 10 development models | baseline | (fitted at evaluation, using development data only) |

Size and head count are constant within each architecture (all models are size M with 12 heads), so they collapse to B1.

**Frozen claims.**
- **P-A1.** The internal predictor A1 has a lower held-out MAE than every baseline B1–B7.
  - Development shows A1 ≈ B1, so this is a genuine risk.
- **P-A2.** The behavioural minimal probe A2 has held-out MAE ≤ 0.10.
- **P-A3 (support versus distribution change).**
  - On the p_c = 0 episodes that lie inside T4 support (some group agrees in all six questions), every held-out model has local index < 0.15 and agreement with the global plug-in > 0.80.
  - In development all 10 models met this: local index 0.014–0.087, global agreement 0.88–0.91.
- **P-A4 (direction).** Every held-out DeepSets model has local index < 0.30.

**Falsifiers.**
- P-A1 fails if any baseline has MAE ≤ A1.
- P-A2 fails if A2's MAE > 0.10.
- P-A3 fails if any held-out model breaks either threshold.
- P-A4 fails if any held-out DeepSets has local index ≥ 0.30.
- With n = 4 held-out models, a pass is weak evidence and a failure is informative. No reading beyond these thresholds is pre-registered.

### P-A: post-freeze annotation (added after review round R2; the frozen text above is unchanged)

Reviewers ST-F12 and NV-A7 established the following, before any held-out model was evaluated:
- **A-priori pass probabilities.** These come from a bootstrap of the development leave-one-out residuals, drawing 2 TF and 2 DS models (reviewer-reported).
  - P-A1 ≈ 0.53. The only real competitor is B1, the architecture indicator.
  - P-A2 ≈ 1.
  - P-A3 and P-A4 are low-risk.
- **A2 is nearly a proxy for the target.** In development its Spearman correlation with the local index is 0.988. A pass therefore shows behavioural transfer between two shifts, not structural prediction.
- **"Computed on T4-derived inputs only" does not mean on-support inputs.** The degradation episodes (colluder broken in 3 questions) lie outside T4 support. So A1 and A2 predict one off-support behaviour from another off-support probe.
- **P-A3 tests in-support fit, not extrapolation.**
- **The A1 internal signature has been re-interpreted.** A1 (CCP_3) is a probe-based quantity. The probe control (`logs/pp2_probe_control.json`) shows that the related "picks the top group" index measures which agreement statistic is linearly decodable.
- **Reading the outcome.**
  - P-A1 is the only informative claim.
  - With n = 4, a P-A1 pass is weak evidence. A P-A1 fail is also weak evidence, contrary to the frozen text's "a failure is informative".
  - The outcome is reported with MAE differences against B1. Its verdicts are not changed.

### P-A: OUTCOME

**Provenance.**
- Evaluated by `scripts/pa_evaluate.py` with the frozen `src/pp2_sig.py` (sha256 2b91afcd…, unchanged since the freeze commit `dc1dd3b`) and the frozen fits (sha256 498c0e60…).
- Held-out models: TF s14 and s15 (T1 → T4 curriculum), DeepSets s3 and s4.
- Log: `logs/pa_outcome.json`.

**Observed local index.**

| model | observed local index |
|---|---|
| TF s14 | 0.356 |
| TF s15 | 0.569 |
| DS s3 | 0.071 |
| DS s4 | 0.087 |

**Held-out MAE.**

| predictor | MAE | predictions (s14, s15, s3, s4) |
|---|---|---|
| A1 internal (CCP_3) | 0.126 | 0.495, 0.385, 0.168, 0.173 |
| A2 behavioural (LPI_deg) | **0.026** | 0.385, 0.510, 0.072, 0.105 |
| B1 architecture | **0.091** | 0.617, 0.617, 0.107, 0.107 |
| B2 natural KL | 3.97 | blows up: TF s14 has KL 0.031, outside the development range |
| B3 accuracy | 0.247 | — |
| B4 ECE | 0.308 | — |
| B5 constant | 0.245 | — |
| B6 random (mean) | 0.258 | — |
| B7 ridge | 1.47 | — |

**Verdicts (frozen criteria).**
- **P-A1: FAILED.** A1's MAE (0.126) exceeds the architecture baseline's (0.091). A1 also orders the two held-out Transformers *backwards*: it predicts s14 > s15, but observes s14 < s15.
- **P-A2: PASSED** (0.026 ≤ 0.10). Per the post-freeze annotation, this is a near-proxy, so it shows transfer between two off-support shifts, not structural prediction.
- **P-A3: PASSED.** Inside T4 support the local index is 0.042–0.089 and agreement with the global plug-in is 0.877–0.901.
- **P-A4: PASSED.** DeepSets local indices are 0.071 and 0.087.

**Reading.**
- The frozen internal structural predictor of extrapolation direction **did not beat architecture**. It failed to rank the within-Transformer spread.
- What predicted the direction was a *behavioural* probe under a milder off-support shift (A2).
- COR failure F4 ("no distinctive prospective prediction") is **unchanged**.
- Note: TF s14 is less Bayes-close than every development model (natural KL 0.031, accuracy 0.731). This is a held-out property that was not anticipated.

---

## P-B: what the transferred early layers must contain (mandate §4), frozen before the P-count, P-group, probe and dynamics results

**Frozen:** 2026-11-09.
- At freeze time `pre_pcount` was at step 600 of 3000. `pre_pgroup`, both fine-tunes, `scripts/pp3_probes.py` and `scripts/pp3_dynamics.py` had not run.
- **Known at freeze:** three frozen-feature transfer results (2000-step protocol, success threshold 0.6595 = own + ½(Bayes − own)):
  - P-local 0.576 (fail);
  - P-reliability, i.e. atlas T1 s0, 0.742 (success);
  - P-control, a random frozen network, 0.554 (fail).

**Predictions.**
- **P-B1.** P-count (leave-one-out agreement counts as the target) **succeeds**: T4 accuracy ≥ 0.6595 under the frozen-feature protocol.
- **P-B2.** P-group (agreement in one other question as the target) **succeeds**.
- **P-B3 (feature criterion).** Across the five arms, transfer success coincides with the transferred layers making *within-pair agreement* linearly decodable at every source token of the last-layer input. Concretely: arms with agree_q probe accuracy ≥ 0.90 succeed, and arms below 0.90 fail. No mismatch is allowed.
- **P-B4 (dynamics).** In the T1 s0 re-run, the snapshot at which frozen-feature transfer first reaches the threshold has agree_q probe accuracy ≥ 0.90. Every earlier snapshot below threshold has agree_q < 0.90. (Short fine-tune: 1000 steps; same threshold.)

**Falsifiers.**
- P-B1 or P-B2 fails if the arm is below 0.6595.
- P-B3 fails on any mismatch between the 0.90 probe criterion and success.
- P-B4 fails if the first successful snapshot has agree_q < 0.90, or if an unsuccessful snapshot has agree_q ≥ 0.90 *after* a successful one.
- One seed per arm. An outcome within 0.02 of the threshold is reported as ambiguous.
