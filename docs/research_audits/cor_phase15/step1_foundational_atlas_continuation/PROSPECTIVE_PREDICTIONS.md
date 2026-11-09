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
