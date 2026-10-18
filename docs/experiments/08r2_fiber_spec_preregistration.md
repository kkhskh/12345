# 08R2-FIBER-SPEC preregistration

Frozen before new intervention outputs. This study does not change 08R2 validation, 08R2-S, or 08R2-FIBER scientific labels.

Official status remains:

- `08R2_VALIDATION_COMPLETE — NEGATIVE`
- `0A C / PARTIAL`
- `SECTION_5_3 = PARTIAL`
- `OBSTRUCTION_READY = NO`

Confirmation remains sealed. No 08R3. No obstruction energy. No new editor.

## Estimand

For recipient prefix \(x\) and intended target board \(B_t\),

\[
\Delta_{\mathrm{specific}}
=
Y(h_E+\delta_{\mathrm{correct}})
-
\frac{1}{3}\sum_{j=1}^{3} Y(h_E+\delta_{\mathrm{wrong},j}),
\]

where every \(Y\) is scored on the same intended \((L_t\setminus L_r, L_r\setminus L_t)\), every \(\delta\) lies in \(\ker W\), and \(\|\delta_{\mathrm{wrong},j}\|_2=\|\delta_{\mathrm{correct}}\|_2\).

A matched correct-minus-random contrast uses three frozen-seed nullspace directions at the same L2.

## Pair construction

- Validation split only.
- Exclude 08R2-FIBER primary recipient and target games (recomputed ABD seed 1082).
- Same player, same occupancy bucket, simulator-valid, nonempty became-legal and became-illegal.
- A: Hamming \(>8\). B: \(2\le\) Hamming \(\le 8\).
- Seeded disjoint matching, seed `2182`, 400 shuffle trials.
- Target design: 32 A + 16 B, 48 distinct recipients, 48 distinct targets, no game in both roles.

Local census (validation prefixes, holdout-excluded, FIBER games excluded):

- Candidate A ordered pairs: 191842
- Candidate B ordered pairs: 1228
- Selected: 32 A + 16 B
- Distinct recipients: 48
- Distinct targets: 48
- Recip/target overlap: none
- `pair_manifest_sha256`: `2789accfc4b0908168509f7a998f9e95ddb8a4f36f7a661aa1b9d5e4a11143d3`

If a later environment cannot reproduce this hash, stop. Do not substitute another pair set.

## Wrong-donor matching

Per pair, three distinct games with:

- same player as the recipient,
- occupancy within 4 of the correct donor,
- legal-set size within 4,
- prefix length within 4,
- not the recipient or correct-donor game,
- not confirmation games.

Seed `2282`. Record actual occupancy, legal-size, and prefix differences. Do not select using model outputs.

## Random controls

Three directions in \(\ker W\), seeds `3182, 3282, 3382`, each scaled to \(\|\delta_{\mathrm{correct}}\|_2\).

## Editor A

Frozen 08R2 budget-16 reconstruction: 40 steps, lr 0.4, \(\lambda=0.01\), \(\eta=1.0\), \(\mu=0\). Failures are recorded, not retuned.

## Tolerances

- Probe contrast rank: relative `1e-8` to the largest singular value. Sensitivity recorded at `1e-6, 1e-8, 1e-10`. Not tuned to behavior.
- Probe-probability invariance: max abs `1e-8` float64, `1e-5` float32.
- Actual L2 relative error `1e-6`. Degenerate directions \(\|\cdot\|_2<1e-8\) are rejected, not replaced.

## Primary gate

On pairs that pass invariance and L2 matching:

1. Mean \(\Delta_{\mathrm{specific}}>0\)
2. Recipient-game clustered 95% CI lower bound \(>0\)
3. Mean \(\Delta_{\mathrm{specific}}\ge 0.05\)
4. Mean \(G_{\mathrm{correct}}-\overline{G_{\mathrm{wrong}}}>0\) and mean \(Q_{\mathrm{correct}}-\overline{Q_{\mathrm{wrong}}}<0\)
5. Mean correct-minus-random \(Y\) contrast \(>0\) and its CI lower bound \(>0\)
6. No silent exclusions; failed attempts remain in accounting

Bootstrap: 2000 resamples, seed 82. Aggregate the three wrong and three random responses inside each pair before clustering by recipient game.

The 0.05 floor is the minimum meaningful effect for this new estimand. It is not tuned after seeing SPEC outputs.

## Verdicts

- `TARGET_SPECIFIC_NULL_EFFECT_SUPPORTED` only if the gate holds.
- `TARGET_SPECIFIC_NULL_EFFECT_NOT_SUPPORTED` if the study completes and the gate fails.
- `TARGET_SPECIFIC_NULL_EFFECT_INCONCLUSIVE` if integrity or sample support is insufficient.
- `FIBER_SPEC_ENGINEERING_FAILED` for implementation or identity failures.

A supported gate does not define a new causal germ and does not change Section 5.3 from PARTIAL.
