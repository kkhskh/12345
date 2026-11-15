# XIII-C development notes and threshold justification (written before the freeze; no confirmation output opened)

## A. Factorial (development) — all four crossed cells follow the INITIALISATION

v = (lik_RO1, mix_mass) at step 4000; stream 86_100_000.

| cell | init diagonal (v) | data diagonal (v) | cell v | route_Q | ρ (MSE) |
|---|---|---|---|---|---|
| uP161 L21 W21 D22 | s21 (0.006, 0.966) | s22 (0.78, 0.20) | (0.007, 0.965) | 0.37 (s21: 0.39) | 0.85 (s21: 0.87) |
| uP161 L22 W22 D21 | s22 (0.78, 0.20) | s21 | (0.779, 0.215) | 0.08 | 0.53 (s22: 0.56) |
| uP160 L21 W21 D23 | s21 (0.81, 0.17) | s23 (0.03, 0.97) | (0.806, 0.167) | 0.09 | 0.62 (s21: 0.59) |
| uP160 L23 W23 D21 | s23 (0.03, 0.97) | s21 | (0.025, 0.968) | 0.78 | 0.91 |

In every cell, d_init ≤ 0.02 and d_data ≥ 1.0. The same initialisation seed gives different classes in different parents (seed 21: mixture
in uP161, early in uP160), so the effect is init × parent, not the seed value itself.

Step-0 features do not predict the class (17 known fine-tunes; `logs/init_features_dev.jsonl`):

| feature | r with final lik_RO1 |
|---|---|
| max readout→q mass of the fresh layer-2 heads | −0.41 |
| layer-2 vs layer-0 gradient ratio | −0.34 |
| first-order mass change | −0.23 |

**Trajectories** (XIII-B units plus factorial cells; `logs/unit_metrics_traj_xiiib.jsonl`, `_peek`, `_fx_dev`):
- Every run grows early readout (lik_RO1 0.12–0.39 by steps 111–246).
- Mixture-class runs form a q-mass head early: mix_mass 0.77–0.87 at step 246, or 0.35 → 0.89 between 246 and 546. Their lik_RO1 then
  FALLS to ≈ 0, a suppression.
- Early-class runs never form one: mix ≤ 0.25 throughout.
- The hybrid uP161 s23 forms the head late (0.47 at 546, 0.89 at 1211), after lik_RO1 had reached about 0.49. Both persist (0.42 / 0.92).

mix_mass at step 546:

| class | mix_mass at 546 |
|---|---|
| mixture | 0.89–0.94 |
| hybrid | 0.47 |
| early | 0.17–0.23 |

The R2c thresholds (≥ 0.6 → final lik ≤ 0.2; ≤ 0.35 → final lik ≥ 0.6; in between no prediction) sit in the empty gaps of this
distribution.

**ro0 intervention (development, uP161 L22/W22/D22, early-destined):**
- mix_mass 0.69 at step 246 and 0.91 at 546; lik_RO1 −0.002 throughout.
- Final mixture: lik −0.001, mix 0.958, ρ_MSE 0.87, against 0.56 unintervened.
- Thresholds for R2a are mix ≥ 0.8 and lik ≤ 0.2.

**l2q: no development run** (descoped for the critical path).
- The R2b threshold (lik ≥ 0.4) comes from the competition hypothesis and the hybrid trajectory: early readout reaches about 0.45–0.7 by
  step 546–1000 when no mixture head exists.
- If the head forms after release, the hybrid case shows lik_RO1 stays near 0.4.
- This is a deliberately risky prediction.

**L2-vs-WA split: no development run.** R1b is an a-priori prediction: the mixture head lives in the fresh layer 2.

## B. Decision law (thresholds for R3)

**ρ_KL (PM, KL objective)** by class:

| class | ρ_KL | bootstrap 5–95% |
|---|---|---|
| early | 0.71–0.73 | widths ≈ 0.05–0.06 |
| hybrid | 0.80 | |
| mixture | 0.91–0.93 | widths ≈ 0.03 |

- An organisation change from early to mixture therefore implies Δρ ≈ +0.2, and from mixture to hybrid ≈ −0.1.
- The min_shift of 0.08 is above two bootstrap half-widths of a single unit (0.015–0.03 each) and below the expected class gaps.
- The R3b correlation threshold is −0.7; XIII-B AQ1 gave −0.99 over 5 units with the MSE ρ.

## C. Two-colluder repair (thresholds for R4 and R5; dev unit uP161 s21, β = 0.6, Bayes acc 0.792)

| arm | acc | J slope | R² | D0/DM | DECOY/DM | trainable |
|---|---|---|---|---|---|---|
| native | 0.469 | — | — | — | — | — |
| L2 | 0.722 | 0.000 | 0.00 | 0.84 | 0.54 | 50,048 |
| early01 | 0.783 | 0.920 | 0.85 | 0.14 | 0.08 | 99,456 |
| L1only | 0.701 | 0.000 | 0.00 | 0.95 | 0.43 | 49,728 |
| **L0only** | 0.786 | **0.847** | 0.80 | 0.20 | 0.11 | 49,728 |
| L2plus | 0.723 | 0.001 | 0.00 | 0.83 | 0.46 | 99,776 |
| preattn | 0.731 | 0.142 | 0.16 | 0.62 | 0.17 | 16,512 |
| pretok | 0.659 | −0.001 | 0.00 | 0.82 | 0.46 | 16,640 |
| supL2 (m group-agnostic) | 0.732 | 0.024 | 0.09 | 0.54 | 0.35 | 50,240 |
| supFrozen | 0.504 | 0.075 | 0.32 | 0.26 | 0.16 | 192 |
| supDecoy | 0.728 | 0.013 | 0.04 | 0.74 | 0.49 | 50,240 |
| **supG (per-group format)** | 0.787 | **0.802** | 0.81 | 0.06 | 0.04 | 50,240 |
| supGFrozen | 0.548 | 0.144 | 0.45 | 0.20 | 0.12 | 192 |

R4 thresholds: < 0.2 for the arms that fail (dev values ≤ 0.024) and ≥ 0.6 for the arms that succeed (dev ≥ 0.80). XIII-B early units
gave layer-2 J slopes of 0.53 and 0.63, hence ≥ 0.4.

**c_causal (dev, uP161 s21 early01):**
- m is decodable after adaptation, residualised R² 0.72–0.81 (native ≈ 0). The decoy statistic is decodable too (0.60–0.78).
- m-probe 3-dim interchange: 0.002–0.008. Responsibility-probe interchange: 0.015. Ablating either leaves 0.93–0.94 of the effect.
- Learned DAS at QS+RO carries 0.64 (k = 1), 0.875 (k = 3) and 0.92 (k = 8). Its alignment with the probe spans is at the random baseline
  (max cos 0.34 vs 0.39 for k = 3).
- Frozen layer-2 Shapley: rows 0.81, MLP 0.13. corr(Δ group mass, Δ exact role posterior) is 0.89 and 0.87 for the two mixture heads.
- R5 thresholds: DAS3 ≥ 0.6, DAS8 ≥ 0.7, probe ≤ 0.1, rows ≥ 0.6, corr ≥ 0.7.

## D. Controls (thresholds for R6, R7, R8)

**R6.**
- TRUE R² was 0.90–0.96 in the pure mixture dev units (0.40 in the hybrid, which is why the R6 arm requires lik ≤ 0.15).
- SHUFD ≤ 0.005; SHUF/SHUFB/PMEAN ≤ 0.06; NOOP = 0.
- OTHER-head R² ≤ 0.08, except uP161 s20 at 0.40. That gives a gap of 0.50 at worst, hence a gap threshold of 0.4.

**R7.** In the early dev units at ALL2, k = 8:
- excess over SRCFREE 0.31–0.33;
- E−A2 margins: E5 0.21–0.26, E3 ≥ 0.52, QF4 ≥ 0.72;
- E−A1 margin on QF5 ≥ 0.54;
- random subspaces ≤ 0.16.

Thresholds: excess ≥ 0.2, margins ≥ 0.15, random ≤ 0.25. The E5 margin is the tightest.

**R8.** Mixture dev units: sufficiency Q 0.40–0.69 and R 0.31–0.60; |redundancy| ≤ 0.05. Early dev units: O 0.44–0.47.
Thresholds: Q, R ≥ 0.25; O ≥ 0.3; |red| ≤ 0.1.

## Borderline procedure (pre-declared)

Any unit measure whose value lies within ±0.05 of a threshold is flagged "borderline". For flagged lik_RO1 / mix_mass values, unit_metrics
is recomputed on an independent second stream (88_150_000) and both values are reported. **The frozen verdict uses the first stream only.**
The sensitivity result is reported next to it and never replaces it.
