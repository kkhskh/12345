# Frozen confirmation and statistics

COR Phase 1.5, Step 1, Investigation XIII-D. This document covers the frozen protocol and the verdicts; it does not validate COR.

## Freeze chronology

Each freeze commit precedes every confirmation output it governs. All are on branch `cor-phase15-step1-causal-algorithm-selection-validation`.

| freeze | commit | time (UTC) | predictions | preflight | hashes |
|---|---|---|---|---|---|
| 1 | `8a116e4` | 17:54 | P2a, P2b, P2c (Workstream B) | `logs/preflight_f1.json`: P0, N1–N5 OK | `logs/FREEZE1_HASHES.txt` |
| population 2 part 1 | `a931af7` | 18:29 | declares A-C1..A-C4, C-C1, C-C2 before launch | — | — |
| 2 | `971f07f` | 18:49 | P1, P3, P4, P6, P7, P8 | `logs/preflight_f2.json`: P0, N1–N10 OK | `logs/FREEZE2_HASHES.txt` |
| 3a | `da9e569` | 21:16 | P5a, P5b, P5c (Workstream D) | `logs/preflight_f3a.json`: P0, N1–N5 OK | `logs/FREEZE3A_HASHES.txt` |
| 3b | `dcf511a` | 22:42 | P10 (Workstream A follow-up) | `logs/preflight_f3b.json`: P0, N1–N4 OK | `logs/FREEZE3B_HASHES.txt` |

**Decision rules (all freezes).**

- A missing or malformed input makes its arm FAIL. It never makes the arm non-diagnostic.
- A non-competent model makes its arm non-diagnostic. Competence means KL to the posterior ≤ 0.02 at step 4000, or ≤ 0.05 at step 1211.
- A prediction is CONFIRMED iff it has at least one diagnostic arm and every diagnostic arm passes.
- A prediction is FALSIFIED if any diagnostic arm fails.

Scorers: `scripts/score_f1.py`, `score_f2.py`, `score_f3a.py`, `score_f3b.py`. Each preflight builds synthetic consistent and violating inputs, and checks that each violation flips its target prediction and that malformed or empty inputs fail.

### Disclosed protocol events (COMPUTE_LOG)

1. **18:44:55Z.** A monitoring grep printed the C-C1 adaptation's training summary line before freeze 2. This is a competence precondition, not a carrier measurement.
2. **18:49:50Z.** The P6 row unit uP161 s20 had been analysed by d_rows in XIII-C development under the old scoring. Its TRUE-arm R² was therefore known; the corrected arms were not.
3. **18:43Z.** A claim-key collision would have silently skipped A-C2, repeating the XIII-C C8 failure mode. It was fixed with an inert `FX_CONF=1` prefix before A-C2 started.
4. **Repeated OOM kills on the 2-CPU, 8 GB container.**
   - Development and confirmation jobs were killed and re-run, with inert `*_RETRY` prefixes. The commands were otherwise unchanged; every run is deterministic, as determinism_check shows.
   - One development calibration run, for P5b, was lost. P5b was therefore frozen from theory.
   - A lane-script edit while lanes were running truncated four stdout logs. No run, checkpoint or result was affected.
5. **Erratum.** The freeze-3a file cites "COMPUTE_LOG 21:23Z"; the correct time is 21:16Z.
6. **Freeze 3b is a post-hoc hypothesis.** It was formulated after the P1 outcome had been seen, and it was tested on a parent not used before (uP164).

## Model inventory (confirmation)

| role | units |
|---|---|
| B cohort (freeze 1) | uP160 21/21/21 with ro0 mask 0–100 (F1a), 0–250 (F1b), 546–1146 branched (F1e); uP160 23/23/23 with l2q mask 0–100 (F1c), 0–250 (F1d), 546–1146 branched (F1f) |
| A W-only (P1) | uP161 21/22/21 (A-C1), 22/21/22 (A-C2); uP160 21/23/21 (A-C3), 23/21/23 (A-C4) |
| A requirement (P10) | uP164 20/21/21, 21/20/21, 21/21/20 at step 1211; references uP164 s21, s20; descriptive: uP163 21/20/21 |
| C carrier (P3, P4) | early01 adaptations of uP160 s23 (C-C1), uP165 s21 (C-C2) |
| D access (P5) | uP165 s20 |
| E1 (P6) | uP164 s21, uP165 s20 |
| E2 (P6) | uP161 s20, uP165 s23 |
| E4 (P7) | uP164 s22, uP163 s20, uP160 s20 |
| E3 (P8) | uP165 s20, uP163 s22, uP164 s21 |

Development used:

- uP161 s21/s22;
- uP160 s21/s23;
- the uP163 L/W/D 2³ factorial;
- XIII-C runs.

## Verdicts

| id | statement (short) | verdict | deciding numbers |
|---|---|---|---|
| P1 | W-only change moves the organisation to the same-W diagonal, both directions, two parents | **FALSIFIED** | early→mixture direction passes (d_W 0.009, 0.039); mixture→early fails in both parents (d_W 1.09, 1.02) — all four cells end mixture |
| P2a | commitment signature (present, unused, readout-reactivatable, RO1-dependent) | **FALSIFIED** | F1a / F1e pass every criterion except the cap on gain with RO1 removed (0.054, 0.0505 vs ≤ 0.05); F1b fails substantively (present 0.74, gain 0.063) |
| P2b | 100-step masks recover (F1a early, F1c mixture) | **FALSIFIED** | F1a → mixture, F1c → early: a 100-step block permanently flips the organisation |
| P2c | no suppression of the unused route | **CONFIRMED** | F1a: final presence 0.934 vs peak 0.931 while lik fell 0.198 → −0.003 |
| P3 | carrier: sufficiency, nulls, encoding ≠ decoding, content, K/Q trace | **CONFIRMED** | all five arms in both C-C units (DAS 0.90 / 0.82; DEC_m −0.001 / 0.033; ENC_m 0.85 / 0.76; K+Q 0.96 / 0.94) |
| P4 | natural-donor necessity; synthetic ablation non-specific | **CONFIRMED** | reverse leaves 0.059 / 0.038 (random 0.97 / 0.79); ablation damage KL 0.57 / 0.51 |
| P5a | access not capacity at lr 3e-3; supply fails at lr 1e-3 | **FALSIFIED** | supL2 lr 3e-3 0.906, L2 0.000 (pass); supL2 lr 1e-3 0.168 (predicted < 0.1) |
| P5b | either supply position suffices | **CONFIRMED** | supRO 0.894, supQS 0.727 |
| P5c | layer-0-specific repair at matched capacity | **CONFIRMED** | L0only 0.832, L1only 0.004 |
| P6 | corrected mediation E1 + E2 | **FALSIFIED** | E1 source-free excess 0.13 in uP165 s20 (≥ 0.15 required; E3 family source-free 0.92); every other arm passes |
| P7 | positional-asymmetry hypothesis of the early error is false | **CONFIRMED** | SYM1 MOVE reduction −0.06, −1.60, −0.08 (≤ 0.10); KL increases in all three |
| P8 | routes are interdependent components | **CONFIRMED** | interactions ≤ 0.07; single open route ≤ 0.51 of baseline |
| P10 | early needs more than W; uP164 L-only cell not early | **FALSIFIED** | uP164 L21/W20/D21 stays EARLY (lik 0.78, mix 0.16) |

**Totals:** 13 frozen predictions; 7 confirmed, 6 falsified, 0 non-diagnostic.

## Threshold sensitivity

Source: `scripts/uncertainty_13d.py` → `logs/uncertainty_13d.json`. Every numeric threshold was multiplied by 0.8 and by 1.25 in turn, and the frozen scorer was re-run on the frozen inputs, using a temporary copy so the frozen files were never touched.

- Freeze 1: 14 perturbations, 1 flip. P2c becomes FALSIFIED if the presence-retention ratio is raised from 0.9 to 1.125, which would mean demanding *more* than the peak. P2a stays FALSIFIED under every perturbation because F1b fails substantively.
- Freeze 2: 48 perturbations, 2 flips.
  - P4 becomes FALSIFIED if the random-subspace retention bound is raised from 0.7 to 0.875 (C-C2 RAND0 = 0.79).
  - P6 becomes CONFIRMED if the source-free excess bound is lowered from 0.15 to 0.12.
- Freeze 3a: 14 perturbations, no flips.

So two verdicts are borderline, and both are reported as frozen:

- **P6**, falsified, would flip under a 20% looser threshold.
- **P2a**, falsified: the F1a/F1e arms miss by < 0.005, but F1b fails clearly.

## Effect sizes with unit-level bootstrap intervals

Means over units, with 95% percentile bootstrap intervals over units (10,000 draws, seed 0). n is small (3–6), so these intervals describe the spread across units; they are not population inferences.

| quantity | mean [95% interval] | n units |
|---|---|---|
| carrier DAS sufficiency slope | 0.858 [0.821, 0.903] | 3 (dev + 2 conf) |
| encoding − decoding slope (m) | 0.71 [0.55, 0.85] | 3 |
| DAS − source-free slope | 0.249 [0.198, 0.315] | 3 |
| natural-donor reverse remaining slope | 0.051 [0.038, 0.059] | 3 |
| keys + queries share of trace | 0.947 [0.938, 0.964] | 3 |
| best single open route / baseline | 0.43 [0.36, 0.50] | 5 |
| true row interchange R² | 0.933 [0.904, 0.962] | 4 |
| decoy rows vs borrowed R² | 0.002 [0.001, 0.004] | 4 |
| SYM1 MOVE reduction | −0.85 [−1.32, −0.35] | 6 |

## Factorial effects (parent-level)

Absolute change in final lik_RO1 for single-factor changes with the other two factors held fixed, pooled over parents uP160, uP161 and uP163 (`logs/uncertainty_13d.json`, factorial):

- **W-only:** mean 0.44, 8 pairs.
- **L-only:** mean 0.21, 8 pairs.
- **D-only:** mean 0.04, 10 pairs.

**Data order never changes the organisation.** In uP164 at step 1211:

- a change of L alone left the outcome early;
- a change of W alone (W21 placed on L20/D21) did not produce early.

**Combined pattern.** Which fresh component decides varies by parent:

- uP163: W decides.
- uP164: L decides.
- uP160 and uP161: both are needed.

## Interpretation of the failures

- **P1, P10 and P2b together overturn XIII-C's positive result.** XIII-C held that the organisation is set by the readout-map initialisation. The corrected statement:
  - The organisation is set within the first ~100 steps by the initialisation of the freshly initialised layer-2/readout pair. Which component matters depends on the pretrained parent. Data order does not matter.
  - Mixture is the default organisation; early is fragile.
- **P5a.** Its failure qualifies only the claim that the supply result is budget dependent. The access result itself replicated.
- **P6.** Its failure is a specificity failure in one unit and one family. The rule comparison replicated.
- **P2a.** Its failure is substantive only for the 250-step mask: a longer early block degrades the unused route itself.
