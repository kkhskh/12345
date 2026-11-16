# Frozen confirmation and statistics

COR Phase 1.5, Step 1, Investigation XIII-D. This document covers the frozen protocol and the verdicts; it does not validate COR.

## Freeze chronology

Each freeze commit precedes every confirmation output it governs. All are on branch `cor-phase15-step1-causal-algorithm-selection-validation`.

| freeze | commit | time (UTC) | predictions | preflight | hashes |
|---|---|---|---|---|---|
| 1 | `8a116e4` | 17:54 | P2a, P2b, P2c (Workstream B) | `logs/preflight_f1.json`: P0, N1–N5 OK | `logs/FREEZE1_HASHES.txt` |
| population 2 part 1 | `a931af7` | 18:29 | declares A-C1..A-C4 and C-C2 before launch; C-C1 was launched 18:08Z, before its declaration (not analysed before freeze 2) | — | — |
| 2 | `971f07f` | 18:49 | P1, P3, P4, P6, P7, P8 | `logs/preflight_f2.json`: P0, N1–N10 OK | `logs/FREEZE2_HASHES.txt` |
| 3a | `da9e569` | 21:16 | P5a, P5b, P5c (Workstream D) | `logs/preflight_f3a.json`: P0, N1–N5 OK | `logs/FREEZE3A_HASHES.txt` |
| 3b | `dcf511a` | 22:42 | P10 (Workstream A follow-up) | `logs/preflight_f3b.json`: P0, N1–N4 OK | `logs/FREEZE3B_HASHES.txt` |

**Decision rules (all freezes).**

- A missing or malformed input makes its arm FAIL. It never makes the arm non-diagnostic.
- A non-competent model makes its arm non-diagnostic. Competence means KL to the posterior ≤ 0.02 at step 4000 (freezes 1–2), or ≤ 0.05 at step 1211 (freeze 3b). Freeze 3a has no competence exclusion: failing to learn is the measured outcome.
- A prediction is CONFIRMED iff it has at least one diagnostic arm and every diagnostic arm passes.
- A prediction is FALSIFIED if any diagnostic arm fails.

Scorers: `scripts/score_f1.py`, `score_f2.py`, `score_f3a.py`, `score_f3b.py`. Each preflight builds synthetic consistent and violating inputs, and checks that each violation flips its target prediction and that malformed or empty inputs fail.

### Disclosed protocol events (COMPUTE_LOG)

1. **18:44:55Z.** A monitoring grep printed the C-C1 adaptation's training summary line before freeze 2. This is a competence precondition, not a carrier measurement.
2. **18:49:50Z.** The P6 row unit uP161 s20 had been analysed by d_rows in XIII-C development under the old scoring. Its TRUE-arm R² was therefore known; the corrected arms were not.
3. **18:43Z.** A claim-key collision would have silently skipped A-C2, repeating the XIII-C C8 failure mode. It was fixed with an inert `FX_CONF=1` prefix before A-C2 started.
4. **Repeated OOM kills on the 2-CPU, 8 GB container.**
   - Development and confirmation jobs were killed and re-run, with inert `*_RETRY` prefixes. The commands were otherwise unchanged.
   - Determinism was tested directly only for training (`ft_factorial.py`; determinism_check). No confirmation trainer needed a resume.
   - The killed analysis and adaptation jobs (adapt4, d_rows, route_status, unit_metrics) had written no output, or, for adapt4, only completed arms, and were rerun from scratch. adapt4 seeds each arm identically, so a partial rerun is equivalent to a full one.
   - One development calibration run, for P5b, was lost. P5b was therefore frozen from theory.
   - Editing `lane.sh` while lanes were running made the lanes, at their ends, execute nine stray empty commands. These occurred at 20:20:53, 20:38:22, 20:40:35, 20:41:48, 20:44:13, 21:45:18, 22:21:07, 22:40:15 and 22:42:56.
   - The stray commands truncated six stdout files: `44e05fb2e80e` (A-C4, while it was running), `c55bf7ca5fe7`, `c614e77c4df3`, `860474af5974`, `7fa9fdc58618` (P1 unit metrics) and `dc9f5e7c1cda`.
   - COMPUTE_LOG 20:52Z wrongly named A-C3's log as truncated; it is intact (review 3, finding 6).
   - Only empty commands ran. No run, checkpoint or JSON result was affected. In principle, a fragment of a real command could have run. Later lanes use the unedited copy `lane2.sh`.
5. **Erratum.** The freeze-3a file cites "COMPUTE_LOG 21:23Z"; the correct time is 21:16Z.
7. **Interim scoring of freeze 2 (undisclosed until review 3, finding 1).**
   - The freeze-2 scorer was run on partial inputs at 19:15Z and 20:00Z (`logs/interim/`).
     - At 19:15Z P7 and P8 showed CONFIRMED; the other predictions had missing inputs.
     - At 20:00Z P3 and P4 showed CONFIRMED and P1 and P6 FALSIFIED, while A-C1..A-C4 were still training.
   - Decisions taken afterwards: A-C parallelisation at 20:20Z (scheduling only), and the P5 development-scope reduction at 20:02Z.
   - **Freeze 3a's P5b drew its theoretical basis from the interim P3 result.**
   - The verdicts are unaffected: the scorer is deterministic, and the A-C runs were fixed and queued.
8. **Overlap and labelling errata (review 3, finding 11).**
   - Freeze 2 states that no confirmation unit appears in development. But natural uP160 s23, the parent model of C-C1 before adaptation, was analysed in `d_rows_dev.json`.
   - Freeze 3a's "new unit" uP165 s20 had already been analysed in freeze-2 P6/P8, though not adapted. The same parent, uP165, supplied C-C2.
   - P10's frozen text labels cell (a) L20/W21/D21 as "the early diagonal's W on the non-early L, D". Relative to the early diagonal 21/21/21, it is an **L-only** change, and cell (b) L21/W20/D21 is a **W-only** change. The scorer is unaffected.
   - The freeze-3a erratum time should read 21:16:38Z.
9. **P1 reference stream.** The P1 references are frozen constants measured on XIII-B/XIII-C streams; the cells were measured on 92.1 M. The verdict is robust: d_W 0.009 / 0.039 vs d_other ≈ 1.08, and 1.09 / 1.02 vs 0.002 / 0.13.
10. **Post-freeze queue changes.** `queue_f3a.txt` gained two `AD_RETRY=1` lines after freeze 3a; retry queues were not hashed at their creation. Their post-hoc hashes are in `logs/FREEZE_ADDENDUM_HASHES.txt`.
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
| P2a | commitment signature (present, unused, readout-reactivatable, RO1-dependent) | **FALSIFIED** | F1a / F1e pass every criterion except the cap on gain with RO1 removed (0.054, 0.0505 vs ≤ 0.05; within noise); F1b passes presence (0.743 ≥ 0.6) but fails both gain criteria (0.063 < 0.10; 0.066 > 0.05) |
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
| P10 | in uP164 a W-only change (L21/W20/D21) is not early | **FALSIFIED** | the W-only cell stays EARLY (lik 0.78, mix 0.16); the L-only cell L20/W21/D21 is not early |

**Totals:** 13 frozen predictions; 7 confirmed, 6 falsified, 0 non-diagnostic.

## Threshold sensitivity

Source: `scripts/uncertainty_13d.py` → `logs/uncertainty_13d.json`. Every numeric threshold was multiplied by 0.8 and by 1.25 in turn, and the frozen scorer was re-run on a temporary copy of the frozen inputs.

The first version of the script wrote the perturbed verdicts into `logs/` and restored the files afterwards (review 3, finding 2). It was corrected, and the frozen score files were verified unchanged by sha256 after the rerun. Freeze 3b has no sensitivity analysis, because its thresholds are in prose.

- Freeze 1: 14 perturbations, 1 flip. P2c becomes FALSIFIED if the presence-retention ratio is raised from 0.9 to 1.125, which would mean demanding *more* than the peak. P2a stays FALSIFIED under every perturbation because F1b fails substantively.
- Freeze 2: 48 perturbations, 2 flips.
  - P4 becomes FALSIFIED if the random-subspace retention bound is raised from 0.7 to 0.875 (C-C2 RAND0 = 0.79).
  - P6 becomes CONFIRMED if the source-free excess bound is lowered from 0.15 to 0.12.
- Freeze 3a: 14 perturbations, no flips.

So two verdicts are borderline, and both are reported as frozen:

- **P6**, falsified, would flip under a 20% looser threshold.
- **P2a**, falsified: the F1a/F1e arms miss by < 0.005, but F1b fails clearly.

## Effect sizes (confirmation units only)

These are means over confirmation units, with 95% percentile bootstrap intervals over units (10,000 draws, seed 0). With n = 2–3, the interval is essentially the observed range.

Development units are excluded because they set the thresholds (review 3, finding 5); their values are given in the workstream reports.

| quantity | mean [range] | units |
|---|---|---|
| carrier DAS sufficiency slope | 0.862 [0.821, 0.903] | C-C1, C-C2 |
| encoding pattern − decoder filter slope (m) | 0.789 [0.730, 0.847] | C-C1, C-C2 |
| DAS − source-free slope | 0.275 [0.235, 0.315] | C-C1, C-C2 |
| natural-donor reverse remaining slope | 0.049 [0.038, 0.059] | C-C1, C-C2 |
| keys + queries share of trace | 0.951 [0.938, 0.964] | C-C1, C-C2 |
| best single open route / baseline (scored routes) | 0.43 [0.39, 0.51] | uP165 s20, uP163 s22, uP164 s21 |
| true row interchange R² | 0.904 [0.898, 0.911] | uP161 s20, uP165 s23 |
| decoy rows vs borrowed R² | 0.002 [0.001, 0.003] | uP161 s20, uP165 s23 |
| SYM1 MOVE reduction | −0.58 [−1.60, −0.06] | uP164 s22, uP163 s20, uP160 s20 |

## Factorial effects (parent-level)

Absolute change in final lik_RO1 for single-factor changes, with the other two factors held fixed. These are pooled over uP160, uP161 and uP163, use overlapping cells and selected contrasts, and mix measurement streams (`logs/uncertainty_13d.json`). They are descriptive only:

- **W-only:** mean 0.44, 8 pairs.
- **L-only:** mean 0.21, 8 pairs.
- **D-only:** mean 0.04, 10 pairs.

**Data order never changed the class in any tested pair.**

The base rates of natural diagonals are 7 early, 6 mixture and 4 hybrid. Off-diagonal cells are early in 3 of 11 cases, against 7 of 17 diagonals (Fisher p = 0.69; review 2). The single-factor results therefore support no low-order factor law (READOUT report).

## Interpretation of the failures

These are post-hoc readings, not confirmed results.

- **P1 and P10** falsify XIII-C's "the readout-map initialisation sets the organisation" as a general cause. What remains is descriptive. The organisation depends on the parent and on the fresh (L, W) draw, with strong interactions, and not on data order. The "mixture default / early fragile" reading proposed in the first draft is withdrawn (review 2, F1 and F3).
- **P2b.** Route availability during the first 100 steps redirects the organisation permanently, in both directions (n = 1 each). No recovery window of ≥ 100 steps exists. Shorter windows and sham masks are untested.
- **P5a.** Its failure qualifies only the claimed budget dependence of supply. The access contrast (supL2 0.906 vs L2 0.000 at lr 3e-3) replicated.
- **P6.** Its failure is a specificity failure in one unit and one family.
- **P2a.** F1a/F1e miss one cap within noise. F1b's information is present but not readout-reactivatable.
