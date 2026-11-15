# Frozen confirmation and statistics (Investigation XIII-C)

COR Phase 1.5, Step 1 (provisional ≈ 25%). The full scorer output is in `logs/score_13c.json` and the input bundle in
`logs/bundle_13c.json`.

## 1. Protocol and chronology

| time (UTC) | event | commit / record |
|---|---|---|
| 06:52 | development factorial launched | `logs/COMPUTE_LOG.txt` |
| 07:23 | container restart killed the first three development runs; trainer made resumable (bitwise check) | `logs/repro_resume.json` |
| 09:11:44 | **confirmation population C1–C9 declared, committed and pushed** before any confirmation run | 9ec86d6 |
| 09:11:45 | confirmation training starts (allowed: no output opened before the freeze) | lanes C1/C2 |
| 10:21 | **prediction freeze** R1–R8: scorer, null preflight (all checks OK), analysis queue, hashes; confirmation outputs excluded from the commit | **d6281c6** (pushed) |
| 10:21:49 | confirmation analysis lane starts | |
| 12:32 | **incident:** confirmation unit C8 had been silently skipped by its lane, because its queue line was byte-identical to a descoped development line whose claim existed. Detected, logged and launched; nothing was replaced | COMPUTE_LOG |
| 13:23 | scoring | `logs/score_13c.json` |

**Integrity review 2** (agent, `reviews/REVIEW_2_INTEGRITY.md`):
- It verified the chronology, and that all 16 frozen-file hashes are unchanged.
- File access times show the first read of every pre-freeze confirmation file at the moment git hashed it for the freeze commit; no
  earlier reads are recorded.
- It reproduced the preflight.

Units: nine confirmation runs, all **fine-tuning runs of three pretrained parents** (uP160, uP161, uP163). Two of these parents were also
development parents. They are not independent parents.

## 2. Verdicts

| prediction | verdict | arms (diagnostic: pass / fail) | what failed |
|---|---|---|---|
| R1 init vs data; L2-init vs readout-init | **FALSIFIED** | a: 4/0; b: 0/2 | both split cells follow the **readout-map (WA) initialisation and data**, not the layer-2 initialisation |
| R2 pathway competition | **FALSIFIED** | a: 1/0; b: 2/0; c: 3/2 (1 non-diagnostic) | the step-546 predictor: uP161 L23/D20 (mix546 0.72 → final lik 0.24 > 0.2, borderline) and uP163 L22/W21/D21 (mix546 0.26 → final lik 0.45 < 0.6) |
| R3 routing → decision law (causal) | **CONFIRMED** | a: 3/0; b: 1/0 | — |
| R4 repair: capacity / access / format | **FALSIFIED** | late: 5/1; early: 1/1 | supG reached J slope 0.28 (< 0.6) at the frozen 1000 steps; early-unit L2 gave 0.378 (< 0.4, borderline) |
| R5 constructed carrier ≠ decodable statistic; frozen rows carry it | **CONFIRMED** | 3/0 | — |
| R6 attention-row specificity | **FALSIFIED** | 0/1 (5 non-diagnostic) | the frozen criterion "no-op max \|Δlogit\| = 0 exactly" failed: 7.2e-7, float32 summation order. Every specificity criterion passed |
| R7 early-class E mediator | **CONFIRMED** | 4/0 (unit 2 non-diagnostic: hybrid) | — |
| R8 route structure | **CONFIRMED** | 3/0 (3 non-diagnostic: hybrids) | — |

Tally:
- 4 confirmed, 4 falsified, 0 inconclusive.
- No missing inputs, after the C8 incident was repaired.
- The predictions share units; they are not independent.

## 3. Arm-level evidence

**R1a. Init versus data.** v = (lik_RO1, mix_mass). Brackets give episode-bootstrap 95% intervals of the distances (B = 200;
`logs/uncertainty_13c.json`).

| cell | v | d_init | d_data |
|---|---|---|---|
| uP161 L20/D23 | (0.002, 0.978) | 0.014 [0.013, 0.016] | 0.42 [0.38, 0.46] |
| uP161 L23/D20 | (0.242, 0.821) | 0.20 [0.17, 0.24] | 0.28 [0.25, 0.31] |
| uP163 L21/D22 | (0.266, 0.945) | 0.12 [0.09, 0.15] | 0.89 [0.87, 0.92] |
| uP163 L22/D21 | (0.823, 0.202) | 0.044 [0.035, 0.054] | 0.84 [0.82, 0.86] |

The hybrid diagonal (uP161 s23, lik 0.42) is reproduced only approximately when the data changes (0.24). The binary part (head / no head)
follows the initialisation exactly; the continuous hybrid residue also moves with the data.

**R1b. Split cells.**

| cell | v | distance to L2-init reference | distance to WA+data reference |
|---|---|---|---|
| uP163 L22/W21/D21 | (0.451, 0.843) | 0.70 [0.68, 0.73] | 0.10 [0.07, 0.13] |
| uP163 L21/W22/D22 | (0.816, 0.161) | 0.87 [0.85, 0.89] | 0.08 [0.08, 0.09] |

The organisation follows the readout-map initialisation together with the data stream, and not the layer-2 initialisation. R1a shows that
data alone (with L = W) does not move the organisation. Under additivity of the two factors, **the fresh readout map's initialisation is the
carrier.** This does not exclude a WA × data interaction, and review 3 discusses the run that would discriminate.

**R2.**

| unit | intervention | final v | reference | ρ_KL (reference) |
|---|---|---|---|---|
| uP160 21/21/21 | ro0 | (−0.001, 0.973): **mixture** | uP160 s21 (0.83, 0.17): early | 0.947 (0.749) |
| uP161 21/21/21 | l2q | (0.708, 0.167): **early readout** | uP161 s21 (0.007, 0.965): mixture | 0.563 (0.933) |
| uP160 23/23/23 | l2q | (0.821, 0.148): **early readout** | uP160 s23 (0.034, 0.971): mixture | 0.803 (0.935) |

All three interventions flip the organisation in the predicted direction, and the flipped organisation **persists for 3400 steps after the
mask is lifted**.

Caveat: the l2q uP161 run is **badly trained** (accuracy 0.677, KL to Bayes 0.207, against 0.0018 for its reference). The uP160 l2q run
trained normally (KL 0.0036).

R2c (the early predictor): 3 of 5 diagnostic units passed. It is not a reliable predictor near the class boundary.

**R3.**
- ρ_KL shifts: +0.198, −0.370 (badly trained unit) and −0.132, all beyond the bootstrap ranges (widths ≤ 0.03, except the badly trained
  unit's [0.47, 0.61]).
- corr(lik_RO1, ρ_KL) = −0.81 over 12 non-independent units (3 parents; 3 of the units are development references).
- Without the badly trained unit, the causal arm still has two diagnostic passes (+0.20, −0.13).

**R4.** Late unit: uP161 L20/W20/D23, Bayes accuracy 0.795, native 0.472.

| arm | acc | J slope | D0/DM | trainable | rule | result |
|---|---|---|---|---|---|---|
| L2 | 0.721 | −0.001 | 1.01 | 50,048 | < 0.2 | pass |
| L2plus | 0.728 | 0.018 | 0.74 | 99,776 | < 0.2 | pass |
| L1only | 0.717 | 0.003 | 0.88 | 49,728 | < 0.2 | pass |
| L0only | 0.790 | 0.913 | 0.17 | 49,728 | ≥ 0.6 | pass |
| supL2 | 0.730 | 0.028 | 0.56 | 50,240 | < 0.2 | pass |
| supG | 0.753 | 0.277 | 0.18 | 50,240 | ≥ 0.6 | **FAIL** |
| early01 | 0.785 | 0.927 | 0.15 | 99,456 | (R5 input) | — |

Early unit uP163 L22/W22/D21:
- L2: acc 0.760, J slope 0.378 → **FAIL** (borderline);
- early01: acc 0.784, J slope 0.833 → pass.

Review 2 notes that the "fail-predicted" arms would also pass if training had done nothing. All four trained: accuracy rose from 0.47 to
0.72–0.73, against Bayes 0.795.

*Exploratory, post-freeze, not scored* (`logs/adapt3_expl_s3000/`): on the same unit,
- with 3000 steps, supG reaches acc 0.790, J slope 0.895, and L2 stays at −0.002;
- with lr 3e-3 and 1000 steps (`logs/adapt3_expl_lr3e-3/`), supG reaches 0.791 / 0.934, and L2 stays at −0.001.

The frozen failure of supG is therefore a learning-speed failure at the frozen budget. The per-group statistic is usable by a layer-2
repair. The group-agnostic statistic and the plain layer-2 repair do not reach it.

**R5.** Late unit, early01 adapted:
- learned DAS at the q + readout tokens: k = 3 → 0.80, k = 8 → 0.88;
- interchange along the m-probe span: 0.009; along the responsibility-probe span: 0.009;
- yet m is decodable after adaptation (R² 0.70–0.82);
- frozen layer-2 rows Shapley 0.815;
- corr(Δ group mass, Δ exact role posterior) 0.84 and 0.89 for the two mixture heads.

**R6.**

| unit | TRUE R² | SHUFD | SHUF | PMEAN | SHUFB | OTHER | no-op max \|Δlogit\| |
|---|---|---|---|---|---|---|---|
| uP161 L20/D23 (the only pure mixture unit) | 0.971 | 0.010 | 0.029 | 0.046 | 0.036 | 0.009 | 7.2e-7 |
| uP161 L23/D20 (hybrid; not diagnostic) | 0.813 | 0.013 | | | | 0.131 | |
| uP163 L21/D22 (hybrid; not diagnostic) | 0.627 | | | | | 0.280 | |
| uP163 L22/W21/D21 (hybrid; not diagnostic) | 0.238 | | | | | 0.331 | |

In the pure mixture unit, the frozen verdict failed only on the strict no-op equality.

The no-op residue is float32 reordering: the row-patch path recomputes attention through a different einsum. It is ≤ 1e-6 in every unit,
while the d_das no-op is exactly 0. The frozen criterion said "= 0" and is reported as failed. Scientifically, the specificity claim
survives in the one diagnostic unit; the no-op is valid at tolerance 1e-6, not exactly.

**R7.** uP163 L22/W22/D21, ALL2, k = 8:
- source-specific excess 0.355;
- R² E vs A2 on E5 0.89 vs 0.67, E3 0.58 vs 0.00, QF4 0.94 vs 0.26;
- QF5 E vs A1 0.91 vs 0.35;
- random subspaces ≤ 0.25;
- a matched-budget A2-trained subspace reaches R²_A2 only 0.06 / 0.02 on E5 / E3;
- no-op exactly 0.

**R8.** Sufficiencies (Q / R / O) and redundancy:

| unit | Q / R / O | redundancy |
|---|---|---|
| uP161 L20/D23 (mixture) | 0.54 / 0.45 / 0.02 | −0.000 |
| uP163 L22/D21 (early) | 0.13 / 0.51 / 0.39 | 0.035 |
| uP163 L21/W22/D22 (early) | 0.07 / 0.43 / 0.50 | −0.002 |

## 4. Borderline cases (pre-declared procedure: second stream 88_150_000; the verdict uses the first stream)

| quantity | first stream | second stream | threshold | change of arm |
|---|---|---|---|---|
| R2c uP161 L23/D20 final lik_RO1 | 0.242 | 0.232 | ≤ 0.2 | no (still fails) |
| R1a uP161 L23/D20 d_init | 0.20 | (same pairs) | ≤ 0.25 | no |
| R4 early L2 J slope | 0.378 | not re-measured (one training run) | ≥ 0.4 | — |

## 5. Statistical units and what was not computed

- Factorial effects are descriptive contrasts of single runs per cell (`logs/analyze_A_final.json`). Init main effect on lik_RO1:

  | parent | init effect | data effect | interaction |
  |---|---|---|---|
  | uP160 (dev) | 0.79 | 0.01 | 0.03 |
  | uP161 (dev) | 0.77 | 0.00 | 0.00 |
  | uP161 (conf) | 0.33 | 0.09 | 0.18 |
  | uP163 (conf) | 0.50 | 0.06 | 0.11 |

- With n = 1 per cell no test is possible. The data effect is small relative to the init effect in all four designs, but is **not shown to
  be zero**: 0.06–0.09 in the confirmation parents.
- Episode-bootstrap intervals cover measurement noise only, not training-seed variation.
- No interval is computed across parents.
- Equivalence is never inferred from a non-significant difference. "Follows the initialisation" is a statement about relative distances
  with bootstrap intervals that do not overlap.
