# Extended relational learning (Workstream A)

**Evidence labels:** [proof], [num], [dev], [frozen], [expl], [spec].

**Question.** XV's six censored arity-3 models stopped at the gate-free rule under a cosine schedule that decays to zero. Were they unable to acquire the gate, or did they run out of effective optimisation?

## 1. Arms (FROZEN_PROTOCOL_XVI.md §2)

| arm | models | schedule | budget | independence |
|---|---|---|---|---|
| **original** (XV, existing) | C_s201–204 and D_s201–204 (arity 3; ν = 1/3 and 0.05) | warm-up then cosine to 0 at step 4000 | 4000 steps, exposure 2.00 | 8 XV models (4 initialisations, each used in both C and D) |
| **constant rate** (fresh) | 4 new models, seeds 401–404, condition C (3:0.9:0.9:1/3) | warm-up 200, then constant 1e-3 | 15,000 steps, exposure 14.9 | **4 independent initialisations and streams** |
| **learning-rate restart** | the 6 censored XV models: C_s201, s202, s203; D_s201, s202, s204 | from XV step 4000; new AdamW state; warm-up 200, then constant 1e-3; XV's data stream continued | 4000 more steps (total exposure 5.90) | 6 distinct trained models. **Not fresh replications; selected as XV's failures** |
| **schedule-matched continuation, no restart** | — | the XV schedule continued | — | **not run: provably inert** [proof + num]. The cosine rate is exactly 0 from step 4000, and optax's AdamW scales both the Adam step and the weight decay by the rate, so every update is exactly 0 (checked numerically: update = −0.0) |

**Exposure** is the sum of the applied learning rates, in units of steps × rate, and is reported alongside steps.

## 2. Frozen results

**P1, fresh constant rate: BROAD, 4 of 4 acquired** [frozen]:

| model | t_first | t_A (sustained acquisition) | exposure at t_A | width (F_G 0.1 → 0.9) | regressions | time near gate-free optimum | final KL / F_G |
|---|---|---|---|---|---|---|---|
| s401 | 1550 | 1600 | 1.50 | 150 | 0 | 1300 steps | 0.0006 / 1.00 |
| s402 | 1050 | 1100 | 1.00 | 200 | 0 | 800 | 0.0003 / 1.00 |
| s403 | 3000 | 3000 | 2.90 | 150 | 0 | 2800 | 0.0005 / 1.00 |
| s404 | 3400 | 3500 | 3.40 | 150 | 0 | 3200 | 0.0012 / 1.00 |

- Acquisition proportion 4/4 (exact Clopper–Pearson 95% interval 0.40–1.00).
- Kaplan–Meier by exposure: 1/4 by 1.0, 2/4 by 2.0, 3/4 by 3.0, 4/4 by 3.4.
- All 4 were competent at the end, and none regressed.

**P2, learning-rate restart: CONT_BROAD, 4 of 6 acquired, with FAST_UNLOCK** [frozen]:

| XV model | steps after restart to t_A | total exposure at t_A | width | regressions | end state |
|---|---|---|---|---|---|
| C_s201 | 1200 | 3.10 | 100 | 0 | competent (KL 0.0011) |
| C_s202 | 400 | 2.30 | **1850** | **1** | competent (KL 0.0013) |
| C_s203 | 550 | 2.45 | 400 | 0 | competent |
| D_s201 | — | censored at 5.90 | — | — | at the gate-free optimum (KL 0.139, F_G 0.00) |
| D_s202 | — | censored at 5.90 | — | — | at the gate-free optimum (KL 0.140, F_G 0.00) |
| D_s204 | 1950 | 3.85 | 200 | 0 | competent |

- Acquisition proportion 4/6 (95% interval 0.22–0.96).
- The median post-restart t_A of the acquirers is 875 steps (FAST_UNLOCK: ≤ 1000).
- **By condition:** C 3/3; D 1/3.
- Matched-exposure comparison with the fresh cohort at E* = 5.90: 4/6 against 4/4, so CONT_LOWER (Fisher exact p = 0.47; no inference of a difference).

**Frozen expectation for P2.** The frozen record is inconsistent (ADJUDICATION W8).
- The protocol states CONT_SELECTIVE with no FAST_UNLOCK, revised before the freeze after development model D_s100 failed to unlock within 2050 post-restart steps. Against it, the outcome is a miss.
- The hashed scorer's docstring still states CONT_BROAD. Against it, the outcome is a hit.

**Fragility.** FAST_UNLOCK is true only at the frozen 0.5 threshold (false from 0.51; Review B), because C_s202's t_A rests on 3 of 6 slots. The count 4/6 is robust at every threshold.

## 3. Comparison across arms on the exposure axis [frozen numbers; cross-arm comparison exploratory]

Acquired / evaluable by cumulative exposure. A censored run is evaluable at a threshold only if its observed exposure reached it (`logs/xvi_stats.json`):

| arm | ≤ 1.0 | ≤ 2.0 | ≤ 3.0 | ≤ 4.0 | ≤ 5.8 |
|---|---|---|---|---|---|
| XVI fresh, constant, C | 1/4 | 2/4 | 3/4 | 4/4 | 4/4 |
| XVI restart, C (total exposure, including XV's 2.0) | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 |
| XVI restart, D | 0/3 | 0/3 | 0/3 | 1/3 | 1/3 |
| XV cosine, C (exploratory, corrected definitions) | 0/4 | 1/4 | — | — | — |
| XV cosine, D (exploratory) | 0/4 | 1/4 | — | — | — |

**Reading [inference; corrected after review, reviews/ADJUDICATION.md W2, W3].**

- **Condition C (gate value 0.277, weak agreement cue).**
  - Given more effective optimisation, every model tested acquired the gate: 4 fresh models and the 3 restarted XV failures.
  - Acquisition happened at total exposures of 1.0–3.9, against XV's whole run of 2.0, which spent its last quarter at a rate below 30% of the peak.
  - **XV's censoring in C is attributable to insufficient effective optimisation (Explanation A),** not to an inability of the architecture or of these seeds.
- **Condition D (gate value 0.138, strong agreement cue).**
  - 2 of 3 restarted XV failures were still at the gate-free optimum after a further 4000 full-rate steps (total exposure 5.9). Their parameters kept moving: plateau update norms of 0.049–0.058 per step.
  - **This does not establish a barrier** (Review B, finding 3). An escape rate estimated from the D runs themselves (about 1 per 10,000–12,000 plateau steps) gives a 45–51% chance that both stay censored for 4000 steps.
  - Their plateau statistics look like those of C models before escape.
  - **They are right-censored. Late learning and a persistent barrier cannot be distinguished.**
  - No fresh D models were trained.
- **Restart against constant rate: not separable** (Review B finding 4, Review A finding 9).
  - On total exposure the routes look comparable. On exposure since the restart the restarted models look faster (descriptive p = 0.11, n = 3 against 4).
  - The frozen FAST_UNLOCK descriptor fired, but only at the 0.5 threshold. The frozen matched-exposure call was CONT_LOWER (Fisher p = 0.47).
  - The restart is confounded with an Adam moment reset (XV saved no optimizer state), which leaves a visible signature in the update norms.
  - Whether exposure accumulated at a decaying rate is equivalent to exposure at the peak rate is not known.

## 4. Classification (mandate A8; corrected after review)

| condition | outcome | classification |
|---|---|---|
| **C** | 7 of 7 acquired, including all 3 of XV's C failures | **A1:** extended learning succeeds broadly |
| **D** | 1 of 3 restarted failures acquired; 2 are right-censored with substantial updates | **A4:** inconclusive |

Overall: **A1 in C, A4 in D.** The frozen verdicts (P1 BROAD, P2 CONT_BROAD) describe the cohorts as preregistered. The split by condition is post hoc and descriptive.

## 5. Abruptness under dense measurement (D4) [frozen data; exploratory reading; corrected after review]

**Acquisition is slot-local** (Review B finding 1, re-checked; ADJUDICATION W5).
- The model answers 6 question slots per episode. F_G pools gate flips over them, and the slots acquire the gate **separately**.
- Intermediate plateaus of pooled F_G are whole numbers of acquired slots.
- **WA_cont_C_s202 is the clearest case.** It held F_G = 0.502, i.e. 3 of 6 slots, from step about 350 to about 2000, then the remaining slots jumped.
  - Its frozen t_A = 400 rests on 0.50187 against the 0.5 threshold, about 6 of 3471 decisive flips.
  - Its one "regression" (0.4987) is a fluctuation of about 11 flips.
  - Its "1850-step transition" is two sharp jumps, not a gradual process.

**Per slot, every observed acquisition is sharp.**
- Pooled F_G widths (≤ 0.1 to ≥ 0.9) are 75–400 steps in 15 of 16 acquisitions: fresh 150–200, restart 100–400, WB 75–250.
- These widths are **lower bounds**. F_G is close to binary on this task (Review A finding 4). The graded G gives 175–450 steps.

**Sampling.** Diagnostics every 25–50 steps resolve all of these transitions. XV's earlier "less abrupt at arity 3" (250-step spacing) is superseded.

**Confidence.** No acquisition can be produced by a confidence change: t_A requires KL ≤ 0.75 of the gate-free KL. Mean confidence rose from 0.785–0.82 to 0.85–0.91 across the arity-3 acquisitions, together with the decision change.

## 6. Optimizer behaviour (A6) [frozen data; exploratory summaries; `scripts/xvi_optimizer_summary.py` → `logs/expl/optimizer_summary.json`]

| run | plateau gradient norm | plateau update norm | transition gradient norm | transition update norm |
|---|---|---|---|---|
| fresh constant (4) | 0.23–0.34 | 0.033–0.040 | 0.29–0.64 | 0.064–0.117 |
| restart C (C_s201) | 0.20 | 0.055 | 0.24 | 0.110 |
| restart D, censored (2) | 0.11–0.12 | 0.049–0.058 | — | — |
| restart D, acquired (D_s204) | 0.11 | 0.057 | 0.16 | 0.124 |

- Update norms roughly double during transitions.
- On the plateau, the censored D models had gradient norms and update magnitudes indistinguishable from D_s204 before it acquired the gate.
- The rebuilt learning-signal estimator found the gate-attributable expected gradient **below its detection limit** (about 1–3% of the expected gradient) on these plateaus (LEARNING_SIGNAL_AND_STAGE_VALIDATION.md §6.2).

## 7. Representability (A7)

- **Mathematically:** the gate has decision order a + 1 = 4 [proof]. The order-4 log-linear class contains Bayes [num].
- **Architecturally:** the tested Transformer represents it. Arity-3 models of exactly this architecture reached KL from Bayes of 0.0003–0.0013 with F_G = 1.00: 4 fresh models, 4 restarted models, and XV's C_s204 and D_s203 [frozen].
- **Within the seeds tested in D:** D_s204, a model that sat at the gate-free optimum for 4000 XV steps plus 1900 restart steps, then acquired the gate. So "at the gate-free optimum for a long time" does not imply "cannot acquire".
- **The two censored D models** are not acquired *within the budget*. That is not representational or architectural impossibility, and **I make no claim that they would never acquire the gate**: their censoring is consistent with late learning (§3).
