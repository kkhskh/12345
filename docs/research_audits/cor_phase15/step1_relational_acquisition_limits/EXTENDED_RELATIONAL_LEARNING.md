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

**My frozen expectation for P2 was CONT_SELECTIVE, with no FAST_UNLOCK. It was wrong** [frozen]. It rested on one development model, XV's development D_s100, which did not unlock within 2050 post-restart steps. The confirmation shows that this development model was typical of the **D** condition (2 of 3 D continuations also never unlocked) but not of the **C** condition.

## 3. Comparison across arms on the exposure axis [frozen numbers; cross-arm comparison exploratory]

Acquired / evaluable by cumulative exposure. A censored run is evaluable at a threshold only if its observed exposure reached it (`logs/xvi_stats.json`):

| arm | ≤ 1.0 | ≤ 2.0 | ≤ 3.0 | ≤ 4.0 | ≤ 5.8 |
|---|---|---|---|---|---|
| XVI fresh, constant, C | 1/4 | 2/4 | 3/4 | 4/4 | 4/4 |
| XVI restart, C (total exposure, including XV's 2.0) | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 |
| XVI restart, D | 0/3 | 0/3 | 0/3 | 1/3 | 1/3 |
| XV cosine, C (exploratory, corrected definitions) | 0/4 | 1/4 | — | — | — |
| XV cosine, D (exploratory) | 0/4 | 1/4 | — | — | — |

**Reading [inference].**

- **Condition C (gate value 0.277, weak agreement cue).**
  - Given more effective optimisation, every model tested acquired the gate: 4 fresh models and the 3 restarted XV failures.
  - Acquisition happened at total exposures between 1.0 and 3.9. XV's whole run had only 2.0, and its last quarter had a rate below 30% of the peak.
  - **XV's censoring in C is therefore attributable to insufficient effective optimisation (Explanation A),** not to an inability of the architecture or of these particular seeds.
  - The fresh constant-rate models reached the gate at about the same exposures as the restarted ones (≤ 3.0: 3/4 against 2/3). So, as far as n = 7 can show, nothing points to the restart *itself* doing anything beyond restoring exposure.
- **Condition D (gate value 0.138, strong agreement cue, cue overlap 0.88).**
  - 2 of 3 restarted XV failures stayed at the gate-free optimum through a further 4000 steps at the full rate (total exposure 5.9, about 3× XV's), together with development model D_s100 (another 2050 steps).
  - During those steps their parameters kept moving. The mean update norm on the plateau was 0.049–0.058 per step, against 0.033–0.040 in the fresh C models, and the parameter norm grew from about 50 to 52.
  - So this is **not** an optimizer that has stopped moving. **Under this procedure, and for these selected seeds, a finite-budget barrier persists (Explanation B), selectively.**
  - No fresh D models were trained (budget; FROZEN_PROTOCOL_XVI.md §3), so the D result rests on 3 + 1 selected models. **It cannot say how common the barrier is in D.**
- **Restart against constant rate.** The four restart successes came 400–1950 steps after the restart. Fresh constant-rate models needed 1100–3500 steps from initialisation, and the restarted models began already at the gate-free optimum.
  - The design cannot separate "the rate was restored" from "the Adam moments were reset", because XV saved no optimizer state. It also cannot say whether exposure accumulated under a decaying rate counts like exposure at the peak rate [design limit].

## 4. Classification (mandate A8)

**A2, extended learning succeeds selectively, with a strong condition dependence:**
- **Condition C:** extended optimisation succeeds broadly (A1-like): 7 of 7, including all 3 of XV's C failures.
- **Condition D:** 1 of 3 restarted failures succeeded. The other 2 remain at the gate-free optimum despite substantial parameter updates (A3-like for those seeds).

The frozen verdicts (P1 BROAD, P2 CONT_BROAD) describe the cohorts as preregistered. The split by condition above is post hoc and descriptive.

## 5. Abruptness under dense measurement (D4) [frozen data; exploratory reading]

- **Fresh constant rate (diagnostics every 50 steps):**
  - F_G rose from ≤ 0.1 to ≥ 0.9 within 150–200 steps (3–4 intervals) in all 4 models.
  - Before that, each spent 800–3200 steps at the gate-free optimum with F_G < 0.2.
- **Restart arm:** widths of 100, 200 and 400 steps, plus one exception, **C_s202**.
  - C_s202's F_G crossed 0.5 at step 400, fell back below 0.5 once, and took 1850 steps to pass 0.9.
  - **That transition was not abrupt.** It is the only regression among the 16 acquisitions in the confirmation set.
- **Workstream B (every 25 steps):** widths of 75–250 steps.
- **XV's arity-3 transitions,** previously reported as "less abrupt" (600–750 steps), are not comparable. Their 250-step spacing cannot resolve them, and the XVI data replace them: of the 8 arity-3 acquisitions measured with 50-step diagnostics, 6 took 100–200 steps, one took 400 and one (C_s202) took 1850.
- **The jumps are not explained by sharpening.** F_G cannot be raised by a confidence change: it is exactly 0 for the gate-free optimum at every temperature. Mean confidence rose from 0.785–0.81 to 0.90–0.91 across the arity-3 transitions, as expected when a correct decision rule is adopted. The rise came together with the decision change, not instead of it.

## 6. Optimizer behaviour (A6) [frozen data; exploratory summaries]

| run | plateau gradient norm | plateau update norm | transition gradient norm | transition update norm |
|---|---|---|---|---|
| fresh constant (4) | 0.23–0.34 | 0.033–0.040 | 0.29–0.64 | 0.064–0.117 |
| restart C (C_s201) | 0.20 | 0.055 | 0.24 | 0.110 |
| restart D, censored (2) | 0.11–0.12 | 0.049–0.058 | — | — |
| restart D, acquired (D_s204) | 0.11 | 0.057 | 0.16 | 0.124 |

- Update norms roughly double during transitions.
- On the plateau, the censored D models had gradient norms and update magnitudes **indistinguishable** from D_s204 before it acquired the gate.
- The rebuilt learning-signal estimator found **no first-order gate-attributable expected gradient** on these plateaus (LEARNING_SIGNAL_AND_STAGE_VALIDATION.md §6).

## 7. Representability (A7)

- **Mathematically:** the gate has decision order a + 1 = 4 [proof]. The order-4 log-linear class contains Bayes [num].
- **Architecturally:** the tested Transformer represents it. Arity-3 models of exactly this architecture reached KL from Bayes of 0.0003–0.0013 with F_G = 1.00: 4 fresh models, 4 restarted models, and XV's C_s204 and D_s203 [frozen].
- **Within the seeds tested in D:** D_s204, a model that sat at the gate-free optimum for 4000 XV steps plus 1900 restart steps, then acquired the gate. So "at the gate-free optimum for a long time" does not imply "cannot acquire".
- **The two persisting D models** are therefore a case of **optimisation failure within a finite budget**, not representational or architectural impossibility. I make no claim that they would never acquire the gate.
