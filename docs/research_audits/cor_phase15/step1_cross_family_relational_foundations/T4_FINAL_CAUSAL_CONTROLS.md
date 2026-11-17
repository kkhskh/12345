# T4 final causal controls (XIV Stage I)

**Status.** COR Phase 1.5, Step 1 (≈ 25%, unchanged).

**Freeze and scoring.**
- FREEZE I: commit `aca3b6c` (`logs/FREEZE_I.md`).
- Scoring: commit `18e0b73`.
- Scorer: `scripts/score_I.py`, fail-closed. Preflight P0–P5 passed. Output: `logs/score_I.json`.

**Cohort.**
- One parent, uP160.
- Two bases: E (early organisation, L = W = 21) and M (mixture, L = W = 23).
- Three data orders per base. The statistical unit is the base × data-order pair (n = 6). These pairs are not independent pretraining replications.

**Compute.** 38 new training runs and 178 measurements.
- The container restarted once, at about 22:34Z. Two continuation runs resumed bitwise from their checkpoints.
- Four step-2000 checkpoints of XIII-E continuations had been excluded from the XIII-E archive. They were missing at FREEZE I and were silently skipped by `unit_metrics.py`. They were restored from the session's intermediate saves (a backup with XIII-E-era timestamps) and measured by hand with the frozen command. Their hashes are in `logs/I_restored_checkpoints.txt`; review C verified them.
- All replays of the baseline trajectories are bitwise identical to the originals at step 600.

## I-A. Off-route sham and location-matched disturbance

**Interventions** (FREEZE_I). All share identical L, W and D tensors, and all act during steps 0–100.

**Real block** (reused XIII-E / XIII-D runs, bitwise reproducible):
- E: layer-0 readout → all sources;
- M: layer-2 readout → own-question sources.

**Off-route sham.** The same −1e9 masking mechanism, in the same layer and on the same readout queries, applied to edges outside both routes:
- E: layer-0 readout → other-question sources;
- M: layer-2 readout → readout and episode tokens.

**Location-matched disturbance.** Gaussian logit noise on exactly the real block's edges, matched in attention total variation (σ_E = 2.5, σ_M = 1.25).

**Sham validity** (frozen check, confirmation stream, all 6 baselines at step 1211): **valid on all 6.**
- Applying the sham raises KL by 0.000–0.012. The real block raises it by 0.60–1.38.
- Neither route's own-question attention is reduced.

**Results at step 1211:**

| base, D | baseline | real block | off-route sham | location-matched noise |
|---|---|---|---|---|
| E, 21 | EARLY | MIXTURE (δ 0.80) | EARLY (δ −0.01) | EARLY (δ 0.17) |
| E, 801 | EARLY | MIXTURE (0.79) | EARLY (−0.01) | EARLY (0.02) |
| E, 802 | EARLY | MIXTURE (0.77) | EARLY (−0.03) | EARLY (−0.02) |
| M, 23 | MIXTURE | EARLY (0.80) | MIXTURE (0.00) | MIXTURE (0.01) |
| M, 801 | MIXTURE | EARLY (0.79) | MIXTURE (−0.01) | MIXTURE (−0.01) |
| M, 802 | MIXTURE | EARLY (0.81) | MIXTURE (0.00) | MIXTURE (−0.01) |

Notes on the table:
- δ is the shift in mix_mass towards the predicted direction, relative to the paired baseline. Its maximum possible size is about 0.8.
- Every run is competent: KL 0.011–0.036 at step 1211.

**Verdicts.**
- **R-S: CONFIRMED.** The real block flips 6/6 and the sham flips 0/6. Paired δ_real − δ_sham = 0.80, 95% bootstrap CI [0.795, 0.804].
- **R-N: CONFIRMED.** Noise flips 0/6. Paired difference 0.76, CI [0.71, 0.80].

**Persistence at step 4000** (D = 801, both bases):
- The real-block flips persist: δ 0.80 on E and 0.80 on M, classes MIXTURE and EARLY.
- The sham and noise runs stay in the baseline class: |δ| ≤ 0.013, KL ≤ 0.004.

**Answers to mandate questions 1 and 2.**
- **The route-blocking effect survives an actual off-route sham.** It also survives a location-matched disturbance of equal attention-TV severity. Within this cohort, blocking the pathway itself changes the learned organisation, and comparable interventions that do not block it do not.

**Explanations that remain open:**
- **Neither control is neutral (post-review, review B A1).** The sham concentrates softmax mass on the route edges: on E, own-question attention rises from 0.134 to 0.33 during steps 0–100. Review B measured that the location-matched noise *also* raises the route's share (E 0.134 → 0.22; M 0.142 → 0.23). Both controls therefore push towards the base route, and a no-flip result is expected for them.
  - They rule out "any masking or logit perturbation of these readout edges" as the cause.
  - They do **not** test "any early reduction of the route's attention share or access". That needs a control that diverts attention while keeping every route edge reachable, and it was not run.
  - The sham-validity check was made on trained step-1211 models, not during steps 0–100, and it is one-sided (review B A2).
  - On M, the outcome measure mix_mass *is* attention on the intervened edges. The M flips are also shown by lik_RO1: ≈ 0.75 against ≈ 0.04 in the baseline (review B A5).
- **The noise is matched in attention TV, not in loss.** On E its immediate loss disruption is about one third of the block's: Δloss 0.18 against 0.55 at step 100. A loss-matched noise on E would need TV close to 1, which approaches a block. On M the block itself barely changes the loss (Δ 0.011), so loss disruption cannot be what causes the M flip.
- **One parent, two bases, three data orders.**

## I-B. Do early changes persist more than late changes?

**Design.**
- The same operation and the same 100-step duration: early block over steps 0–100, late block over steps 546–646. The late runs branch from the baseline's step-546 state.
- Each block is compared with its step-matched baseline at equal numbers of updates after release: r ∈ {0, 50, 200, 500, 1111, 1900}, plus the final step 4000.

**Shift δ(r) after release** (oriented as above; `logs/score_I.json`):

| base, D | early: δ at r = 0 / 50 / 200 / 500 / 1111 / 1900 → final, class | late: δ at r = 0 / 50 / 200 / 500 / 1111 / 1900 → final, class |
|---|---|---|
| E, 21 | −0.01 / 0.02 / 0.48 / 0.77 / 0.80 / 0.79 → 0.80, MIXTURE | 0.01 / 0.01 / 0.00 / 0.00 / 0.00 / 0.00 → −0.01, EARLY |
| E, 801 | −0.01 / 0.04 / 0.43 / 0.76 / 0.79 / 0.80 → 0.80, MIXTURE | 0.00 / 0.03 / 0.01 / 0.01 / 0.00 / 0.01 → 0.00, EARLY |
| E, 802 | −0.01 / 0.03 / 0.45 / 0.68 / 0.77 / 0.78 → 0.79, MIXTURE | 0.06 / 0.21 / 0.43 / 0.67 / 0.73 / 0.75 → 0.76, HYBRID |
| M, 23 | 0.03 / 0.10 / 0.45 / 0.75 / 0.80 / 0.81 → 0.81, EARLY | **0.29** / 0.04 / 0.02 / 0.02 / 0.00 / 0.00 → 0.00, MIXTURE |
| M, 801 | 0.03 / 0.10 / 0.62 / 0.74 / 0.79 / 0.79 → 0.80, EARLY | **0.24** / 0.05 / 0.02 / 0.01 / 0.00 / 0.00 → 0.00, MIXTURE |
| M, 802 | 0.04 / 0.09 / 0.33 / 0.75 / 0.81 / 0.81 → 0.82, EARLY | **0.31** / 0.01 / 0.00 / 0.01 / 0.02 / 0.01 → 0.01, MIXTURE |

**Persistence index.** PI = δ(1900) / max δ(r ≤ 500).
- Early runs: 1.03–1.15 in all 6.
- Late runs: 0.00–0.02 on M, 1.11 on E-802, and undefined on E-21 and E-801, because those runs never shifted by 0.1.

**R-P: CONFIRMED** under the frozen rule.
- 4 valid pairs; PI_early > PI_late in 4/4.
- Mean difference 0.81, 95% bootstrap CI [0.30, 1.08].
- The E-802 pair contributes only +0.04.
- The two E pairs whose late block had no effect were excluded by the frozen rule, because PI is undefined when nothing changed.

**What the data show** (answer to mandate question 3):

1. **Early 100-step blocks.**
   - On both bases and in all data orders, the organisation was not yet formed at release (δ ≈ 0 at r = 0).
   - The redirected organisation then develops over the next 200–500 updates and persists to step 4000 (δ ≈ 0.8).
   - The effect is therefore not an "immediate change that persists". The early block determines which organisation forms.
2. **Late 100-step blocks on M.**
   - They cause an immediate partial shift (δ 0.24–0.31 at release).
   - The shift reverses within 50–200 updates, to zero by r = 500, and leaves no trace at step 4000.
   - XIII-E's 600-step late blocks on M left a lasting hybrid. The duration of the late block therefore matters.
3. **Late 100-step blocks on E.**
   - In D = 21 and 801 the mixture-mass shift δ stays ≈ 0. This is **not** "no effect" (post-review, review B A6): at release lik_RO1 drops from 0.69–0.71 to 0.09–0.15 and KL rises about 6×, then the early route recovers within about 200 updates. The block transiently disrupts the early route, and the disruption reverses.
   - In D = 802 the run flips gradually towards mixture and ends HYBRID (δ 0.76).
   - So a late block on E can leave a lasting change, but only in one data order.
4. **Interpretation (reframed after review B A7).** PI means different things in the two arms:
   - The early block acts *during formation*. The organisation develops only after release, so PI_early ≈ 1 almost by construction.
   - The late block acts *after formation*, and PI_late measures how fast the formed organisation recovers.

   The supported statement: **a 100-step block during formation selects the organisation permanently, while the same block after formation is mostly reversed.** This holds on M in 3/3 runs and on E in 2/3. The frozen R-P verdict rests mainly on M: 3 of the 4 valid pairs. The one valid E pair is close to a tie (+0.04). The intervals describe 4–6 related pairs from one parent.
   - It does not support a closed critical period: XIII-E's longer late blocks and E-802 show that lasting late changes are possible.

**Remaining confounds:**
- **Learning-rate exposure.** The late block sees about 4× more learning-rate mass (Σ lr 0.097 against 0.025, because warm-up ends at step 200). Exposure from release to the end is larger for the early block (1.98 against 1.46).
  - Both differences cut in the same direction for M: a stronger late block reverses anyway.
  - For E the larger post-release budget of the early block could contribute to its persistence.
- **"Immediate magnitude" is not comparable.** The early organisation does not exist at release, so the early block's immediate δ is about 0 by construction. PI normalises by the peak shift within 500 updates instead.
- **Sample size.** One parent; n = 6 pairs, of which 4 enter R-P.

## Final interpretation

**Supported in this cohort** (frozen confirmation, one parent):
- Blocking a decision route for the first 100 steps changes the learned organisation reproducibly (6/6). It does so beyond an off-route sham of the same mechanism and beyond location-matched noise of equal attention-TV severity.
- The change is persistent to the end of training.
- The same block applied late (steps 546–646) is reversed on M and mostly ineffective on E.

**Not established:**
- That the specific removal of *information access*, rather than any reduction of the route's attention share, is the cause. Both controls increase that share.
- The minimal own-question-only block on E (review B A3).
- A closing critical period.
- Generality beyond uP160.
