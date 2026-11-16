# FREEZE I: XIV Stage I bounded T4 controls (committed before any Stage I confirmation output exists)

**Hypotheses.**

- **I-A.** Blocking a route for the first 100 steps changes the learned organisation more reliably than (i) an off-route sham that uses the same masking mechanism, and (ii) a disturbance at the same attention location matched in severity.
- **I-B.** After an equal number of updates since release, changes from the early block (steps 0–100) are more persistent than changes from the same block applied late (steps 546–646).

**Competing predictions.**

- **Masking in general.** If the effect comes from masking the readout queries of that layer at all, the off-route sham also flips the organisation.
- **Disruption at the site.** If disrupting the route's attention site is enough, without removing access, the location-matched noise also flips it.
- **Same persistence.** If early and late changes persist equally, PI_early ≈ PI_late.

## Models and units

- **Parent.** One parent, uP160. The two bases (XIII-D/E):
  - E (early organisation): L = W = 21;
  - M (mixture organisation): L = W = 23.
- **Data orders.** D ∈ {anchor (21 or 23), 801, 802}.
- **Matched tensors.** Matched conditions share identical L, W and D tensors and data streams.
- **Statistical unit.** The base × data-order pair, 6 in total. All share one parent, and within a base the 3 pairs share the L and W initialisation. They are therefore not independent pretraining replications.

## Interventions

All interventions run during steps 0–100, except the late arm.

**Real block** (reused XIII-E/XIII-D runs, bitwise reproducible):
- E: ro0, layer-0 readouts → all 36 source tokens, bias −1e9.
- M: l2q, layer-2 readout q → its own 6 source tokens.

**Off-route sham.** The same −1e9 bias mechanism, in the same layer and on the same readout queries, applied to edges outside both routes:
- E: `oq0`, layer-0 readout q → the 30 sources of other questions.
- M: `rd2`, layer-2 readout → readout and episode tokens.

Chosen in development (`logs/i_attn_dev.json`, `logs/i_sham_dev.json`):
- At trained bases (steps 546 and 1211) these masks change KL by ≤ 0.05 and do not reduce the own-question route attention (they raise it).
- Other candidates were rejected:
  - `oq2` on E raises the layer-2 own mass from 0.17 to 0.78. That would act as a pro-mixture push.
  - `rd0` on E is more disruptive (KL +0.04).
- Blocked attention mass at onset (steps 25–100):
  - E sham 0.44–0.46 against the real block's 0.60–0.68;
  - M sham 0.18–0.20 against the real block's 0.14–0.17.
- **Known asymmetry.** Because softmax renormalises, both shams concentrate attention on the remaining edges, including the route edges. They therefore reinforce the base organisation rather than leave it untouched.

**Location-matched disturbance.** Gaussian noise σ·N(0,1) added to the attention logits on exactly the real block's edge set, in the same layer, fresh per step and per element (key `PRNGKey(525252)` folded with the step).
- σ_E = 2.5, σ_M = 1.25.
- **Severity matching (frozen).** The mean total-variation distance between the perturbed and unperturbed readout-query attention rows of the intervened layer, averaged over base checkpoints at steps 25, 50 and 100 (`logs/i_calib.json`).

  | | real block TV | noise TV at σ | immediate Δloss at step 100 (real / noise) |
  |---|---|---|---|
  | E | 0.62 | ≈ 0.62 at σ = 2.5 (interpolated) | 0.55 / ≈ 0.18 |
  | M | 0.15 | ≈ 0.15 at σ = 1.25 | 0.011 / ≈ 0.003 |

  Matching is in attention TV, not in loss. On E the noise disrupts the loss less than the block does, and this is reported.

**Late block (I-B).** The same operation (ro0 on E, l2q on M) and the same 100-step duration over steps 546–646, branched from each baseline's step-546 optimiser and data-stream state.

**Step-matched baselines** at the late-recovery steps are bitwise replays of each baseline trajectory, branched from saved optimiser states. Steps 100–600 of the D = 801/802 XIII-E runs were restored from the XIII-E intermediate saves; their hashes are in `logs/I_restored_checkpoints.txt`, and the step-1211 copies match the XIII-E manifest.

**Harness.** `scripts/ft_factorial_c.py`, which is `ft_factorial_b.py` plus the `msk` and `lnz` interventions and extra snapshot steps. Two bitwise checks were made, logged as development in COMPUTE_LOG:
- `msk` with the real ro0 edge set reproduces the XIII-E ro0 run at step 25 (|Δ| = 0);
- `lnz` with σ = 0 reproduces the baseline (|Δ| = 0).

## Measurements

- **Tool.** `unit_metrics.py` with `UM_RHO=0 UM_STREAM=95300000 UM_TAG=_I`. This confirmation stream is unused in development.
- **Units and steps.** 178 (unit, step) pairs, listed in `logs/I_design.json`.
- **Classes at step 1211** (the XIII-E rule): EARLY if mix < 0.5 and lik ≥ 0.4; MIXTURE if mix ≥ 0.8 and lik ≤ 0.15; HYBRID otherwise.
- **Classes at step 4000:** EARLY if lik ≥ 0.6 and mix < 0.5.
- **Competence:** KL ≤ 0.05 at step 1211; KL ≤ 0.02 at step 4000.
- **Shift δ.** δ = shift in mix_mass against the step-matched baseline, oriented towards the predicted direction: +mix on E, −mix on M. The likelihood counterpart is also reported, as is cumulative learning-rate exposure.
- **Recovery offsets.** r ∈ {0, 50, 200, 500, 1111, 1900} after release, i.e.
  - early steps 100, 150, 300, 600, 1211, 2000;
  - late steps 646, 696, 846, 1146, 1757, 2546;
  - plus step 4000 for both.
- **Sham validity** (`scripts/i_sham_validity.py`, confirmation stream, the six baselines at step 1211). A sham is valid on a baseline if applying it:
  - raises KL by ≤ 0.02, and
  - does not lower the own-question attention mass of either route by more than 0.01.

## Frozen verdict rules

These are implemented in `scripts/score_I.py`, which is fail-closed. Preflight P0–P5 all pass (`logs/preflight_I.json`).

**R-S (sham) and R-N (location noise)**, over the 6 pairs at step 1211:
- **CONFIRMED** (route-specific beyond the control): the real block flips ≥ 5/6 competent pairs, the control flips ≤ 1/6, and the paired mean of δ_real − δ_ctrl is ≥ 0.4 with 95% bootstrap lower bound > 0.
- **FALSIFIED:** the control flips ≥ 3/6, or mean δ_ctrl ≥ 0.5 × mean δ_real.
- **INCONCLUSIVE:** otherwise.
- If any sham fails the validity check, R-S becomes **SHAM INVALID** and is never treated as a successful control.

**R-P (persistence).** For each pair, PI = δ(1900) / max δ(r ≤ 500). PI is defined only if that maximum is ≥ 0.1 and the run is competent at step 4000.
- **CONFIRMED:** at least 4 valid pairs; PI_early > PI_late in all but at most one; mean difference ≥ 0.25 with bootstrap lower bound > 0.
- **FALSIFIED:** at least 4 valid pairs, and either the mean difference is ≤ 0.05 or early exceeds late in at most half of them.
- **INCONCLUSIVE:** otherwise.

**Confounds frozen as reported, not controlled:**
- **Learning-rate exposure differs.** The block mass is about 4× larger late, since warm-up ends at step 200. Exposure from release to the end is also larger for the early block.
- **The early "immediate" organisation (step 100) is not yet formed** in either the baseline or the blocked run. PI therefore normalises by the peak shift within 500 steps of release, not by the shift at release.

## Required outputs

- every run in `logs/I_design.json`, ending in `stop_done.json` or `final.json`;
- `logs/unit_metrics_I.jsonl`, containing all 178 measurements exactly once;
- `logs/i_sham_validity.json`;
- `logs/score_I.json`, written only by the scorer.

**Hashes.** `logs/FREEZE_I_HASHES.txt`. The scorer refuses to run if any of these files has changed.
