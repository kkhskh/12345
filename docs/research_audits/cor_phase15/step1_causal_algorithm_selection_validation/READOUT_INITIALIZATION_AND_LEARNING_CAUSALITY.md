# Readout initialisation and learning causality (Workstreams A and B)

COR Phase 1.5, Step 1, Investigation XIII-D. This is about how small trained transformers came to organise their computation. It does not validate COR and makes no claim about the O/D operators.

## Question

XIII-C's positive result was that the organisation (mixture vs early) is set by the readout-map initialisation through an early, path-dependent race. XIII-D tests that claim directly:

- **A:** the readout-map seed W isolated from the layer-2 seed L and the data order D, in more parents.
- **B:** competition versus commitment, critical periods, and the fate of the unused route.

## Setting

**Fine-tuning setup (`full_c1`).**

- Fresh layer 2: `N.init(L+7777)["layers"][2]`.
- Fresh readout map: `N.init(W+7777)["WA"]`.
- Data stream rng: `default_rng(1000003(D+1)+31782)`.
- Training: 4000 steps.
- Script: `scripts/ft_factorial.py`.

**Classes** (lik_RO1 = early readout use; mix_mass = layer-2 head attention on the q tokens):

- mixture: mix ≥ 0.8 and lik ≤ 0.15;
- early: lik ≥ 0.6 and mix < 0.5;
- hybrid: otherwise.

**Reproducibility** (`logs/determinism_check.json`). Three independent full-length 4000-step replays are **bitwise identical** to their originals at every common checkpoint (24, 12 and 12 checkpoints). The loss history, final accuracy and KL, and lik_RO1 / mix_mass / route_Q are identical too. The three replays are:

- uP163 L21/W22/D21 versus XIII-C;
- uP160 23/23/23 versus the XIII-B seed-23 run;
- uP160 21/21/21 versus the XIII-B seed-21 run.

Two consequences:

- Training is deterministic on this software/hardware.
- The diagonal fine-tune with L = W = D = s *is* the natural seed-s model.

## A. Isolated readout-map effects

### Factorial outcomes (final, step 4000)

| parent | L/W/D | lik_RO1 | mix_mass | class | source |
|---|---|---|---|---|---|
| uP160 | 21/21/21 | 0.806 | 0.169 | early | XIII-B |
| | 21/21/23 (D only) | 0.806 | 0.167 | early | XIII-C |
| | **21/23/21 (W only)** | 0.062 | 0.945 | mixture | XIII-D A-C3 |
| | **23/21/23 (W only)** | 0.157 | 0.953 | hybrid (mixture-like) | XIII-D A-C4 |
| | 23/23/21 (D only) | 0.025 | 0.968 | mixture | XIII-C |
| | 23/23/23 | 0.034 | 0.973 | mixture | XIII-B |
| uP161 | 21/21/21 | 0.006 | 0.966 | mixture | XIII-B |
| | 21/21/22 (D only) | 0.007 | 0.965 | mixture | XIII-C |
| | **21/22/21 (W only)** | 0.006 | 0.965 | mixture | XIII-D A-C1 |
| | **22/21/22 (W only)** | 0.007 | 0.975 | mixture | XIII-D A-C2 |
| | 22/22/21 (D only) | 0.779 | 0.215 | early | XIII-C |
| | 22/22/22 | 0.783 | 0.199 | early | XIII-B |
| uP163 | 21/21/21 | 0.318 | 0.907 | hybrid | XIII-B |
| | 21/21/22 | 0.266 | 0.945 | hybrid | XIII-C |
| | 21/22/21 | 0.818 | 0.168 | early | XIII-C (and XIII-D replay) |
| | 21/22/22 | 0.816 | 0.161 | early | XIII-C |
| | 22/21/21 | 0.451 | 0.843 | hybrid | XIII-C |
| | 22/22/21 | 0.823 | 0.202 | early | XIII-C |
| | 22/22/22 | 0.801 | 0.247 | early | XIII-B |

### P1 (freeze 2): FALSIFIED

P1 predicted that a W-only change moves the outcome to the side of the diagonal that shares the new W, in two parents and in both directions.

All four confirmation cells were competent (KL ≤ 0.0017), and **all four ended in the mixture organisation**.

- **Early→mixture direction (diagonal with the new W is mixture): confirmed in both parents.**
  - A-C2: d_W = 0.009.
  - A-C3: d_W = 0.039.
- **Mixture→early direction (diagonal with the new W is early): failed in both parents.**
  - A-C1 stayed mixture: d_W = 1.09, d_other = 0.002.
  - A-C4 stayed mixture/hybrid: d_W = 1.02, d_other = 0.13.

**Revised description.** This is post hoc and goes beyond what P1 tested; it was then tested in freeze 3b (below). The three parents behave as follows:

- In uP160 and uP161, the readout map is **necessary but not sufficient** for the early organisation. Early arises only when the early diagonal's W and its layer-2 initialisation L occur together. D never matters. Changing either W or L gives the mixture organisation.
- In uP163, W is **sufficient in both directions**:
  - W22 gives early with L21 or L22;
  - W21 gives hybrid with either L.

So XIII-C's statement that "the organisation is set by the readout-map initialisation" holds only for uP163. In general:

- the mixture organisation is the default outcome;
- the early organisation is a conjunctive and fragile outcome, which needs a particular readout map, usually together with a particular layer-2 initialisation.

The B results below show the same asymmetry in time.

### P10 (freeze 3b): the requirement tested in a new parent

See the section "Freeze 3b result" below. It is filled in from `logs/score_f3b.json`.

### Step-0 readout-map features (development, n = 19 initialisations)

`scripts/w_features.py` computes the following at step 0:

- the candidate-evidence decodability S at the readout token;
- gradient norms of each block;
- the singular values and norm of W_A.

**What the features show:**

- At step 0 no parent has candidate evidence at the readout token (R² ≈ 0).
- The largest correlations with final lik_RO1:
  - S_RO0: +0.50, on values all within ±0.01, so likely noise;
  - ||W_A||: −0.51;
  - σ1(W_A): −0.49.
- ||W_A|| depends on the W seed only.
- The same W seed gives different outcomes in different parents.

So no parent-independent property of W_A predicts the outcome. This agrees with the L×W and parent×W dependence above.

### Controlled transforms of W_A (development)

The screen applied fixed transforms to the fresh readout map, holding L and D fixed, and read out the organisation at step 1211:

- perm (column permutation);
- neg (sign flip);
- scale 0.6 / 1.5;
- rot (input-side rotation).

**It could not be completed during the confirmation phase.** Runs with `FX_STOP` neither resume nor write final.json, so every OOM kill restarted them from step 0 (COMPUTE_LOG). It was re-run once memory was free; results are in "Transform screen" below, labelled development / exploratory.

## B. Competition versus commitment, critical periods, the unused route

### Development (B3, uP161 s21, mixture)

- The early route's information stays decodable at the readout token after layers 0–1 while its use falls:
  - presence (pooled R²): 0.60 → 0.83 → 0.79 → 0.85 at steps 111 / 246 / 546 / 4000;
  - lik_RO1: 0.055 → 0.135 → 0.049 → 0.005.
- **Reactivation:** retraining only W_A + final LN for 300 steps, with the mixture route blocked (l2q mask), raises accuracy 0.34 → 0.566. With the layer-1 readout representation mean-replaced, it only reaches 0.337.
- The early model is unaffected by the block (0.863).

### Confirmation cohort (freeze 1, commit `8a116e4`)

The masks:

- **ro0:** layer-0 readout→source attention masked, which blocks the early route.
- **l2q:** layer-2 readout→own-q-source attention masked, which blocks the mixture route.

Bases: uP160 21/21/21 (early) and 23/23/23 (mixture). Each run is identical to its base apart from the mask window; the base replays are bitwise identical to the originals.

| run | mask | lik / mix @246 | @546 | @1211 | final (4000) | class |
|---|---|---|---|---|---|---|
| F1a | ro0 steps 0–100, early base | 0.200 / 0.583 | 0.024 / 0.934 | 0.003 / 0.964 | −0.003 / 0.968 | **mixture** |
| F1b | ro0 0–250, early base | −0.002 / 0.664 | −0.003 / 0.942 | −0.002 / 0.964 | −0.001 / 0.972 | mixture |
| F1c | l2q 0–100, mixture base | 0.468 / 0.154 | 0.698 / 0.154 | 0.771 / 0.158 | 0.850 / 0.158 | **early** |
| F1d | l2q 0–250, mixture base | 0.456 / 0.151 | 0.677 / 0.157 | 0.784 / 0.166 | 0.856 / 0.164 | early |
| F1e | ro0 546–1146 (branched at 546), early base | — | — | 0.019 / 0.966 | 0.001 / 0.976 | **mixture** |
| F1f | l2q 546–1146 (branched at 546), mixture base | — | — | 0.732 / 0.289 | 0.460 / 0.914 | hybrid, returning to mixture |

Unmasked references:

- early base: 0.36 / 0.25 at step 246, 0.70 / 0.18 at step 546;
- mixture base: 0.35 / 0.35 at step 246, 0.14 / 0.89 at step 546.

**Critical periods.**

- **There is no release-time window in which the organisation recovers.** Blocking either route for only the first 100 steps permanently gives the other organisation, in both directions (P2b predicted the 100-step masks would recover: **FALSIFIED**). The early decision is made within the first 100 of 4000 steps.
- **The asymmetry from A appears again in time.**
  - Early organisation, blocked for 600 steps *after* it has formed (F1e, from step 546): it switches permanently to mixture.
  - Mixture organisation, given the same late block (F1f): it switches to early only while the block lasts. It then returns towards mixture (mix 0.91 at step 4000).
- **So the mixture organisation is an attractor at every stage tested, and the early organisation is not committed even at step 546.**

**Fate of the unused route (P2a, P2c).**

| run | early-route info present at RO1 (final) | lik_RO1 | reactivation gain | gain with RO1 mean-replaced |
|---|---|---|---|---|
| F1a | 0.934 | −0.003 | +0.259 | +0.054 |
| F1b | 0.743 | −0.001 | +0.063 | +0.066 |
| F1e | 0.979 | 0.002 | +0.274 | +0.050 |

- **P2c CONFIRMED (no suppression).** In F1a the early route's information at the readout token is as present at the end as at its peak (0.934 vs 0.931), while its use fell from 0.198 to −0.003. So the route was not suppressed. It is present and unread.
- **P2a FALSIFIED.** P2a predicted a commitment signature: information present, unused, reactivatable by retraining only the readout, and dependent on the RO1 representation.
  - F1a and F1e show that signature strongly. Gains are +0.26 / +0.27, and with RO1 mean-replaced they fall to +0.054 / +0.050, i.e. a fifth.
  - **They fail only the pre-declared absolute cap** "gain with RO1 removed ≤ 0.05", by 0.004 and 0.0005.
  - **F1b fails substantively.** After a 250-step block the information is less present (0.74) and is not reactivatable (+0.063). The longer early block degrades the route, not just its use.

Sensitivity of this verdict to the thresholds is in `logs/uncertainty_13d.json` and FROZEN_CONFIRMATION_AND_STATISTICS.md.

**Competition versus commitment: conclusion.** The early-route information is computed and kept whether or not it is used. The downstream readout's choice decides which route is used. That choice is made within the first ~100 steps.

The choice is a commitment *only* for the mixture organisation:

- once the readout depends on the mixture route, the early route stays present but unread, and is recoverable only by retraining the readout;
- the early organisation, by contrast, can be dislodged at any time tested (F1e).

So the picture is not a symmetric competition that settles early. It is an asymmetric one:

- the mixture route wins by default whenever the early route is disturbed;
- the early route wins only under specific initial conditions (A) and an undisturbed start (B).
