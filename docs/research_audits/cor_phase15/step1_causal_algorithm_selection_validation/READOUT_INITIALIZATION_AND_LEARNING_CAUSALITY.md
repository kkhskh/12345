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
| | 22/21/22 | 0.314 | 0.913 | hybrid | XIII-D dev |
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

**Post-hoc description (not confirmed; revised after review 2, F1/F2).**

**Description of the cells.** It is tempting to read these cells as "early needs the diagonal's W *and* L, and mixture is the default". The data do not support that reading:

- W_A and the layer-2 weights are drawn from independent keys. A same-seed (L, W) pair therefore has no structural relation that a mixed-seed pair lacks; a single-factor change is simply another random initialisation of the fresh head.
- Among natural diagonals, early is as common as mixture: 7 early, 6 mixture, 4 hybrid (`known_outcomes.json`).
- Off-diagonal cells are early in 3 of 11 cases, against 7 of 17 diagonals (Fisher exact p = 0.69).

**What the data do support:**

- In this population the organisation is a **strongly interaction-dependent function of the parent and the fresh (L, W) draw**.
- No low-order factor model is supported.
- **D does not change the class.**
- In uP160 and uP161, every single-factor change away from an early diagonal produced a non-early outcome (4 of 4).
- In uP163, W22 gives early on L21 and on L22, but W20 placed on L21 does not (borderline at 1211; cell below). So "W decides in uP163" holds only for one particular W.

### P10 (freeze 3b): the requirement tested in a new parent — FALSIFIED

**Hypothesis.** H_req (post hoc, formed after P1): outside uP163, the early organisation needs the early diagonal's W *and* L, and changing either gives a non-early outcome.

This hypothesis was formed after P1 and frozen before training. It was tested in parent uP164:

- early diagonal: 21/21/21;
- non-early diagonal: 20/20/20, a hybrid that ends at 0.52 / 0.87.

The class is read at step 1211, on stream 94.1 M.

| cell | change from the early diagonal | lik_RO1 | mix_mass | KL | class | predicted |
|---|---|---|---|---|---|---|
| uP164 s21 (reference) | — | 0.712 | 0.279 | 0.042 | EARLY | EARLY |
| uP164 s20 (reference) | all three | 0.615 | 0.700 | 0.032 | NOT-EARLY | NOT-EARLY |
| L20/W21/D21 | **L only** (frozen text mislabelled it; see errata) | 0.575 | 0.780 | 0.022 | NOT-EARLY | NOT-EARLY ✓ |
| **L21/W20/D21** | **W only** | 0.781 | 0.162 | 0.030 | **EARLY** | NOT-EARLY ✗ |
| L21/W21/D20 | D only | 0.745 | 0.258 | 0.044 | EARLY | EARLY ✓ |
| uP163 L21/W20/D21 (descriptive) | — | 0.608 | 0.553 | 0.038 | NOT-EARLY (borderline) | — |

**In uP164 the layer-2 initialisation L decides the organisation and the readout map W does not:**

- the W-only change (L21/W20/D21) stays early;
- the L-only change (L20/W21/D21) is not early.

uP164 has no mixture diagonal (s20 is hybrid), so these cells say nothing about a "mixture default".

**Caveats on reading the class at step 1211.**

- F1f was EARLY by this rule at 1211 but hybrid at 4000.
- Arm (b), the deciding arm, is far from the boundary (mix 0.16).
- Arm (c) and the s21 reference (mix 0.26–0.28, KL 0.042–0.044) are close to both the class boundary and the competence bound.
- These cells were not continued to step 4000: `FX_STOP` runs cannot resume, and no optimiser state was saved.

**Description of specific cells.** This is not a "which component decides" law; each parent is represented by 2–5 cells.

| parent | what the tested cells show |
|---|---|
| uP163 | W22 gives early on L21/L22; W21 gives hybrid; W20 on L21 is not early (step 1211) |
| uP164 | a W-only change stays early; an L-only change does not (step 1211) |
| uP160, uP161 | each W-only and each L-only change away from the early diagonal is non-early |

**D does not change the class in any tested pair:** 10 pairs at step 4000, plus one pair in uP164 at step 1211. D can still move the continuous lik_RO1, by up to 0.18 (uP161 23/23/20 vs 23/23/23, both hybrid).

**What survives of XIII-C's positive result:**

- The organisation depends on the initialisation of the freshly initialised layer-2/readout pair (together with the parent), not on the data order.
- There is no general "readout-map" cause.
- No component-level law was identified.

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

**It could not be completed during the confirmation phase.** Runs with `FX_STOP` neither resume nor write final.json, so every OOM kill restarted them from step 0 (COMPUTE_LOG). It was re-run once memory was free.

Results (development / exploratory, not frozen):

| run (uP161) | base class | transform of W_A | lik_RO1 @1211 | mix_mass @1211 | KL | class |
|---|---|---|---|---|---|---|
| s21 (21/21/21) | mixture | none | 0.019 | 0.942 | 0.017 | mixture |
| s22 (22/22/22) | early | none | 0.646 | 0.221 | 0.121 | early |
| 21/21/21 | mixture | scale ×0.6 | 0.007 | 0.951 | 0.011 | mixture |
| 22/22/22 | early | sign flip (neg) | 0.020 | 0.964 | 0.013 | **mixture** |
| 22/22/22 | early | column permutation (perm120) | 0.049 | 0.937 | 0.020 | **mixture** |
| 22/22/22 | early | scale ×1.5 | not completed: OOM-killed five times, and it cannot resume (`FX_STOP`) | | | |

A sign flip or a permutation of the candidate columns of an early-producing W_A is enough to give the mixture organisation, even though both transforms preserve every spectral property of W_A. Rescaling a mixture W_A does not create early.

So in uP161 what matters is how W_A is aligned with the fixed layer 2 and the parent, not its norm or spectrum. This is consistent with the interaction-dependent description above. It is three runs in one parent, a development observation.

Two limits on the comparison:

- The untransformed early reference uP161 s22 is not competent at 1211 (KL 0.121).
- A rot1 run exists without metrics.

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
| F1f | l2q 546–1146 (branched at 546), mixture base | — | — | 0.732 / 0.289 | 0.460 / 0.914 | hybrid (stable; both routes used) |

Unmasked references:

- early base: 0.36 / 0.25 at step 246, 0.70 / 0.18 at step 546;
- mixture base: 0.35 / 0.35 at step 246, 0.14 / 0.89 at step 546.

**Critical periods** (revised after review 2, F3–F5).

- **Sensitivity to route availability at the start.** Blocking either route for only the first 100 steps permanently gave the other organisation, in both directions (n = 1 per direction).
  - P2b had predicted recovery: **FALSIFIED**.
  - The 0–100 window lies in warm-up and holds about 1.2% of the learning-rate mass.
  - This shows *sensitivity* in that window. It does not show that the decision is *made* by step 100: the two unmasked bases are still indistinguishable at steps 111–246.
- **Untested.** No windows shorter than 100 steps were tried, and there is no sham-mask control (a mask on an edge in neither route). The conclusion is therefore "no recovery window of ≥ 100 steps", not "no critical window".
- **Late blocks (546–1146) differ in severity.**
  - ro0 on the early run (F1e) drives the training loss to chance: 0.688 → 1.090 at step 600, against ln 3 = 1.099. The model relearns under the mask and ends mixture.
  - l2q on the mixture run (F1f) is a mild perturbation: 0.599 → 0.765. It ends as a **stable hybrid**, with lik 0.46 and 0.79 blocked-route accuracy, in which both routes are used.
  - So the early/late asymmetry is confounded with intervention severity. A severity-matched comparison was not run.
- **The "mixture attractor" description is withdrawn.** F1c and F1d show that the mixture organisation is not an attractor at the start.

**Fate of the unused route (P2a, P2c).**

| run | early-route info present at RO1 (final) | lik_RO1 | reactivation gain | gain with RO1 mean-replaced |
|---|---|---|---|---|
| F1a | 0.934 | −0.003 | +0.259 | +0.054 |
| F1b | 0.743 | −0.001 | +0.063 | +0.066 |
| F1e | 0.979 | 0.002 | +0.274 | +0.050 |

- **P2c CONFIRMED (no suppression)**, on one diagnostic arm (F1a), using the pooled measure relative to the run's own post-mask peak. Final presence 0.934 vs peak 0.931, while use fell from 0.198 to −0.003.
  - Group-resolved presence is lower in the unused route than in early-organised siblings of the same parent: 0.776 in F1a against 0.991 in F1c/F1d. It falls further with longer blocks: 0.551 for F1b, 0.452 for development ro0T600.
  - So pooled evidence remains decodable while group-resolved evidence is reduced.
- **P2a FALSIFIED.** P2a predicted a commitment signature: information present, unused, reactivatable by retraining only the readout, and dependent on the RO1 representation.
  - **F1a and F1e show the signature partially.**
    - Gains are +0.26 / +0.27. This recovers about half of the gap to the early-organisation accuracy of about 0.86; after-KL is still 0.40–0.44.
    - With RO1 mean-replaced, the gains fall to +0.054 / +0.050.
    - They fail only the absolute cap ("gain with RO1 removed ≤ 0.05"), by 0.004 and 0.0005. That margin is below the binomial SE of one accuracy estimate (≈ 0.011), so passing would have been equally uninformative.
  - **F1b passes the frozen presence threshold** (0.743 ≥ 0.6) but **fails both gain criteria**: gain +0.063 < 0.10, and gain with RO1 removed +0.066 > 0.05.
    - By the alternatives frozen with P2a, this is "present but not usable by retraining the readout alone". It is not loss.

Sensitivity of this verdict to the thresholds is in `logs/uncertainty_13d.json` and FROZEN_CONFIRMATION_AND_STATISTICS.md.

**Competition versus commitment (post-hoc reading, not a confirmed result).**

- **The early-route information is computed and retained (pooled) whether or not it is used.** This rests on P2c and on development; the group-resolved form is reduced.
- **Which route is used is sensitive to route availability at the start of training.**
- **After a short block, a readout-only probe can partially re-read the unused route.** After a 250-step block it cannot.

This is compatible with a commitment of the downstream computation, but it does not isolate where the commitment sits. W_A-only retraining is a linear probe on the layer-2 output. It does not separate W_A from layer-2 attention as the locus of lock-in. P2a, the frozen test of the commitment signature, was falsified.

## B5. Learning-transition predictors (exploratory, not frozen)

Can a mechanistic measure early in training predict the final organisation better than behavioural or optimisation baselines?

Method:

- **Script:** `scripts/b5_transition.py`, output `logs/b5_transition.json`.
- **Runs:** 33 with known final class (12 early), on stream 93.1 M:
  - 13 development factorial runs;
  - the 4 A-C confirmation cells;
  - 16 natural seeds of uP160–uP165.
- **Target:** final early class (lik_RO1 ≥ 0.6).

| predictor at step 246 | Spearman ρ with final lik | LOO R² | AUC (early) |
|---|---|---|---|
| mix_mass | −0.69 | 0.35 | **0.86** |
| route_Q | −0.65 | −0.02 | 0.81 |
| accuracy | −0.57 | 0.10 | 0.71 |
| KL | +0.60 | 0.13 | 0.69 |
| lik_RO1 | +0.35 | 0.22 | 0.67 |
| training loss | +0.31 | −0.03 | 0.64 |

**Step 246.** mix_mass has the highest AUC. Its advantage over the behavioural baselines is not established: over accuracy the bootstrap 95% interval is +0.15 [−0.07, +0.37] (runs resampled, 5000 draws); over KL it is +0.17 [−0.07, +0.41].

**Step 111.** No measure exceeds AUC 0.78.

**Conclusion.** The two organisations become distinguishable in the layer-2 mixture mass between steps 111 and 246. Training loss does not predict the outcome. At this n, a mechanistic measure is not shown to be a better predictor than behaviour.
