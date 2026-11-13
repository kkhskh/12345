# Learning dynamics (Workstream C3)

**Checkpoints.** Each native training run saves 12 log-spaced checkpoints (steps 50 → 4000). At each one, the compact
evaluation set (one third of each evaluation family) is scored on KL, accuracy, per-episode R1 error and R2/R3 error. The
final checkpoint is scored on the full set (`scripts/native_train.py`; `checkpoints/native_train/<tag>/curve.json`).

**Sampling caveat (mandate C3).** Twelve log-spaced points cannot distinguish an abrupt transition from a fast
continuous one. "Gradual" below means only that consecutive checkpoints change smoothly. No phase transition is
claimed.

## Development trajectories (complete parent xP81 unless noted; E_R1 / KL)

| step | 50 | 246 | 366 | 546 | 813 | 1211 | 1803 | 2686 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| **full_c1** (native fine-tuning) | −0.23 / 0.551 | −0.53 / 0.206 | **−0.58** / 0.070 | −0.50 / 0.075 | −0.51 / 0.022 | −0.38 / 0.012 | −0.18 / 0.009 | −0.10 / 0.005 | **−0.11 / 0.001** |
| C1 (frozen layers 0–1) | −0.24 / 0.565 | −0.26 / 0.493 | −0.35 / 0.297 | −0.38 / 0.158 | −0.47 / 0.112 | −0.72 / 0.083 | **−0.79** / 0.062 | −0.70 / 0.048 | −0.55 / 0.042 |
| keep3 (frozen 0–2 + fresh block) | −0.25 / 0.569 | −0.35 / 0.492 | −0.40 / 0.454 | −0.54 / 0.381 | −1.52 / 0.215 | **−1.80** / 0.116 | −1.65 / 0.083 | −1.45 / 0.067 | −1.48 / 0.061 |
| C1 on xP80 (incomplete parent) | −0.22 / 0.567 | −0.30 / 0.415 | −0.37 / 0.320 | −0.36 / 0.248 | −0.35 / 0.194 | −0.39 / 0.147 | −0.71 / 0.120 | −0.83 / 0.100 | −0.86 / 0.092 |

At step 50, E_R1 ≈ −0.23 = −dm_B: the model has no R1 response yet.

## Findings (development; NP4 confirms the full_c1 shape on new parents)

1. **Separable first, in the native Transformer too.**
   - Every native run first moves R1 in the *wrong* direction while KL falls, that is, while the model learns the
     count-based part.
   - In full fine-tuning, the most negative R1 (−0.58) coincides with the steepest KL drop (0.206 → 0.070).
   - Only then does R1 correct, gradually over steps 800–2700, while KL keeps falling 0.022 → 0.005.

   This reproduces in a native model the order Investigation VIII found in small readers: separable solution first,
   cross-group conjunction later.
2. **Where the network can change determines whether the second phase happens.**
   - With the pretrained layers frozen, the downstream block keeps deepening the separable solution until mid-training
     (C1 −0.79 at 1803, keep3 −1.80 at 1211). It recovers only slightly by the end of the schedule.
   - With layers 0–1 trainable, the correction happens.

   The causal reconstruction (`CAUSAL_MECHANISM_RECONSTRUCTION.md`) shows what changed. The fine-tuned layers 0–1 write
   cross-group information into the other groups' source tokens at q. The frozen prefix never does this: its patch share
   is 0.08.
3. **Cleaner counts lead to a deeper shortcut.** keep3 sees exact counts and reaches the worst R1 (−1.80). Together with
   the retrospective completeness analysis, this suggests that easily accessible per-group perfectness *accelerates* the
   separable phase. It is development-only evidence (one parent).
4. **When does relational information "become available" vs "become used"?**
   - *Available:* in complete parents, counts and perfectness are fully decodable from the pretrained states *before* T4
     training starts (step 0).
   - *Used correctly:* not until the second phase of full fine-tuning (≈ step 1200–2700 here), and never within 4000
     steps for frozen-prefix models.

   The gap between availability and correct use is therefore a training-dynamics phenomenon, not a representational one.
5. **Native-state readers show the same order.** mlp_q on xP81's final states with Bayes targets has E_R1 −0.155 at
   step 1000, then +0.03 at 8000 and +0.006 at 32000.

## Not established

- Whether C1 recovers with a longer schedule. `xP81 c1 16000` is in the deferred development queue; its status is
  reported in `NATIVE_RELATIONAL_LEARNING.md`.
- Dense sampling around the transition.
- Any mechanism-level checkpoint analysis: the patching was done only at the final checkpoint.
