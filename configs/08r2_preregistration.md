# 08R2 Preregistration — Causal Mathematical State Editing

Written before any 08R2 validation outcomes were inspected.

Official labels remain unchanged until new evidence exists:

- Original 0A: **C / PARTIAL**
- 08R1 execution: **COMPLETE**
- Section 5.3: **PARTIAL**
- `OBSTRUCTION_READY = NO`

This document freezes the 08R2 validation protocol. It does not evaluate the sealed 400-game confirmation split. It does not launch 08R3.

## Scientific question

At `blocks.6.hook_resid_mid`, 08R1 showed that a trained linear player-relative probe decodes board state (exact 0.8775) and that a full donor residual copy changes legal-move probability mass strongly (`ΔP_gain ≈ +0.79`, `ΔP_loss ≈ −0.80`). A simple probe-direction square edit at norm 8 did not outperform a wrong-square control.

08R2 tests whether a **state-constrained iterative edit** of the residual can produce both:

1. a decoded shift toward a simulator-valid target board, and
2. a corresponding shift in the frozen model's legal-move distribution,

without using the target donor activation or the model's target logits inside the primary editor.

## Site and model

- Frozen Nanda synthetic Othello-GPT (`synthetic_model.pth`).
- Site: `blocks.6.hook_resid_mid`, last valid token.
- Decoder: the selected 08c sklearn probe, reconstructed exactly in PyTorch.
- The Transformer and the selected probe are not retrained.

## Data usage

| Split | Allowed use |
|---|---|
| Train | Activation statistics, PCA, reusable map, normalization |
| Validation | Hyperparameters, editor choice, numerical floors, primary sample size |
| Previously examined test | Historical 08R1 baseline reproduction only |
| Sealed 400-game confirmation | Unused until a later approved confirmation command |

Recipient and target states are grouped by full games. Confirmation `game_id`s must never appear in validation pairs.

## Pair categories

Deterministic pairing, same player-to-move, occupancy/stage matching where available, no identical-target primary pairs.

- **A.** Different legal-move sets.
- **B.** Similar valid boards with changed legality (Hamming ≤ 8).
- **C.** Different boards with unchanged legal-move sets.
- **D.** Wrong-target control pairs.

If category B or C is scarce, report the count. Do not fabricate near-identical boards. Do not silently reweight pairing after seeing method performance.

Primary analysis uses A and B. C is a specificity check. D is a control.

## Editors

Only three primary editors:

- **A.** Full-space state-constrained iterative optimization of `δ ∈ R^512`.
- **B.** Same objective, restricted to the probe row space.
- **C.** Train-derived reusable low-rank linear map from state difference to activation delta. No target donor activation at evaluation.

The objective is decoder state loss plus `λ‖δ‖²`, optional manifold penalty, and preserve-unchanged-square penalty. The primary editor may not use target token probabilities, legal-set logit margins, model-logit gradients, or donor activations.

Initial reproducible optimizer setting: 40 SGD steps. Learning rate, step count, `λ`, `μ`, `η`, rank, and norm budget are selected on validation only. Configured budgets include 8, 16, and 32. Report achieved norms, not only configured limits.

## Primary metrics

For original residual `h`, edited residual `h'`, recipient legal set `L_r`, and target legal set `L_t`:

- `ΔP_gain = Σ_{m ∈ L_t \ L_r} [p_e(m) − p_r(m)]`
- `ΔP_loss = Σ_{m ∈ L_r \ L_t} [p_e(m) − p_r(m)]`
- `S_legal = ΔP_gain − ΔP_loss`

Both components are reported separately. A favorable combined score cannot hide a failure of one component.

Also record: target-legal mass, target-illegal mass, mean probability per move (distinctly labeled), argmax legality under the target board, changed-square decoder accuracy, unchanged-square decoder accuracy, whole-board exact match, induced legal-set Jaccard, output KL, edit L2, state loss, manifold cost, hook fire/readback.

## Baselines

Positive control with donor access: natural full donor patch.

Representation / causal baselines:

- PCA-32 donor subspace patch (donor access)
- PCA-32 complement patch (donor access)
- 08R1 simple probe-direction edit (no donor activation)
- Train-derived mean-square delta (no donor activation)
- Same-norm random edit
- Wrong-target state edit
- Wrong-square state edit
- Wrong natural donor (donor access)
- No-op
- Sign / decoder-gradient attribution
- Direct output-logit steering, labeled as a **behavioral baseline that uses model logits**, not as a state editor

Comparisons use matched sample pairs and report effect-versus-norm and effect-versus-KL. A larger perturbation is not a scientific win.

## Success criterion (validation gate)

All three are required for a **promising** validation-stage result (outcome A). Numerical floors below are set from 08R1 historical magnitudes, not from 08R2 confirmation data. Validation may record a calibrated operating point, but may not lower these floors after seeing confirmation results.

### A. Representation

On primary A/B pairs, the selected editor must improve changed-square accuracy versus the unedited residual, keep unchanged-square accuracy from collapsing, and report whole-board exact match.

Preregistered floors:

- mean changed-square accuracy ≥ 0.40
- mean unchanged-square accuracy ≥ 0.80

Whole-board exact match is reported but is not the sole gate.

### B. Causal behavior

On the same pairs:

- mean `ΔP_gain` > 0 and its recipient-game cluster 95% CI excludes 0
- mean `ΔP_loss` < 0 and its recipient-game cluster 95% CI excludes 0
- mean `|ΔP_gain|` and `|ΔP_loss|` each at least **0.05** at the selected budget

The 0.05 floor is an order of magnitude above the 08R1 probe-direction effect (~0.001) and far below the full-donor effect (~0.79). It is a minimum meaningful causal shift, not a requirement to match full patching.

### C. Specificity

Recipient-game cluster paired differences must have 95% CIs excluding 0 for:

- selected editor `S_legal` minus wrong-target `S_legal`
- selected editor `S_legal` minus same-norm random `S_legal`
- selected editor `S_legal` minus simple probe-direction `S_legal`

If representation succeeds and behavior fails, the result is outcome **B (probe-fooling)**. That is a valid scientific finding, not a nuisance.

## Statistics

- Primary resampling unit: recipient game
- `n_boot ≥ 2000`
- Report mean, 95% CI, pair count, recipient-game count, target-game count, bootstrap seed
- Sensitivity: cluster by `recipient|target` game pair
- All validation trials remain in a persistent ledger
- Exploratory comparisons are labeled exploratory

## Probe-fooling control

If decoded target-board accuracy rises sharply while `ΔP_gain` and `ΔP_loss` remain near zero, record outcome B explicitly.

## Reusability

Validation effects are sliced by unseen recipient games, unseen target games, game stage, board Hamming, legal-set size, changed-square count, and player. Editor C tests whether a train-derived map generalizes without the target donor activation.

## Confirmation

The 400-game manifest remains sealed. The confirmation script may be prepared. It must not run during this implementation/validation phase.

## Status labels

Local missing-PyTorch work can only claim `LOCAL_ENGINEERING_VERIFIED — COLAB_REQUIRED`.

Real-model validation may claim `08R2_VALIDATION_COMPLETE` only after genuine frozen-model execution with required controls and saved numerical artifacts.

These 08R2 labels do not change the official 0A letter.
