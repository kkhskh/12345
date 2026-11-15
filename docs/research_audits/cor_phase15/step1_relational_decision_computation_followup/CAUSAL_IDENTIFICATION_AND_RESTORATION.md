# Causal identification and restoration (Workstream B)

Investigation XIII-B, Phase 1.5 Step 1. **[dev]** = XIII seed-20 models; **[conf]** = frozen seed-21 replicates (FROZEN_CONFIRMATION).

**One head-selection rule everywhere** (`src/relpairs.mixture_heads`). A mixture head has readout-token mass ≥ 0.8 on the six q source
tokens **and** a direct-logit share ≥ 0.2 (XIII's frozen class rule). The mass-only list is reported alongside. This resolves XIII's
mismatch, where uP165 used heads {0,1,3} in one script and {0,1} in another, and uP161 included head 2.

Results under the strict rule:
- uP161: {3} (mass-only {2,3});
- uP165: {0,1,3};
- uP164: {0}. With this script's share estimate uP164 passes (0.2); XIII's estimate was 0.199. Its own stream gives the difference.
  Predictions Q4–Q6 therefore condition on the *continuous* readout signature rather than on this threshold.

## B5. Why XIII's mediator explained only 25–33%

XIII's TRUE factorial compared xP with xAP. Those cells differ **at q** (the edited group's agreement at q), so the measured total effect
mixed a candidate-likelihood change with the relational change. XIII-B uses **purely relational** families instead, in which the values at
q are identical:
- **EA** (xA → xAP): e_h and perfectness both flip;
- **P0** (x0 → xP): perfectness flips while e_h stays 0, which separates E from A1;
- each has a control family in which the edited group stays imperfect.

Components at layer 2 come from the natural donor (`scripts/l2_components.py`):
- R = mixture-head attention rows;
- V = mixture-head value inputs at the q tokens;
- O = the other layer-2 heads' outputs at the readout token;
- M = MLP2 output at the readout token;
- K = the residual skip at the readout token.

All 32 subsets were run, giving Shapley shares of the projection on the total effect [dev]:

| model | R | V | O | M | K | all five |
|---|---|---|---|---|---|---|
| uP161 | **0.68** | 0.03 | 0.17 | 0.12 | 0.00 | 1.00 |
| uP165 | **0.81** | 0.01 | 0.00 | 0.17 | 0.00 | 1.00 |
| uP164 (hybrid) | 0.22 | 0.02 | **0.39** | 0.29 | 0.08 | 1.00 |

With purely relational edits, the attention-row pathway carries **68–81%** of the effect in the two strong-mixture models. The remainder
goes mainly through MLP2 and, in uP161, other heads. In the hybrid uP164, other heads and MLP2 carry most of it.

Pairwise interactions are reported in `logs/l2_components_dev*.json`. The R–M interaction is negative, reflecting serial redundancy, so the
shares are not separable contributions.

## B1. Matched decoys for the attention rows [dev]

Every decoy is transplanted by the same procedure. "Projection" is the share of the total EA effect reproduced by the single-component
transplant; fid_E is the R² against the E-mixture prediction.

| decoy | uP161: projection / mean L1 row change / fid_E | uP165: projection / mean L1 row change / fid_E |
|---|---|---|
| mixture rows | **0.84** / 1.14 / 0.95 | **0.99** / 0.94 / – |
| TOP: contribution-matched (largest-share non-mixture layer-2 head) | 0.22 / 1.23 / 0.51 | 0.00 / 0.42 / – |
| MAG: TOP rescaled to the mixture row-change magnitude | 0.20 / 1.14 / 0.50 | 0.00 / 0.76 / – |
| LOW: XIII's decoy (lowest-mass head) | 0.00 / 0.40 / 0.04 | 0.00 / 0.42 / – |
| PERM: mixture rows with groups permuted (same mass, entropy, positions) | 0.53 / 1.24 / 0.33 | 0.69 / 1.43 / – |
| MEAN: source-free mean row | 0.52 / 1.73 / 0.23 | 0.66 / 1.76 / – |

- The specificity of the mixture rows **survives** magnitude- and contribution-matched decoys.
- PERM and MEAN rows produce **large, non-specific** changes: their projection on the effect is moderate, but their fidelity to the E
  prediction is low (≤ 0.33), and their projection changes sign in the control families.
- So "projection" alone is not a safe specificity metric; fidelity is reported with it.

## B3. Restoration through an independent pathway [dev]

- **Removal.** Run the donor cell with the layer-1 attention output at the six q tokens (route Q) taken from the base cell.
- **Restoration.** Insert the *intact* donor mixture-head rows at layer 2. This pathway is not set by any trained subspace: it uses natural
  activations of a different layer and operation.

| | uP161 | uP165 | uP164 (hybrid) |
|---|---|---|---|
| effect removed by route-Q removal | 0.69 | 0.67 | 0.18 |
| removed only (projection / fid_E) | 0.31 / 0.31 | 0.33 / 0.38 | 0.82 / 0.85 |
| **restored with mixture rows** | **0.92 / 0.97** | **0.98 / 0.98** | 0.95 / 0.94 |
| restored with LOW / TOP / MAG rows | 0.31 / 0.44 / 0.43 | 0.33 / 0.33 / 0.33 | 0.82–0.84 |
| restored with PERM / MEAN rows (fid_E) | 0.60 (0.34) / 0.62 (0.28) | 0.70 (0.37) / 0.67 (0.23) | 0.80 (0.53) / 0.85 (0.62) |
| unrelated likelihood-edit effect preserved (slope) | 0.96 | 1.01 | 1.01 |
| **P0 family, E vs A1: restored-row fid_E / fid_A1** | **0.86 / 0.03** | **0.82 / 0.03** | 0.74 / 0.03 |

- **E vs A1 at the natural level** (P0 family). The model's own response is 0.15–0.16, matching the E prediction (0.14) and not the A1
  prediction (1.07).
- Transplanted mixture rows follow E (fid 0.81–0.83) and not A1 (0.02–0.03).
- This is the **discriminating natural-pathway evidence** that XIII lacked: in XIII the E-vs-A1 contrast was observational, on 93 donors
  of one configuration.

## B2. Balanced E-versus-A1 DAS (`scripts/das_bal.py`) [dev]

Training used 50% discriminating pairs. Budgets were identical for the E, A1 and RANDE (decoy target) subspaces. Evaluation used held-out
discriminating pairs only. Each cell gives fid_E / fid_A1:

| patch | uP161 | uP165 |
|---|---|---|
| E-trained subspace | **0.75 / 0.02** | **0.56 / 0.05** |
| E subspace, source-free (filled with the mean source projection) | 0.01 / 0.93 | 0.00 / 0.82 |
| A1-trained subspace | 0.00 / 0.72 | 0.00 / 0.70 |
| A1 subspace, source-free | 0.01 / 0.91 | 0.01 / 0.87 |
| RANDE-trained (decoy) | 0.00 / 0.65 | 0.00 / 0.65 |
| random 6-dim / no-op | 0.00 / ≤ 0.04 | 0.00 / ≤ 0.03 |

- **E is identified.** The E-trained subspace reproduces E counterfactuals, and only when source information is patched in: source-free
  gives 0.01.
- **A1 is not identified.** The A1-trained subspace reproduces A1 counterfactuals **less well than a source-free patch of any subspace**.
  On discriminating pairs the A1 counterfactual coincides with the generic consequence of overwriting the eligibility region with its
  mean.
- So XIII's failure (P8) was a failure of the **training distribution**, not of E as a mediator. "Success" of an A1 subspace is an
  artefact that a source-free null exposes.
- E/A1 subspace principal cosines are 0.77–0.80, 0.75–0.79, 0.45–0.50, 0.32, ≤ 0.12, so the subspaces still partially overlap.
- Pre-assembled (early-readout) models: see FROZEN_CONFIRMATION, and the development rows added there once the OOM-killed runs were
  repeated.

## B4. Partial and resampling ablations

See ALTERNATIVE_NATIVE_DECISION_ALGORITHMS §5. The behavioural residual is computed, not promised. Full layer-1 attention ablation
leaves ≈ 0 relational sensitivity in both classes, and no hidden alternative route appeared.

## Identification limits that remain

1. Shapley shares depend on the chosen component partition.
2. Removal used a base-cell value, which is a natural activation but a specific counterfactual.
3. The head rule's share estimate is sensitive near the 0.2 threshold.
4. Balanced DAS identifies E only at the single site tested (resid2, the seven q tokens, k = 6).
