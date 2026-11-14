# Frozen predictions and confirmation (Investigation XI)

## Part I — predictions (frozen before any confirmation run)

### Freeze record

The prediction text, the scoring code (`scripts/score_conf.py`) and the confirmation queues
(`logs/queue_conf_A.txt`, `logs/queue_conf_B.txt`) are committed together. Their SHA-256 hashes are recorded in the
freeze commit message and in `logs/FREEZE_HASHES.txt`. No confirmation job was started before that commit (see
`logs/COMPUTE_LOG.txt`).

### Confirmation models: pre-existing, held out from XI, but not new

- Models: wP140–wP143, full_c1 and C1, seed 0, from Investigation X's confirmation set.
- Status: these models were trained and evaluated in Investigation X (behaviour, DAS E-fit, selectivity, trajectories).
  **No Investigation XI analysis was run on them before the freeze.** Every XI method, threshold and design choice was
  developed on xP81, yP130–yP134 and the seed-1 runs.
- Limits: they are not newly trained models. Some of their properties are already known from X (F1–F7). The predictions
  below are about quantities never computed on them: factorial interactions, interaction ablations, mapped
  interchange, decoy DAS, generalization environments, and interaction dynamics.
- Label: **"pre-existing held-out / limited confirmation"**. No new training was done (budget).

### Inputs and sites

- Inputs: fresh natural streams, never used in development:
  - factorial / ablation / dynamics: stream 66_000_000;
  - cross-interchange map fit: 63_000_000; evaluation: 63_500_000. These are the same generator streams as in
    development; the models differ.
  - decoy DAS: training 64_000_000, evaluation 64_500_000; selectivity edits use Investigation X's eval_conf10 set;
  - generalization: 65_0/1/2_00_000.
- Site: the Investigation X E-subspace (L2, q7 tokens, k = 6) fitted in X's confirmation (`*_ev62500000_R.npy`).
- Pass rule: each arm must hold in **≥ 3 of the 4 parents**. A prediction is confirmed only if all its arms pass.

### P1 — The conjunction is formed in layer 1, not layer 0 (factorial interaction)

Script: `scripts/factorial.py`. The 2 × 2 within-episode design:
- TRUE: n_h = 4, so P makes group h perfect;
- CTRL: n_h = 3.

Arms:
- (a) full_c1: E-subspace interaction ratio ‖dd_TRUE‖ / ‖dd_CTRL‖ ≥ 2.0.
- (b) full_c1: rel_I at mid0 (after layer-0 attention) ≤ 0.05, and |share of the E-interaction written by layer 0
  (h0.* + mlp0)| ≤ 0.10.
- (c) C1: ratio ≤ 1.75.

Development basis:
- full ratio 2.73–9.81 (8/8 models), mid0 rel_I ≤ 0.02, layer-0 share |·| ≤ 0.01;
- C1 ratio 0.95 / 1.48 / 0.96.

### P2 — Causal necessity of layer 1's conjunction (interaction-removal ablation)

Script: `scripts/factorial_ablate.py`. Each component's xAP output is replaced by its additive completion; downstream is
recomputed.

Arms:
- (a) full_c1: removing layer 1's conjunction (heads + MLP1) leaves ≤ 0.05 of the E-interaction.
- (b) full_c1: after that removal the decision is closer to the "e_h = 0" counterfactual than to Bayes (KL_e0 <
  KL_bayes) on average, and in ≥ 50% of pairs (pref_e0 ≥ 0.5).
- (c) full_c1: removing layer 0's conjunction leaves ≥ 0.95 of the interaction and changes KL(Bayes) by ≤ 0.01.
- (d) C1: removing layer 1's conjunction changes KL(Bayes) by ≤ 0.02. The C1 decision does not depend on it.

Development basis:
- L1 remaining ≤ 0.01 (8/8); pref_e0 0.53–0.74; KL_e0 0.011–0.106 vs KL_bayes 0.39–0.66;
- L0 remaining 1.00, ΔKL 0.000;
- C1 ΔKL ≤ 0.016.

### P3 — Mapped cross-model interchange transfers the source-specific variable; interchange accuracy alone does not

Script: `scripts/cross_interchange.py`. A ridge map is fitted on natural states only (no counterfactuals, no outcomes).
Pairs between independent parents, both directions: wP140↔wP141 and wP142↔wP143.

Source-specific gain = (KL_mean − KL_x) / (KL_mean − KL_within), where "mean" patches B's mean natural state (a
source-free control).

Arms:
- (a) mapped gain ≥ 0.85.
- (b) shuffled-pairing map gain ≤ 0.10, and random-subspace IIA ≤ 0.60.
- (c) *IIA inflation*: the source-free mean patch has IIA on discriminating pairs ≥ base + 0.40.
- (d) C1 → full_c1 within wP140 and wP141 (both required): mapped gain ≤ 0.85. A C1 model's states do not carry the
  full model's eligibility variable in a linearly mappable form.

Development basis:
- full→full gain 0.92–1.00 (7 pairs); shuffled −0.16 to −0.06; random-subspace IIA 0.50;
- mean-patch IIA_disc 0.75–0.95 vs base 0.18–0.20;
- C1→full 0.58 (one pair).

### P4 — DAS false positive, and the diagnostic that exposes it

Script: `scripts/das_decoy.py`. DAS is trained against the counterfactuals of decoy rules.

Arms:
- (a) **Predicted false positive.** In every Bayes-behaving full_c1 model, a subspace trained for the *A1 perfect-only
  rule* reaches IIA ≥ 0.85, although X's F1 showed these models do not follow A1.
- (b) The preservation test separates true from decoy variables. For pairs whose rule value is equal but whose
  episodes differ: E's preserve_KL ≤ 0.03, and each decoy (A1, PERM) has ≥ 3× E's preserve_KL.
- (c) In C1 models, even the E subspace fails preservation: preserve_KL ≥ 0.05.

Development basis:
- A1 decoy IIA 0.926 / 0.923;
- full E preserve_KL 0.008 / 0.008 vs decoys 0.10–0.47;
- C1 E preserve_KL 0.126 (xP81) and 0.142 (yP130).

### P5 — Generalization: the trained relational rule is applied unchanged off-distribution

Script: `scripts/generalize.py`. No retraining; same tokenization, so no new tokenization cost.

Arms:
- (a) **Two colluders (C = 2):** KL(C1-rule || model) ≤ 0.05, and KL(true C = 2 Bayes || model) ≥ 1.0.
- (b) C = 2: accuracy ≤ 0.40 (near chance, against Bayes ≈ 0.75), while the per-group eligibility variable is still
  read out correctly from the E-subspace (accuracy ≥ 0.95). The variable generalizes; the combination rule does not.
- (c) **Partial copying (β = 0.7), S ≥ 1:** KL(C1-rule || model) ≤ 0.2 × KL(true β = 0.7 Bayes || model). The
  model keeps treating copying as exact.
- (d) β = 0.7, S = 0 (off the trained support): no exact-β Bayes posterior on the grid {0.3, …, 0.99} fits within
  KL 0.10. Off-support behaviour is not determined by the trained computation.

Development basis:
- C2: KL_C1B 0.003–0.012, KL_TRUE 1.39–1.52, accuracy 0.326–0.334, e readout 0.958–1.0;
- β = 0.7: C1B ≤ 0.007 vs TRUE ≥ 0.154;
- S = 0 min-KL 0.11–0.48.

### P6 — The representational conjunction precedes the behavioural switch

Script: `scripts/dyn_factorial.py`, over the saved checkpoints (steps 50–4000).

Arms:
- (a) The first checkpoint at which the full-space interaction ratio (TRUE / CTRL, resid2) is ≥ 2.0 comes strictly
  earlier than the first checkpoint whose D2 behaviour is closest to Bayes. The behavioural steps come from
  Investigation X's `traj_alg_conf.json`: 1211 / 1211 / 813 / 546.
- (b) The final ratio is ≥ 2.0.

Development basis: emergence 246 / 546 / 1211 / 246 / 1211 against first-B steps 813 / 1211 / 4000 / 1211 / 1803
(5/5 earlier); final ratios 2.01–5.46.

### What would falsify the account

- P1 or P2 failing: the conjunction is not localized in layer 1, or is not necessary for the Bayes decision.
- P3(a) failing: no shared, linearly mappable variable across parents.
- P4(a) failing: no false positive, so interchange accuracy is more diagnostic than claimed.
- P4(b) failing: preservation is not a reliable discriminator.
- P5 failing: the models do not apply the trained rule off-distribution.
- P6 failing: mechanism and behaviour emerge in the other order, or simultaneously.

## Part II — confirmation results

### How the confirmation was run

- The confirmation was started at 05:32:52Z, after the freeze commit `74c71f5`. The queues are `logs/queue_conf_A.txt`
  and `logs/queue_conf_B.txt`; per-job times and return codes are in `logs/COMPUTE_LOG.txt`, and every job returned 0.
- Before scoring, `sha256sum -c logs/FREEZE_HASHES.txt` passed, so the scoring code, the queues and the analysis
  scripts are byte-identical to the frozen versions.
- Part I of this file is unchanged since the freeze. Part II was appended afterwards, which is why this file's own hash
  in FREEZE_HASHES now differs. The frozen version is recoverable as `git show 74c71f5:<path>`.
- Verdicts were produced by `scripts/score_conf.py` and are stored in `logs/score_conf.json`.

### Verdicts: 6 of 6 frozen predictions confirmed

All six pass the frozen rule (every arm holds in at least 3 of 4 parents). 67 of the 72 individual checks pass. The five
that fail are listed in the table and below.

| prediction | verdict | evidence (wP140 / wP141 / wP142 / wP143) |
|---|---|---|
| **P1** conjunction formed in layer 1 | **confirmed** (4/4, 4/4, 4/4) | E-interaction ratio full 5.27 / 7.02 / 5.54 / 5.35; C1 0.99 / 0.75 / 1.05 / 1.19. mid0 rel_I ≤ 0.014. Layer-0 share ≤ 0.001 |
| **P2** layer-1 conjunction is necessary | **confirmed** (4/4, 4/4, 4/4, 3/4) | Layer-1 removal leaves 0.00 of the interaction. KL_e0 0.029 / 0.033 / 0.042 / 0.039 vs KL_bayes 0.53–0.58. pref_e0 0.67–0.72. Layer-0 removal: Δ = 0.000. C1 ΔKL 0.003 / 0.007 / **0.025 ✗** / 0.020 |
| **P3** mapped transfer is source-specific; IIA alone is not | **confirmed** (4/4, 4/4, 3/4; C1→full 2/2) | Mapped gain 0.97 / 1.00 / 0.89 / 1.02. Shuffled ≤ 0.05. Random-subspace IIA ≤ 0.515. Mean-patch IIA_disc 0.68 / 0.85 / 0.83 / 0.56 vs base 0.19–0.20 (**wP143→wP142 +0.36 ✗**). C1→full gain 0.49 / 0.71 |
| **P4** DAS false positive, exposed by preservation | **confirmed** (3/4, 3/4, 4/4) | A1-decoy IIA in Bayes models 0.922 / 0.896 / **0.827 ✗** / 0.934. E preserve_KL 0.008 / 0.021 / **0.060 ✗** / 0.006, with decoys 0.11–0.21 (≥ 3× except wP142). C1 E preserve_KL 0.134 / 0.112 / 0.113 / 0.165 |
| **P5** trained rule applied unchanged off-distribution | **confirmed** (4/4, 3/4, 4/4, 4/4) | C2: KL_C1rule 0.006 / 0.012 / 0.006 / 0.027 vs KL_true 1.35–1.63. Accuracy 0.326–0.342 (Bayes 0.75). e read-out 0.998 / 0.999 / 0.968 / **0.939 ✗**. β = 0.7, S ≥ 1: C1-rule ≤ 0.008 vs true ≥ 0.155. S = 0: best-β KL ≥ 0.15 |
| **P6** representational conjunction precedes the behavioural switch | **confirmed** (4/4, 4/4) | Emergence 366 / 813 / 366 / 366 vs first-Bayes step 1211 / 1211 / 813 / 546. Final ratio 3.3–4.3 |

### The five failed checks

- **wP142** accounts for three of them (P2-C1, P4a, P4b). It is the parent with the weakest Investigation X E-fit
  (KLcf 0.162).
  - In wP142 the true-E subspace is itself less clean: E preserve_KL 0.060, IIA 0.853.
  - Its decoy-to-E preservation ratio is 2.8× rather than ≥ 3×.
  - It still ranks E best on every diagnostic.
- **wP143→wP142** (P3c) is a transfer *into* wP142, the same weak target.
- **wP143** (P5b): the eligibility read-out under C = 2 is 0.939, against the 0.95 threshold.

### What the confirmation adds beyond the development models

1. **False positives are systematic.** The A1 decoy, a rule these models do not follow behaviourally (Investigation X
   F1), reaches IIA 0.83–0.93. Interchange accuracy therefore does not identify the algorithm. Preservation and
   selectivity do. On selectivity, E's D2c/D2 is 0.21–0.44 against 0.39–0.52 for decoys. This tells E apart in 3 of 4
   parents; wP142 ties.
2. **This explains Investigation X's F4 falsification.**
   - In C1 models every rule, including E, fails preservation (0.11–0.17).
   - The mean-patch control reaches most of the discriminating-pair IIA without any source information.
   - So C1's high discriminating IIA (X: up to 0.878) was a source-independent effect, not evidence of a computed
     variable.
