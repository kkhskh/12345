# XIII-C frozen predictions (rendered from scripts/predictions_13c.json)

Population: logs/CONFIRMATION_POPULATION_13C.md (declared and pushed in 9ec86d6 before any confirmation run). Scorer: scripts/score_13c.py; preflight: scripts/preflight_13c.py (all checks OK); thresholds justified in logs/NOTES_DEV.md. Confirmation units are fine-tuning runs of 3 pretrained parents (uP160, uP161, uP163), not independent parents.

## R1. Initialisation, not data order, sets the organisation; within the initialisation, the fresh layer-2 init (not the readout init) carries it

(a) For every crossed cell (L = W from one XIII-B seed, data from another) whose two diagonal references differ by >= 0.3 in v = (lik_RO1, mix_mass), the cell's final v is closer to its INITIALISATION diagonal than to its DATA diagonal, and within 0.25 of it. (b) For split cells (L2 init from one seed, WA init and data from the other), v is closer to the diagonal of the L2-init seed.

*Alternatives distinguished:* H-A2 data order (cell near the data diagonal); H-A3/H-A5 interaction or sensitive dependence (cell near neither); WA-init carries it (split cell near the other diagonal)

*Thresholds:* `{"min_sep": 0.3, "max_dI": 0.25, "cells": [["uP161:fx:20:20:23", "uP161:s:20", "uP161:s:23"], ["uP161:fx:23:23:20", "uP161:s:23", "uP161:s:20"], ["uP163:fx:21:21:22", "uP163:s:21", "uP163:s:22"], ["uP163:fx:22:22:21", "uP163:s:22", "uP163:s:21"]], "split_cells": [["uP163:fx:22:21:21", "uP163:s:22", "uP163:s:21"], ["uP163:fx:21:22:22", "uP163:s:21", "uP163:s:22"]]}`

## R2. Learning-pathway competition: blocking one route early changes which organisation is reached, and an early internal event predicts the outcome

(a) Masking layer-0 readout->source attention for the first 600 steps (ro0) turns an early-readout-destined run into a mixture model: final mix_mass >= 0.8 and lik_RO1 <= 0.2. (b) Masking layer-2 readout->own-q-source attention for the first 600 steps (l2q) in a mixture-destined run leaves substantial early readout: final lik_RO1 >= 0.4. (c) In unintervened confirmation runs, the mixture mass at step 546 predicts the final readout organisation measured by the intervention-based lik_RO1: mix546 >= 0.6 -> final lik_RO1 <= 0.2; mix546 <= 0.35 -> final lik_RO1 >= 0.6 (0.35 < mix546 < 0.6: no prediction, non-diagnostic).

*Alternatives distinguished:* organisation fixed by the initial weights irrespective of the learning race (masking is undone after release); early readout and mixture heads independent (no suppression); final class unpredictable from step-546 internals

*Thresholds:* `{"ro0_units": [["uP160:fx:21:21:21:ro0T600", "uP160:s:21"]], "ro0_min_mix": 0.8, "ro0_max_lik": 0.2, "l2q_units": [["uP161:fx:21:21:21:l2qT600", "uP161:s:21"], ["uP160:fx:23:23:23:l2qT600", "uP160:s:23"]], "l2q_min_lik": 0.4, "predictor_units": ["uP161:fx:20:20:23", "uP161:fx:23:23:20", "uP163:fx:21:21:22", "uP163:fx:22:22:21", "uP163:fx:22:21:21", "uP163:fx:21:22:22"], "hi_mix546": 0.6, "hi_max_lik": 0.2, "lo_mix546": 0.35, "lo_min_lik": 0.6}`

## R3. Changing the routing organisation by a training intervention changes the fitted decision law (rho_KL) predictably

(a) For each intervention run whose organisation changed relative to its unintervened reference (|Δ lik_RO1| >= 0.3), the KL-fitted power-mean exponent shifts by >= 0.08 in the predicted direction: up when early readout is removed (ro0), down when it is retained/introduced (l2q). (b) Across all confirmation units and references, corr(lik_RO1, rho_KL) <= -0.7.

*Alternatives distinguished:* rho set by parent or initialisation independently of the routing (no shift under intervention); rho unrelated to organisation

*Thresholds:* `{"pairs": [["uP160:fx:21:21:21:ro0T600", "uP160:s:21", 1], ["uP161:fx:21:21:21:l2qT600", "uP161:s:21", -1], ["uP160:fx:23:23:23:l2qT600", "uP160:s:23", -1]], "min_shift": 0.08, "min_org_change": 0.3, "max_corr": -0.7, "corr_units": ["uP161:fx:20:20:23", "uP161:fx:23:23:20", "uP163:fx:21:21:22", "uP163:fx:22:22:21", "uP163:fx:22:21:21", "uP163:fx:21:22:22", "uP160:fx:21:21:21:ro0T600", "uP161:fx:21:21:21:l2qT600", "uP160:fx:23:23:23:l2qT600", "uP160:s:21", "uP161:s:21", "uP160:s:23"]}`

## R4. Why an early-layer repair succeeds where a final-layer repair fails: neither capacity nor information presence, but whether the joint relation reaches layer 2 in PER-GROUP format

Two-colluder adaptation at beta 0.6 (1000 steps, identical data, lr 1e-3; J-pair fidelity slope on m-changing relabellings). In the late-readout confirmation unit: (i) layer-2 + readout repair fails (slope < 0.2); (ii) adding a second trainable block after layer 2 (L2plus, trainable parameters matched to the two-block early repair) still fails (< 0.2); (iii) one trainable block at layer 1 (L1only, matched to L2) fails (< 0.2); (iv) one trainable block at layer 0 (L0only, matched to L2) succeeds (>= 0.6); (v) supplying the exact sufficient statistic m group-agnostically at the layer-2 input (supL2) does not rescue the layer-2 repair (< 0.2); (vi) supplying the same information in per-group format at each group's q tokens (supG) does (>= 0.6). In the early-readout confirmation unit: the layer-2 repair uses the relation (>= 0.4) and the early01 repair does too (>= 0.6).

*Alternatives distinguished:* capacity (L2plus would succeed); optimisation/learning-rate (supG would fail too); information absence at the layer-2 input (supL2 would succeed); 'early layers' per se (L1only would succeed)

*Thresholds:* `{"late_unit": "uP161:fx:20:20:23", "early_unit": "uP163:fx:22:22:21", "late_arms": [["L2", "<", 0.2], ["L2plus", "<", 0.2], ["L1only", "<", 0.2], ["L0only", ">=", 0.6], ["supL2", "<", 0.2], ["supG", ">=", 0.6]], "early_arms": [["L2", ">=", 0.4], ["early01", ">=", 0.6]]}`

## R5. The early-layer two-colluder repair constructs a low-dimensional causal carrier that is NOT the linearly decodable joint statistic, and the frozen layer 2 uses it through its mixture-head attention rows

In the early01-adapted late-readout confirmation unit (beta 0.6): (a) a learned 3-dim subspace at the q-source + readout tokens of the layer-2 input carries >= 0.6 of the exact J-pair change (held-out fidelity slope) and an 8-dim one >= 0.7; (b) interchange along the linear-probe spans of the sufficient statistic m (any of RO/QS/OS, 3 dims) and of the per-group responsibility (QS, 3 dims) carries <= 0.1, although these quantities are decodable; (c) in the frozen layer 2 the readout attention rows carry >= 0.6 (Shapley) of the J effect and, for every mixture head, the change of per-group attention mass correlates >= 0.7 with the change of the exact posterior probability that the group is the reliable one.

*Alternatives distinguished:* explicit linear m code used by layer 2 (probe interchange large); repair bypasses the frozen mixture (rows small, MLP large); no low-dim carrier (DAS small)

*Thresholds:* `{"unit": "uP161:fx:20:20:23", "min_das3": 0.6, "min_das8": 0.7, "max_probe": 0.1, "min_rows": 0.6, "min_corr": 0.7}`

## R6. Attention-row mediator specificity survives rebuilt controls

In every pure mixture confirmation unit (final lik_RO1 <= 0.15, >= 1 mixture head), replacing the mixture heads' readout rows by the valid relational donor's rows explains >= 0.8 of the variance of the Bayes change (R2), the displacement-matched decoy (SHUFD) <= 0.05, the genuine no-op changes no logit (max |Δlogit| = 0), off-distribution/unrelated natural rows (SHUF, SHUFB, PMEAN) R2 <= 0.1, and R2(TRUE) - R2(OTHER heads) >= 0.4.

*Alternatives distinguished:* rows act non-specifically (any comparable perturbation reproduces the Bayes-shaped change); specificity only relative to small-magnitude decoys

*Thresholds:* `{"units": ["uP161:fx:20:20:23", "uP161:fx:23:23:20", "uP163:fx:21:21:22", "uP163:fx:22:22:21", "uP163:fx:22:21:21", "uP163:fx:21:22:22"], "min_true_R2": 0.8, "max_decoy_R2": 0.05, "max_offdist_R2": 0.1, "min_gap_other": 0.4}`

## R7. A low-dimensional eligibility mediator specific to E (against A1 and A2) is identifiable in early-readout units at the covering site

In each early-readout confirmation unit (final lik_RO1 >= 0.6, mix < 0.5), an 8-dim DAS subspace shared across all 43 tokens of the layer-2 input (ALL2), trained only to reproduce the model's own counterfactual outputs on a balanced mix of the four discriminating families, (a) transfers on held-out pairs a mean own-share exceeding the source-free patch of the same subspace by >= 0.2; (b) its patched responses fit E better than A2 by >= 0.15 R2 on E5, E3 and QF4; (c) fit E better than A1 by >= 0.15 R2 on QF5; (d) random 8-dim subspaces transfer <= 0.25.

*Alternatives distinguished:* XIII-B site-limited negative is real (no low-dim mediator at any site); the subspace is source-free/base-rate driven (excess small); it mediates an A2-like (threshold-4) or A1-like variable

*Thresholds:* `{"units": ["uP163:fx:22:22:21", "uP163:fx:22:21:21"], "site": "ALL2", "k": 8, "min_excess": 0.2, "min_margin": 0.15, "max_rand": 0.25}`

## R8. Route structure: two parallel additive routes (Q, R) in mixture units; a direct other-token route (O) in early-readout units; no redundancy

With the donor injected only at the edited group's layer-0 output: in pure mixture units sufficiency(Q) >= 0.25 and sufficiency(R) >= 0.25 with |redundancy| <= 0.1; in early-readout units (lik_RO1 >= 0.6, mix < 0.5) sufficiency(O) >= 0.3 with |redundancy| <= 0.1.

*Alternatives distinguished:* a single serial route (one sufficiency ~1); redundant parallel routes (redundancy > 0.1: either route alone restores most of the effect)

*Thresholds:* `{"units": ["uP161:fx:20:20:23", "uP161:fx:23:23:20", "uP163:fx:21:21:22", "uP163:fx:22:22:21", "uP163:fx:22:21:21", "uP163:fx:21:22:22"], "min_suff": 0.25, "max_abs_red": 0.1, "min_suff_O_early": 0.3}`
