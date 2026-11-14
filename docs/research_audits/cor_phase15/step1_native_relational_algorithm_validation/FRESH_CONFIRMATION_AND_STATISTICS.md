# Fresh confirmation and statistics (Investigation XII)

## Part I — frozen protocol and predictions

### Freeze record

- **Stage-0 protocol** (cohort, seeds, readiness rule, fine-tuning protocol, replicate rule, evaluation set): frozen in
  commit `a7f49aa` (`STAGE0_PROTOCOL_FREEZE.md`, `logs/STAGE0_HASHES.txt`). That commit precedes all fresh-cohort jobs.
- **Predictions**: Part I of this file, the scorer `scripts/score_xii.py`, the statistics helpers `scripts/stats_xii.py`,
  the queue generator `scripts/make_conf_queues.py`, the generated queues, and every analysis script they call.
  - All are committed **before any T4 fine-tuning of the fresh cohort starts**.
  - SHA-256 hashes are recorded in `logs/FREEZE_XII_HASHES.txt`.
  - No fresh-cohort fine-tune or analysis appears in `logs/COMPUTE_LOG.txt` before the freeze commit.
- **What was read about the fresh cohort before the freeze**: pre.json only, the pretraining-completeness covariate
  allowed by Stage 0. P150 aux_acc 1.000 (complete); P151 aux_acc 0.762 (**incomplete**, reported separately).
  P152–P154 were still pretraining.
- **Development models**: xP81, yP130–134 (seeds 0/1) and wP140–143. Every threshold below was set on these.
  Investigation X/XI models are never counted as confirmation.

### Units, pass rule and statistics

- **Unit of analysis.**
  - P1–P4 and P6–P8: the complete fresh parent (seed-10 full_c1 fine-tune; C1 counterpart where stated).
  - P5: model pairs.
  - Episodes are **not** independent replicates of the parent population.
- **Pass rule.** An arm passes iff it holds in ≥ ⌈0.75 · n⌉ of its units. A prediction is confirmed iff every arm
  passes. This is the same rule form as XI, now with an exact per-check listing in `logs/score_xii.json`.
- **Incomplete parents** (aux_acc < 0.99) are scored with the same code and reported separately. They are never part
  of a pass rule.
- **Uncertainty.**
  - Episode-level 95% percentile bootstrap (B = 2000) for every principal quantity: ratios resampled per design;
    paired differences resampled jointly.
  - Parent-level: per-parent values, mean, range, and a bootstrap-over-parents interval.
  - With n ≈ 3–5 parents, the parent-level interval is descriptive, not a population inference.
- **Analysis streams** are all new: factorial/ablation/circuit/restoration 70_000_000, DAS evaluation 70_500_000,
  identification pairs 70_800_000, VS test 70_100_000 / 70_600_000, C = 2 adaptation 71_100_000, map fitting
  63_000_000 with evaluation 63_500_000 (cross-model), and evaluation set `eval_conf12` (stream 21_200_000).
- **E-subspace for fresh models**: Investigation X's `das.py`, unchanged (E variable, L2, q7, k = 6), fitted on the
  fresh models.

### Competing hypotheses (mandate §3) and what each prediction discriminates

| | H1 shared mediator | H2 shared abstraction, different readout | H3 intervention-dependent | H4 redundant / compensating | H5 restricted generalization |
|---|---|---|---|---|---|
| P1 layer-1 locus | predicts | predicts | — | — | — |
| P2 multi-design necessity | predicts | predicts | **falsified if only addcomp works** | partial | — |
| P3 weight-level routing | — | — | — | **predicts sub-head redundancy, amplification not compensation** | — |
| P4 restoration | predicts | — | tested with random control | bounded | — |
| P5 same vs cross parent | predicts equal transfer | predicts transfer may fail | — | — | — |
| P6 identification | E mediates in repaired, A1-like in C1 | — | tests false positives | — | — |
| P7 learning | — | — | — | — | — |
| P8 C = 2 | — | — | — | — | **predicts rule failure + successful readout-only repair that bypasses E** |

### P1 — Fresh replication of the layer-1 conjunction (`factorial.py`)

**Hypothesis.** The repaired (full_c1) models form the perfectness × agreement-at-q conjunction after layer-0
attention, and C1 models do not form it.

**Alternatives.** The conjunction is formed in layer 0; or it is a DAS-subspace artifact; or C1 also forms it.

Arms, for complete parents:
- (a) E-subspace interaction ratio TRUE/CTRL ≥ 2.0, with episode-bootstrap 95% CI lower bound ≥ 1.5.
- (b) rel. interaction after layer-0 attention ≤ 0.05, and |layer-0 share| ≤ 0.10.
- (c) **Subspace-free** full-q7-space ratio ≥ 1.75. This guards against the DAS subspace.
- (d) C1 E-ratio ≤ 1.75.

**Failure.** Any arm fails the pass rule.

**Development basis.**
- E-ratio 2.7–9.8 (13 models); mid0 ≤ 0.02; L0 share ≤ 0.01;
- full-space ratio 2.0–5.5;
- C1 0.75–1.48.

### P2 — Necessity survives alternative intervention designs (`alt_ablation.py`)

**Hypothesis (against H3).** Removing layer 1's contribution moves the decision to the "group h ineligible"
counterfactual cf_e0 under *natural-activation* designs, not only under XI's hybrid additive completion.

Arms, for complete parents:
- (a) additive completion: KL(cf_e0) < KL(Bayes), and pref_e0 ≥ 0.5.
- (b) **path_inA**: layer 1 receives the natural xA residual (h agrees at q but is not perfect), and the bypass keeps
  xAP. Require KL(cf_e0) < KL(Bayes), pref_e0 ≥ 0.5, and the episode-bootstrap CI of mean(KL_e0 − KL_B) entirely
  < 0.
- (c) path_inP (natural xP input): KL(cf_e0) < KL(Bayes).
- (d) Causal scrubbing of the E coordinates. With a *natural donor episode with the same e vector*: KL(Bayes) ≤ 0.05.
  With a donor where e_h = 0: KL(cf_e0) < KL(Bayes).
- (e) Non-specific control. Mean-ablation of layer 1 does **not** produce cf_e0 (KL(cf_e0) ≥ 0.3).

**Development basis (11 models).**
- addcomp / path_inA / path_inP: cf_e0 side in 11/11.
- scrub_match KL_B ≤ 0.03.
- scrub_e0 cf_e0 side in 10/11 (wP142 0.23 vs 0.19).
- mean_L1 KL_e0 0.53–0.85.

### P3 — Weight-level routing: keys/values, redundant sub-routes, MLP1 amplification, smooth nonlinearity (`alt_ablation.py`, `circuit_l1.py`)

**Hypothesis.**
- Layer-1 attention forms a smooth interaction from information read at the *keys/values* of group h's tokens. The
  queries play no role.
- The perfectness signal can reach the conjunction either through attention-pattern changes or through value changes.
- MLP1 amplifies the attention-formed interaction; it does not compensate for it.
- The MLP1 interaction is smooth GELU curvature, not a population of threshold-AND units. LayerNorm is not the source.

Arms, for complete parents:
- (a) Feeding layer 1 the xA input at the **queries only** leaves the decision at Bayes (KL(Bayes) ≤ 0.03). At
  **keys/values only**, it reproduces the full path effect (|ΔKL_e0| ≤ 0.02).
- (b) **Sub-head redundancy.** Removing P's effect on attention patterns alone (pattern_A) keeps the decision on the
  Bayes side (KL_B < KL_e0). So does removing it from layer-1 values alone (values_A).
- (c) **Amplification.** KL(Bayes) after removing the attention interaction with MLP1 recomputed is ≥ 1.5 × the same
  with MLP1 frozen (total ≥ 1.5 × direct, i.e. no compensation).
- (d) **Smoothness.** Threshold-AND neurons (pre-activation crosses 0 only in xAP) carry < 50% of the MLP1
  interaction, and freezing LN2 statistics changes the MLP1 E-interaction by a factor in [0.7, 1.3].

**Development basis.**
- (a) 11/11.
- (b) 10/11 (wP143 pattern-route dominant).
- (c) 11/11 at 1.5×.
- (d) AND share 0.13–0.34; LN2 factor 0.76–1.10.

### P4 — Restoration through the 6-dim E-subspace (`restore.py`)

Arms, for complete parents:
- (a) After layer-1 additive-completion removal, restoring only the E coordinates gives KL(Bayes) ≤ 0.25 and ≤ 0.5 ×
  the ablated KL.
- (b) Restoring a random 6-dim subspace recovers ≤ 0.10 of the gap.

**Development basis.** Restored 0.012–0.157 from 0.39–0.66; random ≤ 0.05.

### P5 — Same-parent versus cross-parent transfer (`cross_interchange.py`, `perm_compare.py`)

**Hypothesis.** Shared pretraining makes the *componentwise* organization more similar, but does **not** make the
causal abstraction more transferable: the variable transfers equally across independent parents.

**Alternative.** Same-parent maps transfer better (H1 weaker), or cross-parent maps fail (H2).

Maps are fitted on natural states (stream 63_000_000, first 6000 states). R² is reported in-sample, on the held-out
tail, and on the separate evaluation stream. Counterfactual transfer is evaluated on stream 63_500_000.

Source-specific gain = (KL_mean − KL_mapped) / (KL_mean − KL_within).

Arms:
- (a) Gain ≥ 0.85 for same-parent pairs (seed 10 ↔ seed 11 of the two replicate parents, both directions) **and**
  for cross-parent pairs (all ordered pairs of complete seed-10 models).
- (b) |mean same − mean cross| ≤ 0.10.
- (c) Mean head/MLP1 E-interaction profile distance, same-parent < cross-parent (raw).
- (d) C1 → full gain ≤ 0.85 (complete parents).
- (e) Source-free mean-patch IIA on discriminating pairs ≥ base + 0.30 (IIA inflation), over all full→full pairs.

**Development basis.**
- Same-parent 0.99–1.00, cross-parent 0.89–1.02;
- profile distance 0.40 (same-parent) vs 0.62 (cross-parent);
- C1→full 0.49–0.71;
- mean-patch inflation +0.36 to +0.77.

### P6 — Identification: discriminating pairs and the linear-versus-projection prediction (`disc_ident.py`, `vs_linear.py`)

**Background.** On XI's decoy pair distribution, the E and A1 counterfactual decisions differ on **0.35%** of pairs
(`math_xii.py` M4). XI's "A1 false positive" therefore reflected a non-discriminating test distribution.

Arms, for complete parents:
- (a) On pairs where cf_E and cf_A1 have different decisions, *and* cf_E differs from the base's own decision,
  identity interchange in the E-subspace gives IIA_E ≥ 0.70. On all discriminating pairs, IIA_A1 ≤ base_A1 + 0.10.
  The subspace does not mediate perfect-only eligibility.
- (b) **C1 contrast.** In C1 models the same test gives IIA_A1 ≥ base_A1 + 0.15. The C1 "E" subspace mediates a
  perfect-only variable, so the label "E" is a false positive there.
- (c) **Mathematical prediction (XI Theorem X1′ → network).** For the VS target, an unconstrained linear map inside
  the E-subspace reaches KL ≤ 0.7 × the best projection (k = 1, 2, 3, or the full E swap), and its largest eigenvalue
  is ≥ 1.2 (non-projective).

**Development basis.**
- IIA_E,inf 0.79–0.85; full IIA_A1 − base 0.02–0.04; C1 IIA_A1 − base 0.26–0.39;
- VS (one development model, xP81, finished before the freeze; weak basis, labelled so): LIN KL 0.40 vs best projection 0.83 (ratio 0.48), largest eigenvalue 3.83. A second development model (yP130) was still running at the freeze; its result is reported post-freeze as development information only.

### P7 — Learning: the conjunction is used as soon as it exists (`dyn_mech.py`, `traj_alg.py`)

**Hypothesis.** The XI "representation before behaviour" ordering is **not** a readout-learning lag. The decision
uses the conjunction (use_hyb ≥ 0.05 nats, TRUE − CTRL) at the checkpoint where it emerges (full-space ratio ≥ 2).
The later "D2-closest-to-Bayes" step reflects the further A2 → B calibration of the decision.

**Alternative.** The readout lags (use onset at least 2 checkpoints after emergence).

Arms, for complete parents:
- (a) |checkpoint index(use onset) − index(emergence)| ≤ 1.
- (b) Use onset ≤ first checkpoint whose D2 behaviour is not A1-closest.
- (c) The D2-B switch comes strictly after use onset.

**Development basis.** (a) 9/9; (b) 9/9; (c) 9/9.

### P8 — Generalization: rule versus representation under two colluders (`adapt_c2.py`)

**Mathematics (`math_xii.py` M1–M2).**
- Independent copying: p ∝ Σ_g W_g L_g e_C(e_{−g}), with e_C the degree-C elementary symmetric polynomial. This is
  exact for all tested (G, C).
- The trained C = 1 readout implements e_1. C = 2 needs e_2 (an AND).
- The actual sampler copies one shared value. Its exact posterior uses the joint set indicator, which the per-group e
  statistic does not capture: KL 0.049–0.076 to e_C, and an accuracy gap of 4.6–6.1 points.

**Correction to Investigation XI.** XI's "true C = 2 Bayes" (genlib.posterior_brute) is the
independent-copying posterior. The exact posterior gives Bayes accuracy 0.80, not 0.75.

Arms, for complete parents (layers 0–1 frozen; 1500 steps; stream 71_100_000):
- (a) The native model applies the C = 1 rule: accuracy ≤ 0.40 and KL(C1-rule ‖ model) ≤ 0.05.
- (b) Re-training **only layer 2 + readout** reaches accuracy ≥ exact-Bayes − 0.03.
- (c) The same re-training with the E-subspace mean-patched also reaches ≥ exact-Bayes − 0.03. The correction does
  not require the eligibility representation; it can bypass it.
- (d) When E is available, the adapted readout uses it: E-dependence KL ≥ 0.1.

**Development basis (2 models):**
- native acc 0.333–0.334, KL_C1rule ≤ 0.006;
- adapt_l2 0.802–0.803, against exact Bayes 0.799;
- noE 0.798–0.802;
- E-dependence 0.67–0.94;
- a fresh layer 2 also reaches 0.798–0.803.

### Interpretation rules

- A prediction that **fails** is reported as falsified, with the failing units. Thresholds are never moved.
- An arm with missing data (a job failure) is **inconclusive**, not passed, and the job failure is reported.
