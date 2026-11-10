# Necessity without collateral damage, restoration, redundancy (mandate §7)

> **Post-review [R].** Overshoot reference corrected (natural marginalisation, ME2); P collateral claim withdrawn; redundancy qualified (ME3). Pre-review text: commit 8c6363e; adjudication: REVIEW_ADJUDICATION.md.

`scripts/necessity.py`; confirmation `logs/conf/necessity.json` (evaluation stream 12_450_000, 1000 episodes × 6 decisions),
calibration `logs/cal/necessity.json`. Own-question Bayes accuracy 0.577, full Bayes 0.727 on this stream.

**Operator.** On-support marginalisation: the count variable at the site is replaced, through the model's own encoder, by counts
of 8 generator draws of X_{-q} given X_q (exact conditional sampler), and the 8 output distributions are averaged. All written
values are encoder images of counts that occur with X_q; no subspace is mean-replaced. If the site's count variable carries the
relational information the model uses, the output should move to the **own-question posterior** (not just lose accuracy).

| model | clean acc | marg acc | drop of relational gain | KL(own‖·) reduction: marg | null (cross-pair counts) | rotated (norm-matched) | wrong site |
|---|---|---|---|---|---|---|---|
| P2 | 0.701 | 0.595 | 0.86 | **0.70** | 0.01 | 0.22 | 0.27 |
| P3 | 0.719 | 0.595 | 0.88 | **0.83** | 0.02 | 0.28 | 0.05 |
| P4 | 0.694 | 0.594 | 0.86 | **0.68** | 0.02 | −0.17 | 0.16 |
| T2 | 0.723 | 0.556 | 1.14 | **0.80** | 0.01 | 0.18 | 0.59 |
| T10 | 0.726 | 0.549 | 1.19 | **0.88** | 0.01 | 0.28 | 0.14 |
| T11 | 0.726 | 0.522 | 1.37 | **0.86** | 0.02 | 0.39 | 0.27 |

(KL reduction = 1 − KL(own‖output)/KL(own‖clean); 1 = output equals the own-question posterior.)

- **Specific contribution: strong, but the frozen threshold failed narrowly.** Marginalising the count variable moves every
  model 68–88 % of the way to the own-question posterior, while a matched-capacity wrong variable moves it 1–2 %. Prediction
  IV(a) required ≥ 70 % in every model and < 30 % for the rotated control; P4 (0.677) and T11's rotated control (0.386) missed.
  T11's rotated delta (same norms, random directions) moving the output 39 % toward the own-question posterior means part of T11's
  marginalisation effect is not direction-specific — a residual H4 element.
- **[R: ME2] Reference for overshoot.** Averaging each model's outputs over real resampled X_{-q} (natural input marginalisation,
  no activation edit) already gives drop fractions 1.02–1.14 in all six models: falling below own-question Bayes accuracy is the
  models' own behaviour. Against that reference T10 has no excess, T11 ≈ 0.25 unexplained excess, and T2's 1.14 is the own-evidence
  leakage of its encoder (refitted encoder removes it). For P, the norm-matched rotated control removes 0.59 (P3) and 1.14 (P4) of
  the accuracy gain although it does not move the output toward the own posterior: **"no collateral damage" is not shown for P**;
  P4's 0.677 is close to what natural marginalisation at that site reaches (0.711).
- ~~**Collateral damage.** P drop fractions 0.86–0.88 (no overshoot).~~ T drop fractions 1.14–1.37: T models end *below* own-question
  Bayes accuracy, i.e. the operator also removes 14–37 % of a gain that the own-question evidence alone would support. Since KL to
  the own posterior is small but non-zero (all models 0.04–0.08 nats after marginalisation, against 0.31–0.43 clean), the overshoot is a modest distortion of own-question processing, not
  arbitrary damage. Necessity of V is therefore shown for P without collateral damage, and for T with a bounded one.
- **Redundancy.** In T, marginalising every source token's own leave-k counts (all 36 tokens) gives the same result as q's six tokens
  only (0.80/0.88/0.86 in both): no redundant copy among other source tokens (withdrawn for T2, whose encoder leaks [R: ME]).
  The readouts of T models do carry part of the effect (union ceilings 0.94–0.96 vs 0.55–0.84 [R: ME3]). The wrong-site controls show P's source tokens and T's
  readout carry 5–59 % of the effect (largest T2 0.59, whose readout decodes n at 0.93): partial copies with partial causal role
  (Theorem D1 learned instance; D4's redundancy pattern is not observed).

**Span ablation and restoration.** Each site position's projection onto its encoder span is replaced by the natural mean, then
the true counts' encoder image is added back.

| | P2 | P3 | P4 | T2 | T10 | T11 |
|---|---|---|---|---|---|---|
| ablated acc | 0.565 | 0.532 | 0.514 | 0.552 | 0.446 | 0.481 |
| restored (true counts) | 0.697 | 0.715 | 0.656 | 0.669 | 0.544 | 0.572 |
| restored (another episode's counts) | 0.545 | 0.528 | 0.513 | 0.548 | 0.465 | 0.474 |
| restoration fraction | 0.97 | 0.98 | 0.79 | 0.68 | 0.35 | 0.37 |

In P, span ablation removes the relational response and true-count restoration returns it (wrong-episode restoration does not):
the restored response is the relational one (KL to full Bayes back to clean levels in P2, P3: 0.137 vs 0.136, 0.081 vs 0.092; P4 only partly: 0.219 vs 0.149). In T, span ablation drops below own-question
accuracy (0.45–0.55): the encoder span at a source token also carries own-report information, so the operator is not clean for T,
as predicted (IV(b) T part held; the P part failed narrowly at P4 0.785).

**Conclusion for §7** [R: ST2 — narrowness is symmetric: P2 passed IV(a) by 0.0006; T11's rotated 0.386 is not narrow]. The identified count variable makes a specific, on-support causal contribution to the relational decision
advantage in all six models, directed at the own-question posterior and far above matched nulls. "Uniquely necessary" is not
claimed: the operator is a replacement within the encoder image, T shows bounded overshoot, and T11 shows partial
direction-nonspecificity.
