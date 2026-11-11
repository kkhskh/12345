# Training stage and approximation quality

> **Post-review [R2-4, R5-3, R5-4, R5-6].** §2 has been reworded: the approximation-error argument is exploratory and
> weaker than stated before. §5 has been added. Pre-review text: commit 9389496.

**Schedule.** Both curricula follow the same matched schedule: 4000 transfer steps, snapshots at 250/500/1000/2000/4000, and the
same optimiser and schedule. The P-count arm received the full long-transfer arm, as the mandate required.

**Measurement.** Every snapshot is measured on the same 6000 natural episodes for both the boundary statistics and quality.
Source: `logs/summary_tables.json` and `logs/confirmatory.json` → `quality_table`.

## 1. Trajectories (mean of 8 parents)

| arm @ step | KL to Bayes | acc | ECE | T* (KL-optimal temperature) | margin scale | slope Δm_M on Δm_B (F45) | B45-R1 | B34-R1 |
|---|---|---|---|---|---|---|---|---|
| T1 @250 | 0.139 | .701 | .018 | 1.00 | 1.01 | 0.50 | −0.135 | −0.207 |
| T1 @500 | 0.080 | .721 | .023 | 1.00 | 1.11 | 0.72 | −0.019 | −0.075 |
| T1 @1000 | 0.060 | .726 | .019 | 1.02 | 1.19 | 0.76 | −0.035 | −0.076 |
| T1 @2000 | 0.045 | .731 | .015 | 1.00 | 1.21 | 0.80 | −0.042 | −0.049 |
| T1 @4000 | 0.037 | .733 | .013 | 1.00 | 1.22 | 0.82 | −0.041 | −0.034 |
| P @250 | 0.444 | .511 | .010 | 0.96 | 0.39 | 0.01 | −0.343 | −0.456 |
| P @500 | 0.274 | .634 | .011 | 1.00 | 0.72 | 0.19 | −0.497 | −0.392 |
| P @1000 | 0.164 | .693 | .014 | 1.01 | 0.92 | 0.44 | −0.623 | −0.215 |
| P @2000 | 0.107 | .713 | .015 | 0.99 | 1.02 | 0.57 | −0.711 | −0.120 |
| P @4000 | 0.088 | .719 | .014 | 0.99 | 1.07 | 0.61 | **−0.739** | **−0.071** |

## 2. What the trajectories show

1. **The P-count deficit at the boundary grows with training while its quality improves.**
   - From step 250 to step 4000, P-count KL falls fivefold (0.444 → 0.088).
   - Its below-boundary deficit B34-R1 closes almost to the T1 level (−0.456 → −0.071).
   - Its boundary excess B45-R1 becomes *more* negative (−0.343 → −0.739).
   - A *uniform-shrinkage* approximation account predicts that both shrink together. That does not happen at 4→5.
     This argument is exploratory, not frozen, and limited in two ways [R2-4, R5-3]:
     - At step 250 the P-count arm barely responds to any edit (slope 0.01, accuracy .51), so its B45-R1 there sits near
       −Δm_B. Part of the "deepening" is the arm beginning to respond at all.
     - The other 4→5 regimes improve strongly over the same period. The P-count R3 excess falls from 1.92 to 0.10, and R2
       from 2.27 to 0.66. At step 4000 the P-count arm is *closer* to Bayes in R3 than T1 is (0.10 vs 0.34).
   - The pattern is therefore the textbook one of an approximator improving its average while its error on a small subgroup
     grows: R1 is about 3 % of natural decisions. It rules out uniform error, not a feature-limited approximation error.
2. **T1 converges at both.** Its B45-R1 and B34-R1 both approach about −0.04 from below.
3. **Global calibration is not the explanation.**
   - The KL-optimal temperature is 1.00 ± 0.02 in both arms at every stage after 250.
   - ECE is ≤ 0.023 throughout.
   - A one-parameter gain fitted to each model predicts B45-R1 ≈ −0.05 for P-count models, against −0.74 observed
     (FITTED_OPERATOR_IDENTIFIABILITY.md).
   - The pure-gain prediction for D-R1 has the opposite sign to the observed difference (Lemma B3 / T4(b)).

## 3. Quality-matched comparisons

**P4(i), frozen.**
- The target is the median P KL at step 4000, which is 0.072.
- Each T1 parent's snapshot closest in KL was selected using KL only: steps 250 (×3), 500 (×3) and 4000 (×2), with KL
  0.042–0.141.
- The P-count range is 0.029–0.212, so the ranges overlap.
- The difference is +0.678 (p = 7.8e-5).
- Early T1 snapshots (step 250, KL 0.06–0.10) have B45-R1 between −0.21 and −0.14, still far from P-count's −0.49 to −0.96.

**P4(ii), frozen ANCOVA.** Across all 80 snapshots, the arm coefficient at fixed KL is +0.648 (permutation over parents,
p = 1.5e-4).

**Stage-matched comparison.** Here the same KL means a *later* P-count stage than T1 stage. Compare P-count at step 2000
(KL 0.107) with T1 at step 250 (KL 0.139): B45-R1 is −0.711 for P-count and −0.135 for T1. The comparison does not hinge on
which snapshot one picks.

## 4. Limits

- **The KL control is weak.** An error of 0.74 on about 3 % of decisions adds only about 1e-3 to KL, so KL matching barely
  "sees" the effect [R5-6]. It shows that average quality does not explain the difference. It is not a strong control.

- **The 'training stage' explanation is excluded only within this schedule.** We cannot exclude that much longer transfer
  (≫ 4000 steps) would close the gap. The trajectory from 2000 to 4000 is flat to slightly diverging (−0.711 → −0.739),
  which gives no sign of closing.
- **Matching is observational.** KL-matching selects different steps for different T1 parents, so stage and quality are not
  independently manipulated. A causal claim about stage would need, for example, a P-count arm with frozen-layer
  re-initialisation or an objective intervention. That was not run (OPEN_QUESTIONS.md).
- **Output-space dependence.** The quality measures are averages, and a small-measure set (F45-R1 pairs) can carry large
  errors at fixed KL (Proposition 1, corrected). The quality controls therefore show that *average* quality does not
  explain the effect. They cannot show that no quality measure does.

## 5. What the dynamics do establish [R5-4]

- **The discount is acquired during transfer.**
  - The P-count pretraining objective contains no discount.
  - The R1 reversal is weak at step 250 and grows during T4 training under the *same* objective and layer-2 learner as T1.
  - Because there is no step-0 readout, the question of whether a P-count model already has the discount immediately
    after pretraining cannot be answered.
- **What separates the arms.** The T1 frozen features (together with the inherited readout) lead layer 2 to avoid the
  R1 error under the same learner, objective and budget; the P-count ones do not.
- **Interpretation.** This is a structured, curriculum-dependent approximation error, concentrated on an "explaining-away"
  subgroup. It is not "beyond approximation error" in the sense of being unrelated to approximation [R5-8].
