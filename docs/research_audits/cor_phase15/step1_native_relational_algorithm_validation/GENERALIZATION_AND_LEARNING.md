# Generalization and learning (Investigation XII, Workstreams D and B5)

## 1. Two colluders: is the failure in the variable, the aggregation, or the readout?

### 1.1 Mathematics

See `CAUSAL_ABSTRACTION_AND_MATHEMATICS.md` §1.3 and XII-1/2 for the statements; the numbers are from
`math_xii.py`, M1–M2.

- **Trained rule (C = 1).** p ∝ Σ_g W_g L_g · e_1(e_{−g}), with e_1(e_{−g}) = Σ_{h≠g} e_h. This is degree-1
  aggregation of per-group eligibility.
- **Independent-copying C = 2.** The same formula with e_2(e_{−g}) = Π_{h≠g} e_h for G = 3: degree-2 aggregation, an
  AND across groups.
- **The actual sampler (shared copied value).** The exact posterior replaces Π e_h by J_H, which requires the two
  colluder groups to report *identical* values at every question. That is a between-group relation the per-group
  statistic e cannot express.

| quantity at G = 3, C = 2 | value |
|---|---|
| exact-Bayes accuracy | 0.799 |
| e-based product rule (= genlib posterior) accuracy | 0.747 |
| C = 1 rule accuracy | 0.333 (chance) |

**Correction to XI.** XI reported "Bayes 0.75" for C = 2. That figure is the independent-copying posterior.

### 1.2 Stage-separated adaptation (`adapt_c2.py`)

Layers 0–1, which hold the learned relational representation, are frozen. Only layer 2, the final LayerNorm and the
readout are re-trained, on C = 2 data: 1500 AdamW steps from a pool of 40k C = 2 episodes.

[dev] results:

| model | condition | accuracy | KL(exact ‖ ·) | KL(C1 rule ‖ ·) | E-dependence |
|---|---|---|---|---|---|
| xP81 full_c1 | native (no adaptation) | 0.334 | 1.48 | **0.005** | 0.02 |
| | **adapt layer 2** | **0.802** | **0.003** | 1.79 | 0.67 |
| | adapt layer 2, E-subspace mean-patched | 0.798 | 0.003 | 1.77 | 0 |
| | fresh layer 2 | 0.798 | 0.002 | 1.80 | 0.12 |
| wP140 full_c1 | native | 0.333 | 1.42 | 0.006 | 0.01 |
| | adapt layer 2 | 0.803 | 0.002 | 1.81 | 0.94 |
| | adapt, E patched | 0.802 | 0.003 | 1.77 | 0 |
| | fresh layer 2 | 0.803 | 0.002 | 1.79 | 0.47 |
| xP81 **C1** (pretrained layers 0–1, no conjunction) | native | 0.357 | 1.09 | 0.05 | 0.07 |
| | adapt layer 2 | 0.800 | 0.004 | 1.82 | 1.10 |
| | adapt, E patched | 0.802 | 0.006 | 1.82 | 0 |
| | fresh layer 2 | 0.772 | 0.063 | 1.60 | 0.34 |

Exact-Bayes accuracy is 0.799 on this evaluation set.

Findings:
1. **The failure is entirely downstream of layer 1.** Re-training the evidence-combination stage alone recovers
   exact-Bayes behaviour (accuracy 0.80, KL ≤ 0.004), while the frozen relational representation is kept.
2. **The repair does not need the eligibility subspace.** With the E coordinates mean-patched, the same adaptation
   reaches the same accuracy. When E is available, the adapted readout *does* use it (E-dependence 0.67–1.10). Use is
   not necessity.
3. **Repairing does not require the fine-tuned conjunction.** Pretrained, unrepaired layers 0–1 (C1) support the same
   repair when layer 2 is adapted from its trained state. A *fresh* layer 2 on the C1 prefix is worse at S = 3
   (accuracy 0.60 vs 0.975).
4. Accuracy at S = 3 (all three groups perfect and agreeing; 7.8% of episodes) is 0.975 for every adapted readout.
   The e statistic cannot distinguish the colluders there; cross-group identity can. The adapted readouts therefore
   read **cross-group equality**, information outside E.

**Interpretation.** Under the real two-colluder generator, the relevant relation is between groups, not within a
group. The repaired readout uses residual-stream information that the 6-dim eligibility subspace does not contain.
The XI framing, "the variable generalizes, the rule does not", is right for the *trained* rule. But the variable is
not sufficient for the new generator, and the repair bypasses it. The obstacle in the native model is
**computational, in the aggregation stage**: degree-1 versus joint aggregation. It is not informational, because the
required information is linearly available in resid2.

The fresh-cohort test is frozen prediction P8 (§3).

## 2. Learning dynamics: why the representation "precedes" behaviour

`dyn_mech.py` [dev, 9 models with saved checkpoints]. Per checkpoint:
- representation **magnitude**: the full-space interaction ratio;
- **use** of the conjunction by the decision, as TRUE − CTRL KL:
  - hybrid version: layer 2 applied to the additive completion of resid2;
  - natural version: layer 1 fed the xA input;
- **readout gain**: use per unit of squared magnitude;
- behaviour from Investigation X's D2 rule classification.

| model | emergence (ratio ≥ 2) | use onset (≥ 0.05 nats) | first non-A1 D2 step | D2-Bayes step |
|---|---|---|---|---|
| xP81 | 246 | 246 | 546 | 813 |
| yP130 | 546 | 366 | 813 | 1211 |
| yP131 | 1211 | 1211 | 1803 | 4000 |
| yP132 | 246 | 165 | 366 | 1211 |
| yP134 | 1211 | 1211 | 1803 | 1803 |
| wP140 | 366 | 366 | 813 | 1211 |
| wP141 | 546 | 813 | 813 | 1211 |
| wP142 | 366 | 366 | 366 | 813 |
| wP143 | 366 | 366 | 546 | 546 |

Findings:
- **No readout lag.** The decision uses the conjunction at the checkpoint where it first appears (9/9 within one
  checkpoint). Readout gain is highest at emergence and then *falls* while magnitude keeps growing. This rules out
  "the readout learns to use an already-present representation later".
- **What actually comes later is a change of behavioural class.** Investigation X's D2 classifier moves A1 → A2 →
  B. The A2 (threshold-4) phase sits between use onset and the Bayes classification. The use onset precedes or equals
  the first non-A1 step in 9/9 models.
- The XI ordering "representation before behaviour" is therefore an ordering between a *local* internal measure and
  a *global* behavioural classification. The decision's use of the conjunction is **not** delayed.
- The remaining delay is the calibration that separates B from A2. The internal measure tested here does not account
  for it. Neither does the n = 4 conjunction leak (CTRL use stays ≤ 0.01 throughout), so the cause of the A2 → B step
  remains open.

The fresh-cohort test is frozen prediction P7 (§3).

## 3. Fresh-cohort results

See `FRESH_CONFIRMATION_AND_STATISTICS.md` Part II; filled after the confirmation.
