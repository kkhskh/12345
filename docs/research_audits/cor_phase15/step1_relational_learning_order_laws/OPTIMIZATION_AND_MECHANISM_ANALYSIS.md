# Optimisation and mechanism analysis

Everything in this document is **exploratory**: it is a post-scoring analysis whose plan was declared in FREEZE XV §6. The two exceptions are labelled. No result here changes a frozen verdict.

## 1. Learning signal (P5; `scripts/xv_gradsig.py`, `logs/expl/gradsig_*.json`). WITHDRAWN after review

**The measure.** The gate-attributable expected gradient is ‖E[∇L(true data) − ∇L(gate-randomised data)]‖, estimated as the norm of the mean over 16 batches of 256 episodes. It was divided by the per-batch noise, at initialisation and at t_S.

**Withdrawn** (Review B, finding 1; ADJUDICATION W1).
- E‖mean D‖² = ‖μ‖² + tr Σ / 16, so this SNR is about 1 even when the true signal is zero.
- The reviewer's null-against-null control gives a 5–95% range of 0.68–1.43. Every measured value (0.75–1.54) lies inside or at the edge of that band.
- The earlier claims are withdrawn:
  - "initial gate SNR tracks the arity";
  - the Spearman table (ρ = −0.80 at initialisation; this headline also replaced the declared t_S statistic, ρ = −0.43, and treated 16 seed-paired models as independent).
- The raw outputs are kept in `logs/expl/` as a record.

**Corrected statement** [expl]. The estimator detected no gate-attributable expected gradient at initialisation or at t_S in any model. P5 and H3 are **untested**. A future test needs:
- a debiased ‖μ‖² against a null-against-null control, with far more batches;
- awareness that a first-order expected-gradient measure is not the quantity that governs escape from a higher-order plateau.

## 2. What changes at the gate transition (Workstream D; `scripts/x_mech.py`, `logs/expl/mech_*.json`)

**Selection.** Seed 201 of each condition was to be analysed if that model acquired the gate. It did for A and B. C and D did not acquire the gate with seed 201, so the competent C_s204 and D_s203 were used instead (a deviation from the declared rule). The censored C_s201 at steps 3750–4000 was added as a contrast.

**Measures.**
- **Probe:** held-out-x₀ linear decodability of g at the evidence tokens (layer-2 input, "E2"), the readout at the layer-2 input ("R2") and the readout after layer 2 ("post2@R"; this site was measured but omitted from the first draft, Review C F2).
  - **The relevant baseline is not 0.5.** The agreement indicator [x₃ = x₄] alone predicts g at 0.883 when ν = 0.05 (A, D) and 0.740 when ν = 1/3 (B, C) [num; Review B F2, re-checked].
  - So a probe can reach these values from the agreement cue alone, without any gate variable.
- **Use:** flip transfer from a full residual patch at the evidence tokens at the layer-2 input ("useE2"), and at the gate tokens at the layer-1 input. The baseline with no information is the value before the transition (0.04 in A, about 0.30 in B and C, about 0.03 in D per Review B F3). Layer-1 gate-token patching ("use G1") reproduces the input edit itself, so it cannot show where the gate is computed (Review B F3). C_s204 has no checkpoint before its transition starts.

| model | checkpoint | gate following | probe E2 | probe R2 | use E2 | use G1 | probe post2@R | cue baseline |
|---|---|---|---|---|---|---|---|---|
| A_s201 (t_G 825) | 625 | 0.00 | 0.57 | 0.49 | 0.04 (baseline) | 0.04 | 0.85 | 0.88 |
| | 775 | 0.31 | 0.50 | 0.46 | 0.35 | 0.31 | 0.88 | 0.88 |
| | 825 | 0.90 | 0.58 | 0.38 | 0.90 | 0.87 | 0.97 | 0.88 |
| | 925 | 1.00 | 0.96 | 0.42 | 1.00 | 0.98 | 1.00 | 0.88 |
| B_s201 (t_G 500) | 300 | 0.00 | 0.38 | 0.46 | 0.30 (baseline) | 0.30 | 0.48 | 0.74 |
| | 450 | 0.35 | 0.28 | 0.40 | 0.52 | 0.47 | 0.51 | 0.74 |
| | 500 | 0.98 | 0.87 | 0.26 | 0.98 | 0.97 | 0.87 | 0.74 |
| C_s204 (t_G 2500) | 2250 | 0.25 | 0.65 | 0.61 | 0.48 | 0.48 | 0.70 | 0.74 |
| | 2500 | 1.00 | 0.85 | 0.60 | 1.00 | 1.00 | 0.97 | 0.74 |
| D_s203 (t_G 2250) | 2000 | 0.28 | 0.78 | **0.87** | 0.25 (baseline) | 0.31 | 0.91 | 0.88 |
| | 2250 | 0.51 | 0.87 | 0.88 | 0.54 | 0.54 | 0.95 | 0.88 |
| C_s201 (censored) | 3750 / 4000 | 0.00 | 0.62 | 0.48 | 0.29 (baseline) | 0.29 | 0.68 | 0.74 |

**Reading (corrected after review)** [expl, 5 models].
- **Before the transition,** no measured site carries gate information beyond the agreement-cue baseline:
  - A_s201 at post2@R: 0.85–0.88, against a baseline of 0.88;
  - D_s203 at post2@R: 0.87–0.91, against 0.88 (0.91 is within about 1.5 SE of the baseline);
  - C_s204 and the censored C_s201: 0.68–0.70, against 0.74;
  - B_s201: below the baseline.
  - Patching at the evidence tokens transfers nothing beyond baseline.
- **During and after the transition,** decodability rises above the baseline (0.97–1.00 at post2@R, 0.86–1.00 at E2), together with causal use through the evidence tokens.
- **So information beyond the cue and its use appear together,** in all four acquiring models.
  - The earlier "D_s203 exception" was the agreement cue, and is withdrawn.
  - Whether the gate is newly *represented* or newly *usable* cannot be decided with these probes, because a linear probe for g is contaminated by the cue.
- **The censored C_s201** is at the cue baseline, with no use of the gate.
- **What survives about the route:** after the transition, the gate information is carried at the evidence tokens by the layer-2 input and reaches the readout. Where it is computed is not established, because the layer-1 patch is uninformative.

**Does the mechanism explain the differences between variants?** Only partly.
- It shows *what* forms at the transition.
- It does not explain *why* arity-3 models form it so much later. The signal measure in §1 was withdrawn, so no measured quantity explains it.

## 3. Comparison of the learning theories

| hypothesis | prediction | outcome | status |
|---|---|---|---|
| H1, complexity first | gate later at higher arity; lower-order agreement stage in every condition | delay 8/8 (6 censored; about 4 independent replicates); stage 16/16, but entailed by censoring in 6 and lasting 3 snapshots in 3 of 4 B models | **frozen P3 verdict H1 stands**, but H1 is not robustly separated from RACE. P2 at arity 2 leans against H1's auxiliary 'no value effect' (post hoc, confounded). Not 'consistent with all frozen tests' (Review A F11) |
| H2, decision benefit | no arity effect at matched gain; stage only where the intermediate payoff is large; earlier gate with larger gain | arity effect large; stage also in B and C; earlier gate with larger gain at arity 2 (4/4, confounded) | **its stage pattern and its "no arity effect" claim are falsified**; its timing claim survives in arity 2 only, confounded |
| RACE (development) | B skips the stage | B shows a weak stage at threshold 0.5 | **falsified** at the frozen threshold; it holds at 0.6 |
| H3, optimisation signal | timing tracks the measured signal beyond the task labels | signal measure at its noise floor (withdrawn) | **untested** |
| H5, task-specific | XIV's skipped stage reproduced? | not in this family; the stage is present | suggests task-specificity, but XIV used a different agreement index, so not like-for-like (Review B F10) |

**Narrowest defensible statement** [frozen + exploratory; one architecture]. In this architecture and task family, models acquire:
1. order-1 source accuracy, with gate-free second-order reliance already rising (no separate source plateau in GATE; Review B F6);
2. gate-free second-order reliance (A_p; not specifically [x₃ = x₄], Review A F2);
3. then the gate, abruptly at arity 2. At arity 3, abruptness is unresolved: 250-step snapshots.

Raising the gate's arity from 2 to 3 at equal decision value delays or prevents acquiring it within the budget. The agreement stage is long when the gate is slow and brief when the gate comes quickly. At a = 2 the gate is itself pairwise, so 'lower order' here means decision order, not relation arity (Review A F10). Censoring is at about 2700–3000 effective steps under the decayed schedule (Review B F8). That fits a picture in which lower-decision-order structure is learned first and the higher-order relation needs much more optimisation per unit of value. It does not show that "benefit" alone sets the order.

## 4. Workstream F: the noisy gate (mathematics only; `logs/xv_noisy_math.json`)

Condition B is modified so that the evidence regime follows g only with probability γ.

| γ | Bayes NLL | gate gain (best order-2 rule minus Bayes) | KL of the order-3 optimum | KL of the order-4 optimum | G of the order-3 optimum |
|---|---|---|---|---|---|
| 1.00 | 0.288 | 0.277 | 0 (exact) | 0 | 1.00 |
| 0.95 | 0.362 | 0.204 | 0.0016 | 0.0012 | 0.96 |
| 0.85 | 0.454 | 0.111 | 0.0021 | 0.0016 | 0.89 |
| 0.75 | 0.512 | 0.054 | 0.0013 | 0.0010 | 0.85 |

**Reading** [num]:
- **Trust uncertainty makes the Bayes rule a mixture.** It then has full interaction order (5). No order ≤ 4 rule reaches it, although order 3 comes within 0.002 nats. The script fits orders 1–4; Review A verified that order 5 is exact (F12).
- **It shrinks the gate's value** sharply, from 0.277 to 0.054.
- So with an uncertain gate, both the "complexity" and the "value" of the relation change at once.

**No neural runs were made** (budget). Whether learning becomes gradual under trust uncertainty remains open.
