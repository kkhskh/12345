# Selective rescue

> **Post-review [R2–R5].** Corrected: the 'Bayes level' claim is withdrawn; the n and random controls are re-described; a generic task-relevant adapter (injna, development) and an extra-training control are added; m's near-oracle character on the deciding subset is stated. Pre-review text: commit 76a9d60.


**Rescue target** (mandate §9.1). Move the P-count R1 response toward Bayes without damaging:

- R2 and R3;
- F34/F23;
- the Bayes-null controls;
- KL and accuracy.

**Constraint.** The repair may use only information available at inference.

## 1. The repair and why it was chosen

**The repair: m-injection (C1_injm).**

| aspect | specification |
|---|---|
| base setup | the original transfer: frozen embeddings and layers 0–1, inherited W_A and final LN, fresh layer 2, 4000 steps, generator-label cross-entropy |
| addition | a zero-initialised trainable linear map, from the one-hot eligibility vector to the six readout-token inputs of layer 2 |
| eligibility vector | m_g = #{other groups that are perfect and agree at q}, for each of the 3 groups, one-hot over 0/1/2 |
| inference-time input | m is a fixed deterministic function of the episode X: the counts and q-agreement that the input already contains |
| what it never receives | the posterior, the answer, a regime label or the latent roles |
| supervision | the generator label only |

**Why m, specifically.** Not "whatever worked". The choice follows three prior results:

1. **Proposition 1.** m_g is the only cross-group factor in the exact rule.
2. **Theorem 1.** Without some cross-group factor, R1 and R3 must share a sign. That is exactly the P-count error pattern: R1
   and R3 both discounted.
3. **Development evidence.**
   - **Oracle readers:** given counts and q-reports but not m, an MLP reaches KL 0.002 yet R1 −0.28 at 4000 steps. Given m,
     the same MLP is exact. Without m, it needs ≈16,000 steps.
   - **Probes:** "another group is perfect" is linearly accessible at the q-source tokens of T1 frozen states (0.98–0.997) but
     barely above the majority rate in P-count states (0.912–0.952 vs majority 0.912).

m was therefore predicted to be the missing *accessible* statistic, and was chosen before any confirmation lineage was evaluated.

## 2. Comparators

All comparators use the same parent, the same layer-2 initialisation, the same data stream and the same 4000 steps.

| class | method | role |
|---|---|---|
| A: representation supplement | **m-injection** (the repair) | mechanism-motivated |
| A: control | **n-injection**: one-hot leave-q counts, separable information already in the state, capacity comparable (18 vs 9 inputs) | predicted to *not* help (Theorem 1) |
| A: control | **random injection**: 9 fixed pseudo-random binary episode features | capacity control |
| generic data | **W_bal**: loss weights equalising the natural decision classes (one perfect; two perfect agreeing; two perfect disagreeing) | generic learning-dynamics baseline |
| development-only, B: downstream capacity | Bsoft (two fresh blocks), Csoft/Esrcsoft (MLP readers), Bayes soft targets | sufficiency and accessibility, SUFFICIENCY.md |
| development-only, B: route removal | _rtemb (readout-token input replaced by its embedding-level state) | tests the reconstruction's route |
| development-only, C: readout | C2/C3/C5 (fresh W_A and/or fresh LN) | READOUT_AND_LAYER2_FACTORIAL.md |
| development-only, generic | C1 for 12,000 steps (3× budget); Asoft 12,000 | READOUT_AND_LAYER2_FACTORIAL.md / LEARNING_DYNAMICS |

## 3. Confirmation results

Source: `logs/confirmatory7.json` (FROZEN_CONFIRMATION_PROTOCOL.md). The cohort is 8 new P-count lineages; values are means.

| condition | E_R1 | Δm_R1 | E_R2 | E_R3 | \|E\| R2∪R3 | E_F34-R1 | F0 \|Δm\| | KL | acc | E_S45-R1 (nested) |
|---|---|---|---|---|---|---|---|---|---|---|
| exact Bayes | 0 | +0.23 | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |
| C1 (original) | −0.724 | −0.495 | +0.521 | −0.068 | 1.228 | −0.007 | 0.144 | 0.070 | 0.725 | −0.752 |
| **C1_injm (repair)** | **−0.029** | **+0.200** | −0.158 | +0.099 | **0.514** | −0.057 | **0.086** | **0.039** | **0.731** | **−0.029** |
| C1_injn | −1.361 | −1.132 | −0.102 | −0.714 | 0.746 | +0.032 | 0.075 | 0.034 | 0.733 | −1.388 |
| C1_injrand | −0.743 | −0.514 | +0.525 | −0.085 | 1.227 | −0.014 | 0.151 | 0.070 | 0.725 | −0.771 |
| W_bal | +0.176 | +0.405 | **+1.462** | **+1.235** | 1.311 | −0.004 | 0.133 | **0.149** | **0.691** | +0.133 |

**Selective-repair score.** These are the frozen thresholds, applied to the other conditions descriptively:

| condition | selective successes out of 8 |
|---|---|
| m-injection | **7** (frozen P3) |
| W_bal | 0 |
| n-injection | 1 |
| random injection | 0 |

## 4. Interpretation

1. **m-injection is a selective behavioural rescue.** It is **not** a return to Bayes.
   - The signed mean E goes from −0.72 to −0.03, but three parents overshoot (+0.32 to +0.43) and five undershoot (−0.13 to
     −0.37).
   - Mean |E_R1| falls from 0.72 to 0.30, a 58 % reduction. Per-pair |E| falls from 0.82 to 0.51.
   - The raw Δm goes from −0.50 to +0.20, positive in 5/8 parents.
   - The pre-review phrase "to the Bayes level" is **withdrawn** (reviews R4-1, R5-7).
   - It does not damage R2/R3; the combined |E| there falls from 1.23 to 0.51.
   - It improves KL (0.070 → 0.039) and accuracy (0.725 → 0.731).
   - It also fixes the nested 3→4→5 test (S45-R1 E −0.75 → −0.03).
   - It does this on independent lineages, under the frozen protocol.
2. **The effect is specific to the precomputed conjunction m.**
   - **Random injection** changes nothing (Δ ≈ 0). It is a weak control: its features take at most 7 distinct patterns per
     episode (a mod-7 hash), so it controls only for an added pathway, not for information (reviews R2-5, R3-4, R5-1).
   - **n-injection** (counts of all three groups) makes R1 worse in 7/8 parents (mean Δ −0.64) while improving KL. The
     pre-review label "separable information" was imprecise. The other groups' count-5 indicators determine m in 95–97 % of
     cells, so n *contains* the cross-group information. What it lacks is the conjunction with q-agreement in computed form
     (reviews R3-1, R5-2).
   - **Post-review generic task-relevant adapter `injna`** (development, 2 parents). It supplies every ingredient of m (all
     counts plus every group's q-agreement), 21 inputs, but not the conjunction. Result: P80 **−2.26**, P81 −0.67 (originals
     −0.90 / −0.72), with KL improved.

   **Corrected reading.** What rescues R1 is the *precomputed conjunction* "number of other perfect groups that agree at q".
   Giving the learner the ingredients, even linearly, does not help, and can make R1 worse. This fits the Theorem-1 picture: the
   learner settles on a near-separable rule over whatever features are easiest. It is a statement about learnability of the
   conjunction within this budget and architecture. It is not a statement about what is absent.
3. **Generic data balancing is not a selective substitute.** W_bal is not fully "generic": its decision classes are defined by
   the same relation, and its weights (×11.9, ×6.0) were untuned and had no development run before the freeze (reviews R3-5,
   R4-4, R5-8).
   - W_bal fixes the R1 sign (+0.18) but over-credits perfect groups everywhere: R2 +1.46, R3 +1.24.
   - It roughly doubles KL and costs 3.4 points of accuracy.
   - Reweighting changes the trade-off along the same separable family. It does not supply the interaction. Mandate outcome F
     ("generic training explains") is therefore **not** supported for selective correction. W_bal *does* show that the R1 sign
     responds to the loss composition (LEARNING_DYNAMICS_AND_ERROR_ORIGIN.md).
4. **m is close to an oracle on the subset that decides R1 (review R3-2).**
   - On two-perfect decisions (8 % of decisions; every R1 pair is one), weighting groups by [m > 0] matches the Bayes choice in
     97.4 % of cases.
   - m alone, without counts, gives KL 0.128, accuracy 0.693 and R1 ≈ 0. With counts, [m > 0]·RHO^n is essentially exact
     (review R5-4).
   - The repair therefore converts the task into a form the learner fits easily. It respects the no-oracle rule (m is a fixed
     function of X, with labels-only training), but it is a hand-derived, generator-specific feature, valid for pc = 1 and ρ = 1.
5. **What the rescue does not show.**
   - It is a behavioural rescue by supplying a computed relational feature. It does not identify an internal operation of the
     original models that computes m wrongly.
   - The original models' reconstruction is in FULL_LAYER2_CAUSAL_RECONSTRUCTION.md.
   - Supplying m is a hand-specified, generator-derived feature. The rescue therefore shows that *the missing piece is the
     cross-group eligibility relation*. It does not show that a learned network would find it unaided.
6. **Failure case.** yP96 improved by only +0.146 (below the 0.20 success threshold) and stayed negative. Its original error
   was among the smallest (−0.45).

## 5. Post-review development controls (not frozen)

All runs: development, P80 unless stated. Source: `checkpoints/dev`.

| control | E_R1 | KL | note |
|---|---|---|---|
| original transfer, 4000 steps | −0.895 | 0.104 | |
| **extra training, 3× budget** (12,000 steps, same schedule shape) | −1.029 @4000, −0.698 @8000, **−0.682 @12,000** | 0.067 | wrong sign persists: Δm −0.45 |
| generic task-relevant adapter injna | **−2.264** (P81: −0.668) | 0.049 (P81: 0.015) | |
| m-injection on a **T1** parent (T70) | −0.115 (original −0.098) | 0.005 | no damage; R2/R3 E +0.04 / +0.07, original +0.24 / +0.26 |

**Not run.** Low-rank (LoRA) layer-2 update, logit recalibration, natural R1 up-weighting (W_r1up; coded, never run), and a
learned conjunction-capable adapter (for example an MLP on counts plus q-agreement). These remain open (OPEN_SCIENTIFIC_PROBLEMS.md).
