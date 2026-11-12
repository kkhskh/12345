# Selective rescue

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

1. **m-injection is a selective behavioural rescue.**
   - It moves R1 to the Bayes level on average: E −0.72 → −0.03, raw Δm −0.50 → +0.20.
   - It does not damage R2/R3; the combined |E| there falls from 1.23 to 0.51.
   - It improves KL (0.070 → 0.039) and accuracy (0.725 → 0.731).
   - It also fixes the nested 3→4→5 test (S45-R1 E −0.75 → −0.03).
   - It does this on independent lineages, under the frozen protocol.
2. **The effect is specific to the relational statistic.**
   - Random injection changes nothing: Δ ≈ 0 in every parent.
   - Injecting the *separable* counts makes R1 **worse** in 7/8 parents (mean Δ −0.64), although it improves KL even more
     (0.034).
   - This is the prediction of Theorem 1 in learning form. Making separable count information easier to use pushes the learner
     to a better-on-average separable discount, which has the wrong R1 sign.
3. **Generic data balancing is not a selective substitute.**
   - W_bal fixes the R1 sign (+0.18) but over-credits perfect groups everywhere: R2 +1.46, R3 +1.24.
   - It roughly doubles KL and costs 3.4 points of accuracy.
   - Reweighting changes the trade-off along the same separable family. It does not supply the interaction. Mandate outcome F
     ("generic training explains") is therefore **not** supported for selective correction. W_bal *does* show that the R1 sign
     responds to the loss composition (LEARNING_DYNAMICS_AND_ERROR_ORIGIN.md).
4. **What the rescue does not show.**
   - It is a behavioural rescue by supplying a computed relational feature. It does not identify an internal operation of the
     original models that computes m wrongly.
   - The original models' reconstruction is in FULL_LAYER2_CAUSAL_RECONSTRUCTION.md.
   - Supplying m is a hand-specified, generator-derived feature. The rescue therefore shows that *the missing piece is the
     cross-group eligibility relation*. It does not show that a learned network would find it unaided.
5. **Failure case.** yP96 improved by only +0.146 (below the 0.20 success threshold) and stayed negative. Its original error
   was among the smallest (−0.45).
